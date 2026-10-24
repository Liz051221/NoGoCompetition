from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
import time
from collections import deque
from pathlib import Path
from queue import Empty
from typing import Callable, Iterable

from .alphazero import (
    augment_samples_with_symmetries,
    describe_samples,
    generate_self_play_samples,
    parameter_fingerprint,
    read_samples,
    samples_fingerprint,
    sanitize_policy_targets,
    sanitize_tactical_policy_targets,
    train_tiny_value_head,
    train_torch,
    write_samples,
    write_sample_parts,
)
from .mcts import NeuralMCTS
from .network import TinyPolicyValueNet
from .protocol import EngineConfig, SAUNoGoEngine
from .rules import BLACK, board_to_lines, color_name, index_to_coord, is_legal_move, legal_moves, new_board, play_move


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nogo-ai")
    sub = parser.add_subparsers(dest="command")

    engine = sub.add_parser("engine", help="run the SAU protocol engine")
    add_engine_args(engine)
    engine.add_argument("--name", default="PyNNNoGo", help="engine name reported to the platform")
    engine.add_argument("--verbose", action="store_true", help="write diagnostics to stderr")
    engine.add_argument("--game-time-limit", type=float, default=900.0, help="total decision-time budget per side")
    engine.add_argument("--time-reserve", type=float, default=30.0, help="seconds held back from the game budget")
    engine.set_defaults(simulations=200000, time_limit=30.0)

    bench = sub.add_parser("benchmark", help="time a single opening search")
    add_engine_args(bench)

    self_play = sub.add_parser("self-play", help="run self-play games")
    add_engine_args(self_play)
    self_play.add_argument("--games", type=int, default=1)
    self_play.add_argument("--max-plies", type=int, default=120)

    data = sub.add_parser("generate-data", help="generate AlphaZero self-play training data")
    add_engine_args(data)
    data.add_argument("--games", type=int, default=2)
    data.add_argument("--max-plies", type=int, default=120)
    data.add_argument("--fixed-simulations", action="store_true", help="use simulation/node limits without wall deadlines for action searches")
    data.add_argument("--label-fixed-simulations", action="store_true", help="use simulation/node limits without wall deadlines for clean label searches")
    data.add_argument("--require-terminal", action="store_true", help="reject incomplete trajectories instead of assigning artificial outcome labels")
    data.add_argument("--temperature-moves", type=int, default=12)
    data.add_argument(
        "--opening-temperature",
        type=float,
        default=1.15,
        help="temperature used only when sampling the first temperature-moves actions",
    )
    data.add_argument(
        "--opening-uniform-mix",
        type=float,
        default=0.05,
        help="uniform mixture over all legal moves during opening sampling",
    )
    data.add_argument("--dirichlet-alpha", type=float, default=0.3, help="root Dirichlet alpha for self-play")
    data.add_argument("--dirichlet-epsilon", type=float, default=0.25, help="fraction of root exploration noise")
    data.add_argument(
        "--clean-label-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use an independent noise-free search to generate training policy labels",
    )
    data.add_argument(
        "--label-simulations",
        type=int,
        default=None,
        help="simulation budget for clean labels (defaults to --simulations)",
    )
    data.add_argument(
        "--label-time-limit",
        type=float,
        default=None,
        help="time budget for clean labels (defaults to --time-limit)",
    )
    data.add_argument("--output", default="data/selfplay.jsonl")
    data.add_argument("--workers", type=int, default=1, help="parallel self-play worker processes")
    data.add_argument(
        "--shared-inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use one batched GPU inference process for all self-play workers",
    )
    data.add_argument("--inference-server-batch-size", type=int, default=64)
    data.add_argument("--inference-server-wait-ms", type=float, default=1.0)
    data.add_argument("--quiet", action="store_true", help="only print the final summary")
    data.add_argument("--log", default=None, help="optional progress log file")
    data.add_argument("--result-json", default=None, help="optional machine-readable run statistics")

    train = sub.add_parser("train", help="train a policy-value model from self-play data")
    train.add_argument("--data", default="data/selfplay.jsonl")
    train.add_argument("--data-weight", type=float, default=1.0)
    train.add_argument("--replay-data", action="append", default=[], help="extra JSONL sample file to mix in")
    train.add_argument(
        "--replay-weight",
        action="append",
        type=float,
        default=[],
        help="weight for the corresponding --replay-data entry",
    )
    train.add_argument(
        "--replay-max-samples",
        action="append",
        type=int,
        default=[],
        help="deterministically cap samples from the corresponding replay file",
    )
    train.add_argument("--output", default="weights/nogo_torch.pt")
    checkpoint_group = train.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint", default=None, help="load model weights for fine-tuning")
    checkpoint_group.add_argument(
        "--resume-training",
        default=None,
        help="resume model, optimizer, scheduler, AMP and RNG state from an epoch checkpoint",
    )
    train.add_argument(
        "--distill-checkpoint",
        default=None,
        help="keep the candidate close to the checkpoint policy/value during fine-tuning",
    )
    train.add_argument("--backend", choices=["torch", "tiny"], default="torch")
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1.0e-3)
    train.add_argument("--minimum-learning-rate", type=float, default=0.0)
    train.add_argument("--validation-fraction", type=float, default=0.0)
    train.add_argument(
        "--validation-disjoint-positions", action="store_true",
        help="exclude validation positions equivalent to training inputs under board symmetry and player-relative colors",
    )
    train.add_argument("--checkpoint-every", type=int, default=0, help="save an epoch snapshot every N epochs")
    train.add_argument("--device", default="auto")
    train.add_argument(
        "--data-transfer",
        choices=["auto", "pinned", "gpu"],
        default="auto",
        help="training tensor placement: auto-select, pinned CPU memory, or preload to GPU",
    )
    train.add_argument("--channels", type=int, default=64)
    train.add_argument("--blocks", type=int, default=4)
    train.add_argument("--input-planes", type=int, default=4)
    train.add_argument("--policy-channels", type=int, default=2)
    train.add_argument("--value-channels", type=int, default=1)
    train.add_argument("--value-hidden", type=int, default=None)
    train.add_argument("--norm-type", choices=["batch", "group"], default="batch")
    train.add_argument("--policy-head", choices=["legacy", "spatial"], default="legacy")
    train.add_argument("--policy-weight", type=float, default=1.0)
    train.add_argument("--value-weight", type=float, default=1.0)
    train.add_argument("--winner-policy-weight", type=float, default=1.0)
    train.add_argument("--loser-policy-weight", type=float, default=1.0)
    train.add_argument(
        "--balance-policy-colors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="balance policy-loss weight between black and white side-to-move samples",
    )
    train.add_argument("--distill-policy-weight", type=float, default=0.0)
    train.add_argument("--distill-value-weight", type=float, default=0.0)
    train.add_argument(
        "--freeze-batchnorm-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep BatchNorm running statistics fixed during fine-tuning",
    )
    train.add_argument(
        "--sanitize-tactical-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="remove immediate-loss moves from late-game policy targets",
    )
    train.add_argument(
        "--skip-policy-sanitize",
        action="store_true",
        help="skip policy target validation/normalization when loading an already preprocessed dataset",
    )
    train.add_argument(
        "--relabel-endgames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="replace late-game value labels with exact minimax results",
    )
    train.add_argument("--relabel-endgame-moves", type=int, default=12)
    train.add_argument("--relabel-endgame-nodes", type=int, default=200000)
    train.add_argument("--relabel-endgame-weight", type=float, default=2.0)
    train.add_argument("--relabel-endgame-workers", type=int, default=1)
    train.add_argument(
        "--preprocessed-output",
        default=None,
        help="save weighted, sanitized and relabelled samples for exact resume",
    )
    train.add_argument("--seed", type=int, default=None)
    symmetry_group = train.add_mutually_exclusive_group()
    symmetry_group.add_argument(
        "--augment-symmetries",
        action="store_true",
        help="expand samples with all 8 board symmetries before training",
    )
    symmetry_group.add_argument(
        "--random-symmetries",
        action="store_true",
        help="apply one random board symmetry per sample and epoch on the GPU",
    )
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="use CUDA mixed precision")
    train.add_argument("--quiet", action="store_true", help="only print the final summary")
    train.add_argument("--log", default=None, help="optional progress log file")

    upgrade = sub.add_parser(
        "upgrade-model",
        help="expand a checkpoint without changing its policy head while preserving its learned function",
    )
    upgrade.add_argument("--checkpoint", required=True)
    upgrade.add_argument("--output", required=True)
    upgrade.add_argument("--input-planes", type=int, default=5)
    upgrade.add_argument("--policy-channels", type=int, default=4)
    upgrade.add_argument("--value-channels", type=int, default=4)
    upgrade.add_argument("--value-hidden", type=int, default=192)
    upgrade.add_argument("--norm-type", choices=["batch", "group"], default="batch")
    upgrade.add_argument("--policy-head", choices=["legacy", "spatial"], default="legacy")

    arena = sub.add_parser("arena", help="pit two engines against each other")
    arena.add_argument("--games", type=int, default=20)
    arena.add_argument("--workers", type=int, default=1, help="parallel arena game processes")
    arena.add_argument("--max-plies", type=int, default=120)
    arena.add_argument("--a-backend", choices=["auto", "tiny", "torch"], default="tiny")
    arena.add_argument("--a-model", default=None)
    arena.add_argument("--a-name", default="A")
    arena.add_argument("--a-device", default="auto")
    arena.add_argument("--a-channels", type=int, default=64)
    arena.add_argument("--a-blocks", type=int, default=4)
    arena.add_argument("--a-inference-amp", action=argparse.BooleanOptionalAction, default=False)
    arena.add_argument("--a-inference-tf32", action=argparse.BooleanOptionalAction, default=False)
    arena.add_argument("--b-backend", choices=["auto", "tiny", "torch"], default="tiny")
    arena.add_argument("--b-model", default=None)
    arena.add_argument("--b-name", default="B")
    arena.add_argument("--b-device", default="auto")
    arena.add_argument("--b-channels", type=int, default=64)
    arena.add_argument("--b-blocks", type=int, default=4)
    arena.add_argument("--b-inference-amp", action=argparse.BooleanOptionalAction, default=False)
    arena.add_argument("--b-inference-tf32", action=argparse.BooleanOptionalAction, default=False)
    arena.add_argument("--simulations", type=int, default=200000)
    arena.add_argument(
        "--fixed-simulations", action="store_true",
        help="ignore move deadlines; use simulation and endgame-node budgets (diagnostic only)",
    )
    arena.add_argument(
        "--opening-plies", type=int, default=0,
        help="seeded random legal opening shared by each color-swapped pair",
    )
    arena.add_argument("--time-limit", type=float, default=30.0)
    arena.add_argument("--game-time-limit", type=float, default=900.0)
    arena.add_argument("--time-reserve", type=float, default=30.0)
    arena.add_argument("--root-tactical-weight", type=float, default=0.0)
    arena.add_argument("--endgame-moves", type=int, default=16)
    arena.add_argument("--endgame-nodes", type=int, default=200000)
    arena.add_argument("--endgame-time-fraction", type=float, default=0.35,
                       help="fraction of each move budget reserved for exact endgame search")
    arena.add_argument("--cpuct", type=float, default=1.45)
    arena.add_argument("--inference-batch-size", type=int, default=32)
    arena.add_argument("--a-inference-batch-size", type=int, default=None,
                       help="A-side MCTS batch size; defaults to --inference-batch-size")
    arena.add_argument("--b-inference-batch-size", type=int, default=None,
                       help="B-side MCTS batch size; defaults to --inference-batch-size")
    arena.add_argument("--seed", type=int, default=None)
    arena.add_argument(
        "--paired-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse one stochastic-search seed for each color-swapped game pair",
    )
    arena.add_argument("--quiet", action="store_true")
    arena.add_argument("--log", default=None)
    arena.add_argument("--result-json", default=None, help="write machine-readable arena summary")
    arena.add_argument(
        "--progress-json",
        default=None,
        help="atomic per-game checkpoint; defaults beside --result-json or --log",
    )
    arena.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume compatible completed games from the Arena progress checkpoint",
    )

    save_model = sub.add_parser("save-model", help="write the default JSON weights")
    save_model.add_argument("--path", default="weights/nogo_tiny_default.json")

    return parser


def add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=str, default=None, help="optional JSON model path")
    parser.add_argument("--backend", choices=["auto", "tiny", "torch"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--inference-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inference-tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--simulations", type=int, default=160)
    parser.add_argument("--time-limit", type=float, default=1.2)
    parser.add_argument("--cpuct", type=float, default=1.45)
    parser.add_argument("--root-tactical-weight", type=float, default=0.0)
    parser.add_argument("--endgame-moves", type=int, default=16)
    parser.add_argument("--endgame-nodes", type=int, default=200000)
    parser.add_argument("--endgame-time-fraction", type=float, default=0.35,
                        help="fraction of each move budget reserved for exact endgame search")
    parser.add_argument("--seed", type=int, default=None)


def _file_sha256(path: str | Path | None) -> str | None:
    """Hash a model file once so self-play resumes cannot mix weights."""
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_network(
    model: str | None,
    backend: str = "auto",
    device: str = "auto",
    channels: int = 16,
    blocks: int = 1,
    inference_amp: bool = False,
    inference_tf32: bool = False,
) -> object:
    if backend in {"auto", "torch"} and model and model.lower().endswith((".pt", ".pth")):
        try:
            from .torch_model import TorchPolicyValueNet

            return TorchPolicyValueNet(
                checkpoint=model, device=device, channels=channels, blocks=blocks,
                inference_amp=inference_amp,
                inference_tf32=inference_tf32,
            )
        except RuntimeError:
            if backend == "torch":
                raise
    if backend == "torch":
        from .torch_model import TorchPolicyValueNet

        return TorchPolicyValueNet(
            checkpoint=model, device=device, channels=channels, blocks=blocks,
            inference_amp=inference_amp,
            inference_tf32=inference_tf32,
        )
    if model:
        return TinyPolicyValueNet.load(model)
    return TinyPolicyValueNet()


def _warm_network(network: object) -> None:
    """Pay CUDA initialization costs before timed search or arena play."""
    device = str(getattr(network, "device", "cpu"))
    if not device.startswith("cuda"):
        return
    predict = getattr(network, "predict")
    board = new_board()
    for _ in range(3):
        predict(board, BLACK)
    torch = getattr(network, "torch", None)
    if torch is not None:
        torch.cuda.synchronize()


def make_searcher(args: argparse.Namespace) -> NeuralMCTS:
    return NeuralMCTS(
        network=load_network(
            args.model, args.backend, args.device, args.channels, args.blocks,
            args.inference_amp, args.inference_tf32,
        ),  # type: ignore[arg-type]
        simulations=args.simulations,
        time_limit=args.time_limit,
        cpuct=args.cpuct,
        seed=args.seed,
        root_tactical_weight=args.root_tactical_weight,
        endgame_moves=args.endgame_moves,
        endgame_nodes=args.endgame_nodes,
        endgame_time_fraction=args.endgame_time_fraction,
        inference_batch_size=args.inference_batch_size,
    )


def make_searcher_from_parts(
    model: str | None,
    backend: str,
    device: str,
    channels: int,
    blocks: int,
    simulations: int,
    time_limit: float,
    cpuct: float,
    seed: int | None,
    inference_amp: bool = False,
    inference_tf32: bool = False,
    inference_batch_size: int = 32,
    endgame_moves: int = 16,
    endgame_nodes: int = 200000,
    endgame_time_fraction: float = 0.35,
    root_tactical_weight: float = 0.0,
) -> NeuralMCTS:
    return NeuralMCTS(
        network=load_network(
            model, backend, device, channels, blocks, inference_amp, inference_tf32
        ),  # type: ignore[arg-type]
        simulations=simulations,
        time_limit=time_limit,
        cpuct=cpuct,
        seed=seed,
        root_tactical_weight=root_tactical_weight,
        endgame_moves=endgame_moves,
        endgame_nodes=endgame_nodes,
        endgame_time_fraction=endgame_time_fraction,
        inference_batch_size=inference_batch_size,
    )


def run_engine(args: argparse.Namespace) -> int:
    engine = SAUNoGoEngine(
        make_searcher(args),
        EngineConfig(
            name=args.name,
            verbose=args.verbose,
            game_time_limit=args.game_time_limit,
            time_reserve=args.time_reserve,
        ),
    )
    engine.run()
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    board = new_board()
    searcher = make_searcher(args)
    _warm_network(searcher.network)
    result = searcher.search(board, BLACK)
    move = index_to_coord(result.move) if result.move is not None else "none"
    print(
        "config "
        f"backend={args.backend} model={args.model or 'default'} device={getattr(searcher.network, 'device', 'cpu')} "
        f"simulations={args.simulations} time_limit={args.time_limit:.3f}s cpuct={args.cpuct:.2f}"
    )
    print(
        f"best move={move} confidence={_percent(result.confidence)} margin={_percent(result.margin)} "
        f"value={result.value:+.3f} win_est={_percent(_value_to_winrate(result.value))} "
        f"visits={result.visits} reason={result.reason} elapsed={_format_seconds(result.elapsed)}"
    )
    average_batch = searcher.inference_positions / max(1, searcher.inference_calls)
    print(
        f"inference calls={searcher.inference_calls} positions={searcher.inference_positions} "
        f"avg_batch={average_batch:.2f} cache_hits={searcher.prediction_cache_hits} "
        f"elapsed={_format_seconds(searcher.inference_seconds)}"
    )
    print("top candidates:")
    print("  move visits   share       q   prior")
    for candidate in result.top_candidates:
        print(
            f"  {index_to_coord(candidate.move):>4} {candidate.visits:>6} "
            f"{_percent(candidate.visit_share):>7} {candidate.q:>+7.3f} {_percent(candidate.prior):>7}"
        )
    return 0


def run_self_play(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    net = load_network(
        args.model, args.backend, args.device, args.channels, args.blocks,
        args.inference_amp, args.inference_tf32,
    )
    black_wins = white_wins = 0
    for game in range(1, args.games + 1):
        board = new_board()
        to_play = BLACK
        moves: list[str] = []
        winner = -to_play
        searcher = NeuralMCTS(
            network=net,
            simulations=args.simulations,
            time_limit=args.time_limit,
            cpuct=args.cpuct,
            seed=rng.randrange(1 << 30),
            root_tactical_weight=args.root_tactical_weight,
            endgame_time_fraction=args.endgame_time_fraction,
            inference_batch_size=args.inference_batch_size,
        )
        for _ply in range(args.max_plies):
            if not legal_moves(board, to_play):
                winner = -to_play
                break
            result = searcher.search(board, to_play)
            move = result.move
            if move is None or not is_legal_move(board, to_play, move):
                winner = -to_play
                break
            board = play_move(board, to_play, move)
            moves.append(index_to_coord(move))
            to_play = -to_play
        else:
            winner = -to_play
        if winner == BLACK:
            black_wins += 1
        else:
            white_wins += 1
        print(f"game {game}: winner={color_name(winner)} plies={len(moves)}")
        print("moves:", " ".join(moves))
        for line in board_to_lines(board):
            print(line)
    print(f"summary: black={black_wins} white={white_wins}")
    return 0


def run_generate_data(args: argparse.Namespace) -> int:
    if args.workers > 1:
        return run_generate_data_parallel(args)

    net = load_network(
        args.model, args.backend, args.device, args.channels, args.blocks,
        args.inference_amp, args.inference_tf32,
    )
    progress = None if args.quiet else _make_progress_printer(args.log)
    samples, stats = generate_self_play_samples(
        network=net,
        games=args.games,
        simulations=args.simulations,
        time_limit=args.time_limit,
        cpuct=args.cpuct,
        max_plies=args.max_plies,
        seed=args.seed,
        temperature_moves=args.temperature_moves,
        opening_temperature=args.opening_temperature,
        opening_uniform_mix=args.opening_uniform_mix,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        clean_label_search=args.clean_label_search,
        label_simulations=args.label_simulations,
        label_time_limit=args.label_time_limit,
        fixed_simulations=args.fixed_simulations,
        label_fixed_simulations=args.label_fixed_simulations,
        require_terminal=args.require_terminal,
        root_tactical_weight=args.root_tactical_weight,
        inference_batch_size=args.inference_batch_size,
        endgame_moves=args.endgame_moves,
        endgame_nodes=args.endgame_nodes,
        endgame_time_fraction=args.endgame_time_fraction,
        progress=progress,
    )
    metadata = _self_play_metadata(args, net)
    metadata["opening_move_counts"] = stats.get("opening_move_counts", [])
    metadata["opening_unique"] = stats.get("opening_unique", [])
    count = write_samples(args.output, samples, metadata=metadata)
    summary = (
        f"wrote {count} samples to {args.output}; wins black={stats['black']} white={stats['white']} "
        f"entropy={float(stats.get('policy_entropy', 0.0)) / max(1, int(stats.get('plies', 0))):.3f} "
        f"label_entropy={float(stats.get('label_policy_entropy', 0.0)) / max(1, int(stats.get('plies', 0))):.3f} "
        f"clean_labels={bool(stats.get('clean_label_search', False))} "
        f"override={int(stats.get('tactical_overrides', 0))} "
        f"immediate_loss={_percent(float(stats.get('tactical_immediate_loss_rate', 0.0)))}"
    )
    print(summary)
    print(describe_samples(samples))
    opening_summary = _format_opening_coverage(stats.get("opening_move_counts"))
    if opening_summary:
        print(opening_summary)
    _append_log(args.log, summary)
    _append_log(args.log, describe_samples(samples))
    _append_log(args.log, opening_summary)
    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "output": str(args.output),
                    "samples": count,
                    "stats": stats,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


def run_generate_data_parallel(args: argparse.Namespace) -> int:
    worker_count = min(max(1, args.workers), max(1, args.games))
    shared_inference = bool(args.shared_inference)
    if shared_inference and args.backend != "torch":
        raise ValueError("--shared-inference requires --backend torch")
    if shared_inference and str(args.device).lower() == "cpu":
        raise ValueError("--shared-inference requires a CUDA device")
    output_path = Path(args.output)
    parts_dir = output_path.parent / f".{output_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    worker_log_dir = parts_dir / "logs"
    worker_log_dir.mkdir(parents=True, exist_ok=True)

    game_counts = _split_work(args.games, worker_count)
    initial_completed = _self_play_completed_games(parts_dir, game_counts)
    eta_tracker = _EtaTracker(max(20, 2 * worker_count))
    progress_state: dict[str, object] = {
        "completed": initial_completed,
        "seen": set(),
    }
    base_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 30)
    run_config = {
        "model": args.model,
        "model_sha256": _file_sha256(args.model),
        "backend": args.backend,
        "device": args.device,
        "channels": args.channels,
        "blocks": args.blocks,
        "games": args.games,
        "game_counts": game_counts,
        "max_plies": args.max_plies,
        "simulations": args.simulations,
        "time_limit": args.time_limit,
        "cpuct": args.cpuct,
        "root_tactical_weight": args.root_tactical_weight,
        "endgame_moves": args.endgame_moves,
        "endgame_nodes": args.endgame_nodes,
        "endgame_time_fraction": args.endgame_time_fraction,
        "inference_batch_size": args.inference_batch_size,
        "inference_amp": args.inference_amp,
        "inference_tf32": args.inference_tf32,
        "temperature_moves": args.temperature_moves,
        "opening_temperature": args.opening_temperature,
        "opening_uniform_mix": args.opening_uniform_mix,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_epsilon": args.dirichlet_epsilon,
        "shared_inference": shared_inference,
        "seed": base_seed,
    }
    # Keep the fingerprint compatible with pre-clean-label jobs when the new
    # feature is disabled, while making every clean-label setting resumable.
    if (
        args.clean_label_search
        or args.label_simulations is not None
        or args.label_time_limit is not None
    ):
        run_config.update(
            {
                "clean_label_search": args.clean_label_search,
                "label_simulations": (
                    args.label_simulations or args.simulations
                ) if args.clean_label_search else 0,
                "label_time_limit": (
                    args.label_time_limit or args.time_limit
                ) if args.clean_label_search else 0.0,
            }
        )
    for key in ("fixed_simulations", "label_fixed_simulations", "require_terminal"):
        if getattr(args, key):
            run_config[key] = True
    run_fingerprint = parameter_fingerprint(run_config)
    started_at = time.perf_counter()
    start_line = (
        "parallel self-play start "
        f"workers={worker_count} games={args.games} simulations={args.simulations} "
        f"time_limit={args.time_limit:.3f}s backend={args.backend} model={args.model or 'default'} "
        f"shared_inference={shared_inference} inference_batch={args.inference_batch_size} "
        f"amp={args.inference_amp} tf32={args.inference_tf32} "
        f"resumed={initial_completed}"
    )
    if not args.quiet:
        print(start_line, flush=True)
    _append_log(args.log, start_line)

    jobs = []
    for worker_id, games in enumerate(game_counts):
        if games <= 0:
            continue
        part_path = parts_dir / f"worker_{worker_id:03d}.jsonl"
        log_path = worker_log_dir / f"worker_{worker_id:03d}.log"
        jobs.append(
            {
                "worker_id": worker_id,
                "worker_total": worker_count,
                "games": games,
                "part_path": str(part_path),
                "log_path": str(log_path),
                "model": args.model,
                "backend": args.backend,
                "device": args.device,
                "channels": args.channels,
                "blocks": args.blocks,
                "simulations": args.simulations,
                "time_limit": args.time_limit,
                "cpuct": args.cpuct,
                "root_tactical_weight": args.root_tactical_weight,
                "endgame_moves": args.endgame_moves,
                "endgame_nodes": args.endgame_nodes,
                "endgame_time_fraction": args.endgame_time_fraction,
                "inference_amp": args.inference_amp,
                "inference_tf32": args.inference_tf32,
                "inference_batch_size": args.inference_batch_size,
                "max_plies": args.max_plies,
                "seed": base_seed + worker_id,
                "temperature_moves": args.temperature_moves,
                "opening_temperature": args.opening_temperature,
                "opening_uniform_mix": args.opening_uniform_mix,
                "dirichlet_alpha": args.dirichlet_alpha,
                "dirichlet_epsilon": args.dirichlet_epsilon,
                "clean_label_search": args.clean_label_search,
                "label_simulations": args.label_simulations,
                "label_time_limit": args.label_time_limit,
                "fixed_simulations": args.fixed_simulations,
                "label_fixed_simulations": args.label_fixed_simulations,
                "require_terminal": args.require_terminal,
                "shared_inference": shared_inference,
                "run_fingerprint": run_fingerprint,
            }
        )

    manager = mp.Manager()
    event_queue = manager.Queue()
    request_queue = None
    response_queues: list[object] = []
    status_queue = None
    inference_server = None
    if shared_inference:
        from .inference_service import run_inference_server

        request_queue = manager.Queue(maxsize=max(8, worker_count * 4))
        response_queues = [manager.Queue(maxsize=4) for _ in range(worker_count)]
        status_queue = manager.Queue(maxsize=4)
        for job in jobs:
            worker_id = int(job["worker_id"])
            job["inference_request_queue"] = request_queue
            job["inference_response_queue"] = response_queues[worker_id]
        inference_server = mp.get_context("spawn").Process(
            target=run_inference_server,
            args=(
                request_queue,
                response_queues,
                status_queue,
                args.model,
                args.device,
                args.channels,
                args.blocks,
                args.inference_amp,
                args.inference_tf32,
                args.inference_server_batch_size,
                args.inference_server_wait_ms,
            ),
            name="nogo-gpu-inference",
        )
        try:
            inference_server.start()
            try:
                status, detail = status_queue.get(timeout=180.0)
            except Empty as exc:
                raise RuntimeError("shared inference server did not become ready") from exc
            if status != "ready":
                raise RuntimeError(f"shared inference server startup failed: {detail}")
        except BaseException:
            if inference_server.is_alive():
                inference_server.terminate()
                inference_server.join(timeout=10.0)
            manager.shutdown()
            raise
        line = (
            "shared inference ready "
            f"server_batch={args.inference_server_batch_size} "
            f"wait_ms={args.inference_server_wait_ms:.3f} amp={args.inference_amp}"
        )
        if not args.quiet:
            print(line, flush=True)
        _append_log(args.log, line)
    results: list[dict[str, object]] = []
    shared_server_stats: dict[str, object] | None = None
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_worker = {executor.submit(_generate_data_worker, job, event_queue): int(job["worker_id"]) for job in jobs}
            pending = set(future_to_worker)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                _drain_progress_events(
                    event_queue,
                    args.log,
                    args.quiet,
                    eta_tracker=eta_tracker,
                    total_games=args.games,
                    progress_state=progress_state,
                )
                for future in done:
                    worker_id = future_to_worker[future]
                    result = future.result()
                    results.append(result)
                    line = (
                        f"worker {worker_id + 1}/{worker_count} done "
                        f"games={result['games']} samples={result['samples']} "
                        f"wins black={result['black']} white={result['white']} "
                        f"elapsed={_format_seconds(float(result['elapsed']))}"
                    )
                    if not args.quiet:
                        print(line, flush=True)
                    _append_log(args.log, line)
            _drain_progress_events(
                event_queue,
                args.log,
                args.quiet,
                eta_tracker=eta_tracker,
                total_games=args.games,
                progress_state=progress_state,
            )
    finally:
        active_exception = sys.exc_info()[0] is not None
        server_error: BaseException | None = None
        if inference_server is not None:
            try:
                request_queue.put(None)
                inference_server.join(timeout=180.0)
                if inference_server.is_alive():
                    inference_server.terminate()
                    inference_server.join(timeout=10.0)
                    raise RuntimeError("shared inference server did not stop cleanly")
                if inference_server.exitcode != 0:
                    raise RuntimeError(
                        f"shared inference server exited with code {inference_server.exitcode}"
                    )
                try:
                    status, detail = status_queue.get_nowait()
                except Empty:
                    status, detail = None, None
                if status == "stopped" and isinstance(detail, dict):
                    shared_server_stats = detail
                    line = (
                        "shared inference done "
                        f"calls={detail['calls']} positions={detail['positions']} "
                        f"avg_batch={float(detail['average_batch']):.2f} "
                        f"elapsed={_format_seconds(float(detail['inference_seconds']))}"
                    )
                    if not args.quiet:
                        print(line, flush=True)
                    _append_log(args.log, line)
            except BaseException as exc:
                server_error = exc
        manager.shutdown()
        if server_error is not None and not active_exception:
            raise server_error

    results.sort(key=lambda item: int(item["worker_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_paths = [Path(str(result["part_path"])) for result in results]
    opening_move_counts = _merge_opening_counts(
        [result.get("opening_move_counts", []) for result in results]
    )
    metadata = _self_play_metadata(args)
    if results:
        actual_network = results[0].get("network_metadata")
        if isinstance(actual_network, dict):
            for key, value in actual_network.items():
                if value is not None:
                    metadata[key] = value
    total_samples = write_sample_parts(
        output_path,
        part_paths,
        metadata={
            **metadata,
            "run_fingerprint": run_fingerprint,
            "base_seed": base_seed,
            "game_counts": game_counts,
            "opening_move_counts": opening_move_counts,
            "opening_unique": [len(counts) for counts in opening_move_counts],
        },
    )

    black_wins = sum(int(result["black"]) for result in results)
    white_wins = sum(int(result["white"]) for result in results)
    aggregate_stats = {
        "games": sum(int(result["games"]) for result in results),
        "black": black_wins,
        "white": white_wins,
        "plies": sum(int(result.get("plies", 0)) for result in results),
        "search_seconds": sum(float(result.get("search_seconds", 0.0)) for result in results),
        "completed_simulations": sum(int(result.get("completed_simulations", 0)) for result in results),
        "action_completed_simulations": sum(
            int(result.get("action_completed_simulations", 0)) for result in results
        ),
        "inference_calls": sum(int(result.get("inference_calls", 0)) for result in results),
        "inference_positions": sum(int(result.get("inference_positions", 0)) for result in results),
        "inference_seconds": sum(float(result.get("inference_seconds", 0.0)) for result in results),
        "policy_entropy": sum(float(result.get("policy_entropy", 0.0)) for result in results),
        "tactical_decisions": sum(int(result.get("tactical_decisions", 0)) for result in results),
        "tactical_immediate_losses": sum(int(result.get("tactical_immediate_losses", 0)) for result in results),
        "tactical_overrides": sum(int(result.get("tactical_overrides", 0)) for result in results),
        "label_policy_entropy": sum(float(result.get("label_policy_entropy", 0.0)) for result in results),
        "label_search_seconds": sum(float(result.get("label_search_seconds", 0.0)) for result in results),
        "label_completed_simulations": sum(int(result.get("label_completed_simulations", 0)) for result in results),
        "label_inference_calls": sum(int(result.get("label_inference_calls", 0)) for result in results),
        "label_inference_positions": sum(int(result.get("label_inference_positions", 0)) for result in results),
        "label_inference_seconds": sum(float(result.get("label_inference_seconds", 0.0)) for result in results),
    }
    aggregate_stats["opening_move_counts"] = opening_move_counts
    aggregate_stats["opening_unique"] = [len(counts) for counts in opening_move_counts]
    aggregate_stats["clean_label_search"] = bool(args.clean_label_search)
    aggregate_stats["label_simulations"] = (
        (args.label_simulations or args.simulations)
        if args.clean_label_search else 0
    )
    aggregate_stats["label_time_limit"] = (
        (args.label_time_limit or args.time_limit)
        if args.clean_label_search else 0.0
    )
    elapsed = time.perf_counter() - started_at
    aggregate_stats["elapsed"] = elapsed
    samples = read_samples(output_path)
    summary = (
        f"wrote {total_samples} samples to {args.output}; "
        f"wins black={black_wins} white={white_wins}; "
        f"entropy={aggregate_stats['policy_entropy'] / max(1, aggregate_stats['plies']):.3f}; "
        f"label_entropy={aggregate_stats['label_policy_entropy'] / max(1, aggregate_stats['plies']):.3f}; "
        f"clean_labels={args.clean_label_search}; "
        f"action_sim/move={aggregate_stats['action_completed_simulations'] / max(1, aggregate_stats['plies']):.1f}; "
        f"label_sim/move={aggregate_stats['label_completed_simulations'] / max(1, aggregate_stats['plies']):.1f}; "
        f"override={aggregate_stats['tactical_overrides']}; "
        f"immediate_loss={_percent(aggregate_stats['tactical_immediate_losses'] / max(1, aggregate_stats['tactical_decisions']))}; "
        f"workers={worker_count}; elapsed={_format_seconds(elapsed)}"
    )
    print(summary)
    print(describe_samples(samples))
    opening_summary = _format_opening_coverage(opening_move_counts)
    if opening_summary:
        print(opening_summary)
    _append_log(args.log, summary)
    _append_log(args.log, describe_samples(samples))
    _append_log(args.log, opening_summary)
    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "output": str(args.output),
                    "samples": total_samples,
                    "stats": aggregate_stats,
                    "workers": results,
                    "shared_inference": shared_server_stats,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


def _generate_data_worker(job: dict[str, object], event_queue: object) -> dict[str, object]:
    started_at = time.perf_counter()
    worker_id = int(job["worker_id"])
    worker_total = int(job["worker_total"])
    log_path = str(job["log_path"])
    part_path = Path(str(job["part_path"]))
    progress_path = part_path.with_suffix(part_path.suffix + ".progress.json")

    def emit_progress(event: dict[str, object]) -> None:
        event = dict(event)
        event["worker_id"] = worker_id
        event["worker_total"] = worker_total
        line = _format_progress_event(event)
        if line:
            _append_log(log_path, line)
        try:
            event_queue.put(event)
        except Exception:
            pass

    completed_games = 0
    sample_count = 0
    black_wins = 0
    white_wins = 0
    total_plies = 0
    total_search_seconds = 0.0
    total_completed_simulations = 0
    total_inference_calls = 0
    total_inference_positions = 0
    total_inference_seconds = 0.0
    total_policy_entropy = 0.0
    total_label_policy_entropy = 0.0
    total_label_search_seconds = 0.0
    total_label_completed_simulations = 0
    total_label_inference_calls = 0
    total_label_inference_positions = 0
    total_label_inference_seconds = 0.0
    total_tactical_decisions = 0
    total_tactical_immediate_losses = 0
    total_tactical_overrides = 0
    opening_move_counts = _empty_opening_counts()
    if part_path.exists() and progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        saved_fingerprint = state.get("run_fingerprint")
        current_fingerprint = job.get("run_fingerprint")
        if (
            saved_fingerprint is not None
            and current_fingerprint is not None
            and str(saved_fingerprint) != str(current_fingerprint)
        ):
            raise RuntimeError(
                f"self-play checkpoint configuration mismatch: {progress_path}; "
                "use a new output path or remove only this run's parts directory"
            )
        raw_completed = state.get("completed_games", 0)
        if isinstance(raw_completed, list):
            completed_games = len(raw_completed)
        else:
            completed_games = int(raw_completed)
        sample_count = int(state.get("samples", 0))
        black_wins = int(state.get("black", 0))
        white_wins = int(state.get("white", 0))
        total_plies = int(state.get("plies", 0))
        total_search_seconds = float(state.get("search_seconds", 0.0))
        total_completed_simulations = int(state.get("completed_simulations", 0))
        total_inference_calls = int(state.get("inference_calls", 0))
        total_inference_positions = int(state.get("inference_positions", 0))
        total_inference_seconds = float(state.get("inference_seconds", 0.0))
        total_policy_entropy = float(state.get("policy_entropy", 0.0))
        total_label_policy_entropy = float(state.get("label_policy_entropy", 0.0))
        total_label_search_seconds = float(state.get("label_search_seconds", 0.0))
        total_label_completed_simulations = int(state.get("label_completed_simulations", 0))
        total_label_inference_calls = int(state.get("label_inference_calls", 0))
        total_label_inference_positions = int(state.get("label_inference_positions", 0))
        total_label_inference_seconds = float(state.get("label_inference_seconds", 0.0))
        total_tactical_decisions = int(state.get("tactical_decisions", 0))
        total_tactical_immediate_losses = int(state.get("tactical_immediate_losses", 0))
        total_tactical_overrides = int(state.get("tactical_overrides", 0))
        opening_move_counts = _coerce_opening_counts(state.get("opening_move_counts"))
        lines = part_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < sample_count:
            raise RuntimeError(f"self-play checkpoint is incomplete: {part_path}")
        if len(lines) > sample_count:
            committed = "\n".join(lines[:sample_count])
            part_path.write_text(committed + ("\n" if committed else ""), encoding="utf-8")
    elif part_path.exists() and part_path.stat().st_size > 0:
        # A part file without a progress record may contain a complete game
        # written just before a process was killed, but its aggregate counters
        # cannot be reconstructed safely.  Preserve it and fail loudly rather
        # than silently deleting valid training data.
        raise RuntimeError(
            f"orphaned self-play part without progress checkpoint: {part_path}; "
            "preserve it and resume with the matching progress file or a new output path"
        )
    else:
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_text("", encoding="utf-8")
        progress_path.unlink(missing_ok=True)

    if bool(job.get("shared_inference", False)):
        from .inference_service import RemotePolicyValueNet

        net = RemotePolicyValueNet(
            worker_id,
            job["inference_request_queue"],
            job["inference_response_queue"],
        )
    else:
        net = load_network(
            str(job["model"]) if job["model"] is not None else None,
            str(job["backend"]),
            str(job["device"]),
            int(job["channels"]),
            int(job["blocks"]),
            bool(job.get("inference_amp", False)),
            bool(job.get("inference_tf32", False)),
        )

    network_metadata = {
        "channels": getattr(net, "channels", None),
        "blocks": getattr(net, "blocks", None),
        "input_planes": getattr(net, "input_planes", None),
        "policy_channels": getattr(net, "policy_channels", None),
        "value_channels": getattr(net, "value_channels", None),
        "value_hidden": getattr(net, "value_hidden", None),
        "norm_type": getattr(net, "norm_type", None),
        "policy_head": getattr(net, "policy_head", None),
    }

    total_games = int(job["games"])
    for local_game in range(completed_games + 1, total_games + 1):
        game_event: dict[str, object] = {}

        def capture_game_event(event: dict[str, object]) -> None:
            if event.get("event") == "self_play_game":
                game_event.update(event)

        samples, stats = generate_self_play_samples(
            network=net,
            games=1,
            simulations=int(job["simulations"]),
            time_limit=float(job["time_limit"]),
            cpuct=float(job["cpuct"]),
            max_plies=int(job["max_plies"]),
            seed=int(job["seed"]) + local_game * 100003,
            temperature_moves=int(job["temperature_moves"]),
            opening_temperature=float(job.get("opening_temperature", 1.0)),
            opening_uniform_mix=float(job.get("opening_uniform_mix", 0.0)),
            dirichlet_alpha=float(job["dirichlet_alpha"]),
            dirichlet_epsilon=float(job["dirichlet_epsilon"]),
            root_tactical_weight=float(job["root_tactical_weight"]),
            inference_batch_size=int(job.get("inference_batch_size", 32)),
            endgame_moves=int(job.get("endgame_moves", 16)),
            endgame_nodes=int(job.get("endgame_nodes", 200000)),
            endgame_time_fraction=float(job.get("endgame_time_fraction", 0.35)),
            clean_label_search=bool(job.get("clean_label_search", False)),
            label_simulations=(
                int(job["label_simulations"])
                if job.get("label_simulations") is not None else None
            ),
            label_time_limit=(
                float(job["label_time_limit"])
                if job.get("label_time_limit") is not None else None
            ),
            progress=capture_game_event,
            fixed_simulations=bool(job.get("fixed_simulations", False)),
            label_fixed_simulations=bool(job.get("label_fixed_simulations", False)),
            require_terminal=bool(job.get("require_terminal", False)),
        )
        with part_path.open("a", encoding="utf-8") as part:
            for sample in samples:
                part.write(json.dumps(sample.to_json(), separators=(",", ":")) + "\n")
            part.flush()
        sample_count += len(samples)
        black_wins += int(stats["black"])
        white_wins += int(stats["white"])
        total_plies += int(stats.get("plies", 0))
        total_search_seconds += float(stats.get("search_seconds", 0.0))
        total_completed_simulations += int(stats.get("completed_simulations", 0))
        total_inference_calls += int(stats.get("inference_calls", 0))
        total_inference_positions += int(stats.get("inference_positions", 0))
        total_inference_seconds += float(stats.get("inference_seconds", 0.0))
        total_policy_entropy += float(stats.get("policy_entropy", 0.0))
        total_label_policy_entropy += float(stats.get("label_policy_entropy", 0.0))
        total_label_search_seconds += float(stats.get("label_search_seconds", 0.0))
        total_label_completed_simulations += int(stats.get("label_completed_simulations", 0))
        total_label_inference_calls += int(stats.get("label_inference_calls", 0))
        total_label_inference_positions += int(stats.get("label_inference_positions", 0))
        total_label_inference_seconds += float(stats.get("label_inference_seconds", 0.0))
        total_tactical_decisions += int(stats.get("tactical_decisions", 0))
        total_tactical_immediate_losses += int(stats.get("tactical_immediate_losses", 0))
        total_tactical_overrides += int(stats.get("tactical_overrides", 0))
        _accumulate_opening_counts(opening_move_counts, stats.get("opening_move_counts"))
        completed_games = local_game
        _write_self_play_checkpoint(
            progress_path,
            completed_games,
            sample_count,
            black_wins,
            white_wins,
            total_plies,
            total_search_seconds,
            total_completed_simulations,
            total_inference_calls,
            total_inference_positions,
            total_inference_seconds,
            total_policy_entropy,
            total_tactical_decisions,
            total_tactical_immediate_losses,
            total_tactical_overrides,
            run_fingerprint=str(job.get("run_fingerprint", "")),
            label_policy_entropy=total_label_policy_entropy,
            label_search_seconds=total_label_search_seconds,
            label_completed_simulations=total_label_completed_simulations,
            label_inference_calls=total_label_inference_calls,
            label_inference_positions=total_label_inference_positions,
            label_inference_seconds=total_label_inference_seconds,
            opening_move_counts=opening_move_counts,
        )
        event = dict(game_event)
        event.update(
            event="self_play_game",
            game=local_game,
            games=total_games,
            samples=sample_count,
            black_wins=black_wins,
            white_wins=white_wins,
        )
        emit_progress(event)

    return {
        "worker_id": worker_id,
        "games": total_games,
        "samples": sample_count,
        "black": black_wins,
        "white": white_wins,
        "plies": total_plies,
        "search_seconds": total_search_seconds,
        "completed_simulations": total_completed_simulations,
        "action_completed_simulations": (
            total_completed_simulations - total_label_completed_simulations
        ),
        "inference_calls": total_inference_calls,
        "inference_positions": total_inference_positions,
        "inference_seconds": total_inference_seconds,
        "policy_entropy": total_policy_entropy,
        "label_policy_entropy": total_label_policy_entropy,
        "label_search_seconds": total_label_search_seconds,
        "label_completed_simulations": total_label_completed_simulations,
        "label_inference_calls": total_label_inference_calls,
        "label_inference_positions": total_label_inference_positions,
        "label_inference_seconds": total_label_inference_seconds,
        "clean_label_search": bool(job.get("clean_label_search", False)),
        "opening_move_counts": opening_move_counts,
        "tactical_decisions": total_tactical_decisions,
        "tactical_immediate_losses": total_tactical_immediate_losses,
        "tactical_overrides": total_tactical_overrides,
        "network_metadata": network_metadata,
        "part_path": str(part_path),
        "elapsed": time.perf_counter() - started_at,
    }


def _write_self_play_checkpoint(
    path: Path,
    completed_games: int,
    samples: int,
    black: int,
    white: int,
    plies: int = 0,
    search_seconds: float = 0.0,
    completed_simulations: int = 0,
    inference_calls: int = 0,
    inference_positions: int = 0,
    inference_seconds: float = 0.0,
    policy_entropy: float = 0.0,
    tactical_decisions: int = 0,
    tactical_immediate_losses: int = 0,
    tactical_overrides: int = 0,
    run_fingerprint: str = "",
    label_policy_entropy: float = 0.0,
    label_search_seconds: float = 0.0,
    label_completed_simulations: int = 0,
    label_inference_calls: int = 0,
    label_inference_positions: int = 0,
    label_inference_seconds: float = 0.0,
    opening_move_counts: list[dict[str, int]] | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "completed_games": completed_games,
                "samples": samples,
                "black": black,
                "white": white,
                "plies": plies,
                "search_seconds": search_seconds,
                "completed_simulations": completed_simulations,
                "inference_calls": inference_calls,
                "inference_positions": inference_positions,
                "inference_seconds": inference_seconds,
                "policy_entropy": policy_entropy,
                "tactical_decisions": tactical_decisions,
                "tactical_immediate_losses": tactical_immediate_losses,
                "tactical_overrides": tactical_overrides,
                "label_policy_entropy": label_policy_entropy,
                "label_search_seconds": label_search_seconds,
                "label_completed_simulations": label_completed_simulations,
                "label_inference_calls": label_inference_calls,
                "label_inference_positions": label_inference_positions,
                "label_inference_seconds": label_inference_seconds,
                "opening_move_counts": opening_move_counts or _empty_opening_counts(),
                "run_fingerprint": run_fingerprint,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _self_play_completed_games(parts_dir: Path, game_counts: list[int]) -> int:
    """Read committed worker progress without counting an unfinished game."""
    completed = 0
    for worker_id, total_games in enumerate(game_counts):
        progress_path = (
            parts_dir / f"worker_{worker_id:03d}.jsonl"
        ).with_suffix(".jsonl.progress.json")
        if not progress_path.exists():
            continue
        try:
            state = json.loads(progress_path.read_text(encoding="utf-8"))
            raw_completed = state.get("completed_games", 0)
            worker_completed = len(raw_completed) if isinstance(raw_completed, list) else int(raw_completed)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid self-play checkpoint: {progress_path}") from exc
        if not 0 <= worker_completed <= total_games:
            raise RuntimeError(
                f"invalid completed_games={worker_completed} in {progress_path}; "
                f"expected 0..{total_games}"
            )
        completed += worker_completed
    return completed


def _split_work(total: int, workers: int) -> list[int]:
    base = total // workers
    extra = total % workers
    return [base + (1 if idx < extra else 0) for idx in range(workers)]


def _empty_opening_counts() -> list[dict[str, int]]:
    return [{} for _ in range(4)]


def _coerce_opening_counts(value: object) -> list[dict[str, int]]:
    """Load bounded, JSON-safe opening counters from a worker checkpoint."""
    result = _empty_opening_counts()
    if not isinstance(value, list):
        return result
    for ply, raw_counts in enumerate(value[:4]):
        if not isinstance(raw_counts, dict):
            continue
        for move, raw_count in raw_counts.items():
            if not isinstance(move, str):
                continue
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count > 0:
                result[ply][move] = count
    return result


def _accumulate_opening_counts(target: list[dict[str, int]], value: object) -> None:
    for ply, counts in enumerate(_coerce_opening_counts(value)):
        if ply >= len(target):
            target.append({})
        for move, count in counts.items():
            target[ply][move] = target[ply].get(move, 0) + count


def _merge_opening_counts(values: Iterable[object]) -> list[dict[str, int]]:
    result = _empty_opening_counts()
    for value in values:
        _accumulate_opening_counts(result, value)
    return result


def _format_opening_coverage(value: object) -> str:
    counts = _coerce_opening_counts(value)
    if not any(counts):
        return ""
    parts: list[str] = []
    for ply, moves in enumerate(counts, start=1):
        top = sorted(moves.items(), key=lambda item: (-item[1], item[0]))[:5]
        top_text = ",".join(f"{move}:{count}" for move, count in top) or "none"
        parts.append(f"ply{ply} unique={len(moves)} top={top_text}")
    return "opening coverage " + " | ".join(parts)


def run_train(args: argparse.Namespace) -> int:
    progress = None if args.quiet else _make_progress_printer(args.log)
    samples = read_samples(args.data)
    if args.data_weight <= 0.0 or any(weight <= 0.0 for weight in args.replay_weight):
        raise ValueError("data source weights must be positive")
    if args.replay_weight and len(args.replay_weight) != len(args.replay_data):
        raise ValueError("provide exactly one --replay-weight for each --replay-data")
    if args.replay_max_samples and len(args.replay_max_samples) != len(args.replay_data):
        raise ValueError("provide exactly one --replay-max-samples for each --replay-data")
    if any(limit <= 0 for limit in args.replay_max_samples):
        raise ValueError("replay sample limits must be positive")
    _scale_sample_weights(samples, args.data_weight)
    source_summary = [(args.data, len(samples), args.data_weight)]
    replay_weights = args.replay_weight or [1.0] * len(args.replay_data)
    replay_limits = args.replay_max_samples or [0] * len(args.replay_data)
    for replay_path, replay_weight, replay_limit in zip(
        args.replay_data, replay_weights, replay_limits
    ):
        replay_samples = read_samples(replay_path)
        if replay_limit and len(replay_samples) > replay_limit:
            replay_samples = _limit_replay_samples(replay_samples, replay_limit)
        _scale_sample_weights(replay_samples, replay_weight)
        source_summary.append((replay_path, len(replay_samples), replay_weight))
        samples.extend(replay_samples)
    print(
        "training sources "
        + "; ".join(
            f"{path}:samples={count}:weight={weight:g}"
            for path, count, weight in source_summary
        )
    )
    if args.skip_policy_sanitize:
        print("skipped policy target sanitization for preprocessed input")
        _append_log(args.log, "skipped policy target sanitization for preprocessed input")
    else:
        policy_stats = sanitize_policy_targets(samples)
        print(
            "sanitized policy targets "
            + " ".join(f"{key}={value}" for key, value in policy_stats.items())
        )
        _append_log(
            args.log,
            "sanitized policy targets "
            + " ".join(f"{key}={value}" for key, value in policy_stats.items()),
        )
    if args.sanitize_tactical_policy:
        sanitize_started_at = time.perf_counter()
        changed = sanitize_tactical_policy_targets(samples, progress=progress)
        print(
            f"sanitized tactical policy targets changed={changed}/{len(samples)} "
            f"elapsed={_format_seconds(time.perf_counter() - sanitize_started_at)}"
        )
    if args.relabel_endgames:
        from .alphazero import relabel_exact_endgames_parallel

        relabel_started_at = time.perf_counter()
        relabel_stats = relabel_exact_endgames_parallel(
            samples,
            max_legal_moves=args.relabel_endgame_moves,
            max_nodes=args.relabel_endgame_nodes,
            exact_weight=args.relabel_endgame_weight,
            workers=args.relabel_endgame_workers,
            progress=progress,
        )
        print(
            "relabelled exact endgames "
            + " ".join(f"{key}={value}" for key, value in relabel_stats.items())
            + f" elapsed={_format_seconds(time.perf_counter() - relabel_started_at)}"
        )
    if args.preprocessed_output:
        preprocessed_metadata = {
            "kind": "preprocessed-training-data",
            "source": args.data,
            "replay_data": args.replay_data,
            "data_weight": args.data_weight,
            "replay_weight": args.replay_weight,
            "replay_max_samples": args.replay_max_samples,
            "sanitize_tactical_policy": args.sanitize_tactical_policy,
            "relabel_endgames": args.relabel_endgames,
            "relabel_endgame_moves": args.relabel_endgame_moves,
            "relabel_endgame_nodes": args.relabel_endgame_nodes,
            "relabel_endgame_weight": args.relabel_endgame_weight,
            "sample_count": len(samples),
        }
        preprocessed_count = write_samples(
            args.preprocessed_output,
            samples,
            metadata=preprocessed_metadata,
        )
        print(f"saved preprocessed samples={preprocessed_count} -> {args.preprocessed_output}")
    validation_samples = []
    if args.backend == "torch" and args.validation_fraction:
        from .alphazero import split_training_samples

        samples, validation_samples = split_training_samples(
            samples,
            args.validation_fraction,
            args.seed,
            disjoint_positions=args.validation_disjoint_positions,
        )
        print(
            f"split samples before augmentation -> train={len(samples)} "
            f"validation={len(validation_samples)}"
        )
    if args.augment_symmetries:
        samples = augment_samples_with_symmetries(samples)
        validation_samples = augment_samples_with_symmetries(validation_samples)
        print(
            f"augmented samples with 8 symmetries -> train={len(samples)} "
            f"validation={len(validation_samples)}"
        )
    data_fingerprint = samples_fingerprint(samples)
    validation_fingerprint = samples_fingerprint(validation_samples)
    if args.backend == "torch":
        metrics = train_torch(
            samples=samples,
            output=args.output,
            validation_samples=validation_samples,
            checkpoint=args.checkpoint,
            resume_training=args.resume_training,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            channels=args.channels,
            blocks=args.blocks,
            input_planes=args.input_planes,
            policy_channels=args.policy_channels,
            value_channels=args.value_channels,
            value_hidden=args.value_hidden,
            norm_type=args.norm_type,
            policy_head=args.policy_head,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
            winner_policy_weight=args.winner_policy_weight,
            loser_policy_weight=args.loser_policy_weight,
            balance_policy_colors=args.balance_policy_colors,
            random_symmetries=args.random_symmetries,
            minimum_learning_rate=args.minimum_learning_rate,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
            amp=args.amp,
            data_transfer=args.data_transfer,
            distill_checkpoint=args.distill_checkpoint,
            distill_policy_weight=args.distill_policy_weight,
            distill_value_weight=args.distill_value_weight,
            freeze_batchnorm_stats=args.freeze_batchnorm_stats,
            data_fingerprint=data_fingerprint,
            validation_fingerprint=validation_fingerprint,
            progress=progress,
        )
    else:
        metrics = train_tiny_value_head(
            samples=samples,
            output=args.output,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            progress=progress,
        )
    summary = f"trained {args.backend} model -> {args.output}"
    metric_text = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
    print(summary)
    print(metric_text)
    _append_log(args.log, summary)
    _append_log(args.log, metric_text)
    return 0


def _scale_sample_weights(samples: list[TrainingSample], factor: float) -> None:
    if factor == 1.0:
        return
    for sample in samples:
        sample.sample_weight *= factor


def _limit_replay_samples(
    samples: list[TrainingSample],
    maximum: int,
) -> list[TrainingSample]:
    """Keep an ordered, evenly spaced subset of an oversized replay file.

    Replay files are trajectories concatenated in game order. Sampling by
    deterministic positions keeps the source reproducible and preserves its
    broad opening-to-endgame coverage without letting old files dominate the
    high-search data by raw sample count.
    """
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if len(samples) <= maximum:
        return samples
    if maximum == 1:
        return [samples[len(samples) // 2]]
    last = len(samples) - 1
    return [samples[(index * last) // (maximum - 1)] for index in range(maximum)]


def _self_play_metadata(args: argparse.Namespace, network: object | None = None) -> dict[str, object]:
    """Record the generation recipe beside JSONL so stale data is detectable."""
    actual_channels = getattr(network, "channels", args.channels)
    actual_blocks = getattr(network, "blocks", args.blocks)
    actual_policy_head = getattr(network, "policy_head", "unknown")
    return {
        "kind": "self-play",
        "backend": args.backend,
        "model": args.model,
        "device": args.device,
        "channels": actual_channels,
        "blocks": actual_blocks,
        "policy_head": actual_policy_head,
        "games": args.games,
        "max_plies": args.max_plies,
        "simulations": args.simulations,
        "time_limit": args.time_limit,
        "cpuct": args.cpuct,
        "temperature_moves": args.temperature_moves,
        "opening_temperature": args.opening_temperature,
        "opening_uniform_mix": args.opening_uniform_mix,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_epsilon": args.dirichlet_epsilon,
        "clean_label_search": args.clean_label_search,
        "policy_label_source": (
            "clean_noise_free_mcts" if args.clean_label_search else "action_mcts"
        ),
        "model_sha256": _file_sha256(args.model),
        "fixed_simulations": args.fixed_simulations,
        "label_fixed_simulations": args.label_fixed_simulations,
        "require_terminal": args.require_terminal,
        "label_simulations": (
            (args.label_simulations or args.simulations)
            if args.clean_label_search else 0
        ),
        "label_time_limit": (
            (args.label_time_limit or args.time_limit)
            if args.clean_label_search else 0.0
        ),
        "root_tactical_weight": args.root_tactical_weight,
        "inference_batch_size": args.inference_batch_size,
        "inference_amp": args.inference_amp,
        "inference_tf32": args.inference_tf32,
        "endgame_moves": args.endgame_moves,
        "endgame_nodes": args.endgame_nodes,
        "endgame_time_fraction": args.endgame_time_fraction,
        "shared_inference": args.shared_inference,
        "seed": args.seed,
    }


def run_upgrade_model(args: argparse.Namespace) -> int:
    from .torch_model import upgrade_checkpoint

    metrics = upgrade_checkpoint(
        checkpoint=args.checkpoint,
        output=args.output,
        input_planes=args.input_planes,
        policy_channels=args.policy_channels,
        value_channels=args.value_channels,
        value_hidden=args.value_hidden,
        norm_type=args.norm_type,
        policy_head=args.policy_head,
    )
    detail = " ".join(f"{key}={value}" for key, value in metrics.items())
    print(f"upgraded model {args.checkpoint} -> {args.output}")
    print(detail)
    return 0


class _EtaTracker:
    def __init__(self, window: int, now: float | None = None) -> None:
        self._points: deque[tuple[int, float]] = deque(maxlen=max(2, window + 1))
        self._completed = 0
        self._points.append((0, time.perf_counter() if now is None else now))

    def add(self, now: float | None = None) -> None:
        self._completed += 1
        self._points.append((self._completed, time.perf_counter() if now is None else now))

    @property
    def rate(self) -> float | None:
        if len(self._points) < 2:
            return None
        first_count, first_time = self._points[0]
        last_count, last_time = self._points[-1]
        elapsed = last_time - first_time
        completed = last_count - first_count
        if completed <= 0 or elapsed <= 0.0:
            return None
        return completed / elapsed

    def estimate(self, remaining: int) -> float | None:
        if remaining <= 0:
            return 0.0
        rate = self.rate
        return remaining / rate if rate else None


def _arena_checkpoint_path(args: argparse.Namespace) -> Path | None:
    explicit = getattr(args, "progress_json", None)
    if explicit:
        return Path(explicit)
    if args.result_json:
        return Path(str(args.result_json) + ".progress.json")
    if args.log:
        return Path(str(args.log) + ".progress.json")
    return None


def _arena_config(args: argparse.Namespace) -> dict[str, object]:
    keys = (
        "games", "max_plies", "a_backend", "a_model", "a_name", "a_device",
        "a_channels", "a_blocks", "a_inference_amp", "a_inference_tf32",
        "b_backend", "b_model", "b_name", "b_device", "b_channels", "b_blocks",
        "b_inference_amp", "b_inference_tf32", "simulations", "time_limit",
        "game_time_limit", "time_reserve", "root_tactical_weight", "cpuct",
        "endgame_moves", "endgame_nodes", "endgame_time_fraction",
        "inference_batch_size", "a_inference_batch_size", "b_inference_batch_size",
        "seed", "paired_seeds",
    )
    config = {key: getattr(args, key) for key in keys}
    # Preserve fingerprints of existing time-limited, empty-board evaluations.
    if getattr(args, "fixed_simulations", False):
        config["fixed_simulations"] = True
    if getattr(args, "opening_plies", 0):
        config["opening_plies"] = args.opening_plies
    return config


def _arena_config_fingerprint(args: argparse.Namespace) -> str:
    encoded = json.dumps(_arena_config(args), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _arena_game_seed(base_seed: int, game: int, paired: bool = True) -> int:
    """Return a common-random-number seed for a color-swapped game pair."""
    if game <= 0:
        raise ValueError("Arena game numbers are one-based")
    sequence = (game + 1) // 2 if paired else game
    return base_seed + sequence * 1000003


def _load_arena_checkpoint(
    args: argparse.Namespace,
) -> tuple[Path | None, int, dict[int, dict[str, object]]]:
    path = _arena_checkpoint_path(args)
    default_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1 << 30)
    if path is None:
        return path, int(default_seed), {}
    if not path.exists() or not getattr(args, "resume", True):
        _write_arena_checkpoint(path, args, int(default_seed), {})
        return path, int(default_seed), {}
    state = json.loads(path.read_text(encoding="utf-8"))
    if int(state.get("version", 0)) != 1:
        raise ValueError(f"unsupported Arena checkpoint version: {path}")
    expected = _arena_config_fingerprint(args)
    if state.get("config_fingerprint") != expected:
        raise ValueError(
            f"Arena checkpoint configuration mismatch: {path}; "
            "use --no-resume or a different --progress-json"
        )
    saved_models = state.get("model_sha256")
    if saved_models is not None and saved_models != getattr(args, "_arena_model_sha256", None):
        raise ValueError(f"Arena checkpoint model hashes mismatch: {path}")
    raw_results = state.get("results", {})
    if not isinstance(raw_results, dict):
        raise ValueError(f"Arena checkpoint has invalid results: {path}")
    results: dict[int, dict[str, object]] = {}
    for raw_game, raw_result in raw_results.items():
        game = int(raw_game)
        if not 1 <= game <= args.games or not isinstance(raw_result, dict):
            raise ValueError(f"Arena checkpoint has invalid game {raw_game}: {path}")
        results[game] = dict(raw_result)
    return path, int(state["base_seed"]), results


def _write_arena_checkpoint(
    path: Path | None,
    args: argparse.Namespace,
    base_seed: int,
    results: dict[int, dict[str, object]],
) -> None:
    if path is None:
        return
    payload = {
        "version": 1,
        "config_fingerprint": _arena_config_fingerprint(args),
        "config": _arena_config(args),
        "base_seed": base_seed,
        "completed_games": len(results),
        "model_sha256": getattr(args, "_arena_model_sha256", None),
        "results": {str(game): results[game] for game in sorted(results)},
    }
    _write_json_atomic(path, payload)


def _arena_totals(
    args: argparse.Namespace,
    results: dict[int, dict[str, object]],
) -> tuple[
    dict[str, int], dict[str, int], dict[str, int], dict[str, float],
    dict[str, int], dict[str, int], dict[str, int], dict[str, int],
]:
    wins = {args.a_name: 0, args.b_name: 0}
    illegal = {args.a_name: 0, args.b_name: 0}
    timeouts = {args.a_name: 0, args.b_name: 0}
    elapsed_total = {args.a_name: 0.0, args.b_name: 0.0}
    move_count = {args.a_name: 0, args.b_name: 0}
    simulations_total = {args.a_name: 0, args.b_name: 0}
    color_games = {"black": 0, "white": 0}
    color_wins = {"black": 0, "white": 0}
    for result in results.values():
        winner = str(result["winner"])
        wins[winner] += 1
        color = "black" if bool(result["a_is_black"]) else "white"
        color_games[color] += 1
        if winner == args.a_name:
            color_wins[color] += 1
        for name in (args.a_name, args.b_name):
            elapsed_total[name] += float(result["elapsed_total"][name])  # type: ignore[index]
            move_count[name] += int(result["move_count"][name])  # type: ignore[index]
            simulations_total[name] += int(result["simulations_total"][name])  # type: ignore[index]
        if result.get("illegal") is not None:
            illegal[str(result["illegal"])] += 1
        if result.get("timeout") is not None:
            timeouts[str(result["timeout"])] += 1
    return (
        wins, illegal, timeouts, elapsed_total, move_count, simulations_total,
        color_games, color_wins,
    )


def _record_arena_result(
    args: argparse.Namespace,
    progress_path: Path | None,
    base_seed: int,
    completed_results: dict[int, dict[str, object]],
    game: int,
    result: dict[str, object],
    eta_tracker: _EtaTracker,
    worker_id: int | None = None,
    worker_total: int | None = None,
) -> bool:
    if game in completed_results:
        return False
    completed_results[game] = dict(result)
    _write_arena_checkpoint(progress_path, args, base_seed, completed_results)
    eta_tracker.add()
    wins, *_ = _arena_totals(args, completed_results)
    progress_event: dict[str, object] = {
        "event": "arena_game",
        "game": len(completed_results),
        "game_id": game,
        "games": args.games,
        "winner": result["winner"],
        "plies": result["plies"],
        "score_a": wins[args.a_name],
        "score_b": wins[args.b_name],
        "a_name": args.a_name,
        "b_name": args.b_name,
        "rate": eta_tracker.rate,
        "eta": eta_tracker.estimate(args.games - len(completed_results)),
    }
    if worker_id is not None and worker_total is not None:
        progress_event["worker_id"] = worker_id
        progress_event["worker_total"] = worker_total
    line = _format_progress_event(progress_event)
    if not args.quiet:
        print(line, flush=True)
    _append_log(args.log, line)
    return True


def _finish_arena(
    args: argparse.Namespace,
    results: dict[int, dict[str, object]],
    workers: int,
) -> int:
    _verify_arena_models(args)
    (
        wins, illegal, timeouts, elapsed_total, move_count, simulations_total,
        color_games, color_wins,
    ) = _arena_totals(args, results)
    completed_games = len(results)
    a_rate = wins[args.a_name] / completed_games if completed_games else 0.0
    b_rate = wins[args.b_name] / completed_games if completed_games else 0.0
    summary = (
        f"arena done {args.a_name}={wins[args.a_name]} ({_percent(a_rate)}) "
        f"{args.b_name}={wins[args.b_name]} ({_percent(b_rate)}) "
        f"illegal {args.a_name}={illegal[args.a_name]} {args.b_name}={illegal[args.b_name]} "
        f"timeout {args.a_name}={timeouts[args.a_name]} {args.b_name}={timeouts[args.b_name]} "
        f"workers={workers}"
    )
    timing = (
        "avg_move_time "
        f"{args.a_name}={_format_seconds(elapsed_total[args.a_name] / max(1, move_count[args.a_name]))} "
        f"{args.b_name}={_format_seconds(elapsed_total[args.b_name] / max(1, move_count[args.b_name]))}"
    )
    a_model_sha256 = _file_sha256(args.a_model)
    b_model_sha256 = _file_sha256(args.b_model)
    identity = (
        f"model_fingerprint same_model={a_model_sha256 is not None and a_model_sha256 == b_model_sha256} "
        f"a={a_model_sha256 or 'missing'} b={b_model_sha256 or 'missing'}"
    )
    print(summary)
    print(timing)
    print(identity)
    _append_log(args.log, summary)
    _append_log(args.log, timing)
    _append_log(args.log, identity)
    if args.result_json:
        _write_arena_result(
            args, completed_games, wins, illegal, timeouts, elapsed_total, move_count,
            simulations_total, workers=workers, color_games=color_games, color_wins=color_wins,
        )
    return 0


def run_arena(args: argparse.Namespace) -> int:
    if getattr(args, "fixed_simulations", False) and args.game_time_limit > 0:
        raise ValueError("fixed simulations requires --game-time-limit 0")
    if not 0 <= getattr(args, "opening_plies", 0) <= 16:
        raise ValueError("opening plies must be between 0 and 16")
    args._arena_model_sha256 = {
        "a": _file_sha256(args.a_model), "b": _file_sha256(args.b_model),
    }
    if args.workers > 1:
        return run_arena_parallel(args)
    progress_path, base_seed, completed_results = _load_arena_checkpoint(args)
    missing_games = [game for game in range(1, args.games + 1) if game not in completed_results]
    eta_tracker = _EtaTracker(max(8, 2 * max(1, args.workers)))
    if missing_games:
        a_net = load_network(
            args.a_model, args.a_backend, args.a_device, args.a_channels, args.a_blocks,
            args.a_inference_amp, args.a_inference_tf32,
        )
        b_net = load_network(
            args.b_model, args.b_backend, args.b_device, args.b_channels, args.b_blocks,
            args.b_inference_amp, args.b_inference_tf32,
        )
        _warm_network(a_net)
        _warm_network(b_net)
        _verify_arena_models(args)
        for game in missing_games:
            result = _play_parallel_arena_game(
                args, a_net, b_net, game,
                _arena_game_seed(base_seed, game, getattr(args, "paired_seeds", True)),
            )
            _record_arena_result(
                args, progress_path, base_seed, completed_results, game, result, eta_tracker
            )
    else:
        line = f"arena resume complete games={len(completed_results)}/{args.games}"
        if not args.quiet:
            print(line, flush=True)
        _append_log(args.log, line)
    return _finish_arena(args, completed_results, workers=1)


def run_arena_parallel(args: argparse.Namespace) -> int:
    progress_path, base_seed, completed_results = _load_arena_checkpoint(args)
    missing_games = [game for game in range(1, args.games + 1) if game not in completed_results]
    configured_worker_count = min(max(1, args.workers), max(1, args.games))
    worker_count = min(configured_worker_count, max(1, len(missing_games)))
    if not missing_games:
        line = f"arena resume complete games={len(completed_results)}/{args.games}"
        if not args.quiet:
            print(line, flush=True)
        _append_log(args.log, line)
        return _finish_arena(args, completed_results, workers=configured_worker_count)
    jobs = []
    for worker_id in range(worker_count):
        game_ids = missing_games[worker_id::worker_count]
        if game_ids:
            jobs.append(
                {
                    "worker_id": worker_id,
                    "worker_total": worker_count,
                    "game_ids": game_ids,
                    "base_seed": base_seed,
                    "args": vars(args),
                }
            )

    start_line = (
        f"parallel arena start workers={worker_count} games={args.games} "
        f"resumed={len(completed_results)} remaining={len(missing_games)} "
        f"simulations={args.simulations} time_limit={args.time_limit:.3f}s"
    )
    if not args.quiet:
        print(start_line, flush=True)
    _append_log(args.log, start_line)

    manager = mp.Manager()
    event_queue = manager.Queue()
    eta_tracker = _EtaTracker(max(8, 2 * worker_count))
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_worker = {
                executor.submit(_arena_worker, job, event_queue): int(job["worker_id"])
                for job in jobs
            }
            pending = set(future_to_worker)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                _drain_parallel_arena_events(
                    event_queue, args, progress_path, base_seed, completed_results, eta_tracker
                )
                for future in done:
                    worker_result = future.result()
                    for item in worker_result["results"]:  # type: ignore[union-attr]
                        game = int(item["game"])
                        _record_arena_result(
                            args, progress_path, base_seed, completed_results, game,
                            dict(item["result"]), eta_tracker,
                            worker_id=int(worker_result["worker_id"]),
                            worker_total=worker_count,
                        )
            _drain_parallel_arena_events(
                event_queue, args, progress_path, base_seed, completed_results, eta_tracker
            )
    finally:
        manager.shutdown()
    if len(completed_results) != args.games:
        raise RuntimeError(
            f"Arena completed {len(completed_results)}/{args.games} games; checkpoint preserved"
        )
    return _finish_arena(args, completed_results, workers=configured_worker_count)


def _arena_worker(job: dict[str, object], event_queue: object) -> dict[str, object]:
    args = argparse.Namespace(**dict(job["args"]))  # type: ignore[arg-type]
    _verify_arena_models(args)
    worker_id = int(job["worker_id"])
    worker_total = int(job["worker_total"])
    base_seed = int(job["base_seed"])
    a_net = load_network(
        args.a_model, args.a_backend, args.a_device, args.a_channels, args.a_blocks,
        args.a_inference_amp,
        args.a_inference_tf32,
    )
    b_net = load_network(
        args.b_model, args.b_backend, args.b_device, args.b_channels, args.b_blocks,
        args.b_inference_amp,
        args.b_inference_tf32,
    )
    _warm_network(a_net)
    _warm_network(b_net)
    _verify_arena_models(args)
    wins = {args.a_name: 0, args.b_name: 0}
    illegal = {args.a_name: 0, args.b_name: 0}
    timeouts = {args.a_name: 0, args.b_name: 0}
    elapsed_total = {args.a_name: 0.0, args.b_name: 0.0}
    move_count = {args.a_name: 0, args.b_name: 0}
    simulations_total = {args.a_name: 0, args.b_name: 0}
    color_games = {"black": 0, "white": 0}
    color_wins = {"black": 0, "white": 0}
    completed = 0
    game_results: list[dict[str, object]] = []
    for game_value in job["game_ids"]:  # type: ignore[union-attr]
        game = int(game_value)
        result = _play_parallel_arena_game(
            args,
            a_net,
            b_net,
            game,
            _arena_game_seed(base_seed, game, getattr(args, "paired_seeds", True)),
        )
        completed += 1
        winner = str(result["winner"])
        wins[winner] += 1
        color = "black" if bool(result["a_is_black"]) else "white"
        color_games[color] += 1
        if winner == args.a_name:
            color_wins[color] += 1
        for name in (args.a_name, args.b_name):
            elapsed_total[name] += float(result["elapsed_total"][name])  # type: ignore[index]
            move_count[name] += int(result["move_count"][name])  # type: ignore[index]
            simulations_total[name] += int(result["simulations_total"][name])  # type: ignore[index]
        if result["illegal"] is not None:
            illegal[str(result["illegal"])] += 1
        if result["timeout"] is not None:
            timeouts[str(result["timeout"])] += 1
        game_results.append({"game": game, "result": result})
        try:
            event_queue.put(
                {
                    "worker_id": worker_id,
                    "worker_total": worker_total,
                    "game": game,
                    "winner": winner,
                    "plies": result["plies"],
                    "result": result,
                }
            )
        except Exception:
            pass
    return {
        "worker_id": worker_id,
        "games": completed,
        "results": game_results,
        "wins": wins,
        "illegal": illegal,
        "timeouts": timeouts,
        "elapsed_total": elapsed_total,
        "move_count": move_count,
        "simulations_total": simulations_total,
        "color_games": color_games,
        "color_wins": color_wins,
    }


def _verify_arena_models(args: argparse.Namespace) -> None:
    expected = getattr(args, "_arena_model_sha256", None)
    current = {"a": _file_sha256(args.a_model), "b": _file_sha256(args.b_model)}
    if expected is not None and current != expected:
        raise ValueError("Arena model files changed during evaluation")


def _arena_opening(seed: int, plies: int) -> tuple[tuple[int, ...], int, list[int]]:
    rng = random.Random(seed ^ 0x4E4F474F)
    board = new_board()
    to_play = BLACK
    moves: list[int] = []
    for _ in range(plies):
        legal = legal_moves(board, to_play)
        if not legal:
            raise ValueError("Arena opening reached a terminal position")
        move = rng.choice(legal)
        board = play_move(board, to_play, move)
        moves.append(move)
        to_play = -to_play
    return board, to_play, moves


def _play_parallel_arena_game(
    args: argparse.Namespace,
    a_net: object,
    b_net: object,
    game: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    a_is_black = game % 2 == 1
    board, to_play, opening_moves = _arena_opening(seed, getattr(args, "opening_plies", 0))
    moves = list(opening_moves)
    winner_name = args.b_name
    plies = len(opening_moves)
    game_elapsed = {args.a_name: 0.0, args.b_name: 0.0}
    move_count = {args.a_name: 0, args.b_name: 0}
    simulations_total = {args.a_name: 0, args.b_name: 0}
    illegal_name: str | None = None
    timeout_name: str | None = None
    a_inference_batch_size = args.a_inference_batch_size or args.inference_batch_size
    b_inference_batch_size = args.b_inference_batch_size or args.inference_batch_size
    black_seed = rng.randrange(1 << 30)
    white_seed = rng.randrange(1 << 30)
    a_seed = black_seed if a_is_black else white_seed
    b_seed = white_seed if a_is_black else black_seed
    searchers = {
        args.a_name: NeuralMCTS(
            network=a_net,  # type: ignore[arg-type]
            simulations=args.simulations,
            time_limit=args.time_limit,
            cpuct=args.cpuct,
            seed=a_seed,
            root_tactical_weight=args.root_tactical_weight,
            endgame_moves=args.endgame_moves,
            endgame_nodes=args.endgame_nodes,
            endgame_time_fraction=args.endgame_time_fraction,
            inference_batch_size=a_inference_batch_size,
            fixed_simulations=getattr(args, "fixed_simulations", False),
        ),
        args.b_name: NeuralMCTS(
            network=b_net,  # type: ignore[arg-type]
            simulations=args.simulations,
            time_limit=args.time_limit,
            cpuct=args.cpuct,
            seed=b_seed,
            root_tactical_weight=args.root_tactical_weight,
            endgame_moves=args.endgame_moves,
            endgame_nodes=args.endgame_nodes,
            endgame_time_fraction=args.endgame_time_fraction,
            inference_batch_size=b_inference_batch_size,
            fixed_simulations=getattr(args, "fixed_simulations", False),
        ),
    }
    while plies < args.max_plies:
        if not legal_moves(board, to_play):
            current_name = _side_name(to_play, a_is_black, args.a_name, args.b_name)
            winner_name = args.b_name if current_name == args.a_name else args.a_name
            break
        current_name = _side_name(to_play, a_is_black, args.a_name, args.b_name)
        decision_started_at = time.perf_counter()
        move_time_limit = _arena_move_time_limit(
            args.time_limit,
            args.game_time_limit,
            args.time_reserve,
            game_elapsed[current_name],
            board.count(0),
        )
        searcher = searchers[current_name]
        searcher.time_limit = move_time_limit
        # Arena is an evaluation harness, so exploration noise must not alter
        # the policy being measured. Self-play enables it separately.
        search_result = searcher.search(board, to_play, add_root_noise=False)
        move = search_result.move
        next_board = (
            play_move(board, to_play, move)
            if move is not None and is_legal_move(board, to_play, move)
            else None
        )
        game_elapsed[current_name] += time.perf_counter() - decision_started_at
        move_count[current_name] += 1
        simulations_total[current_name] += search_result.visits
        if args.game_time_limit > 0.0 and game_elapsed[current_name] > args.game_time_limit:
            timeout_name = current_name
            winner_name = args.b_name if current_name == args.a_name else args.a_name
            break
        if next_board is None:
            illegal_name = current_name
            winner_name = args.b_name if current_name == args.a_name else args.a_name
            break
        board = next_board
        moves.append(move)
        to_play = -to_play
        plies += 1
    else:
        winner_name = _side_name(-to_play, a_is_black, args.a_name, args.b_name)
    return {
        "winner": winner_name,
        "a_is_black": a_is_black,
        "opening_moves": opening_moves,
        "moves": moves,
        "plies": plies,
        "illegal": illegal_name,
        "timeout": timeout_name,
        "elapsed_total": game_elapsed,
        "move_count": move_count,
        "simulations_total": simulations_total,
    }


def _drain_parallel_arena_events(
    event_queue: object,
    args: argparse.Namespace,
    progress_path: Path | None,
    base_seed: int,
    completed_results: dict[int, dict[str, object]],
    eta_tracker: _EtaTracker,
) -> None:
    while True:
        try:
            event = event_queue.get_nowait()  # type: ignore[attr-defined]
        except Empty:
            break
        except Exception:
            break
        _record_arena_result(
            args,
            progress_path,
            base_seed,
            completed_results,
            int(event["game"]),
            dict(event["result"]),
            eta_tracker,
            worker_id=int(event["worker_id"]),
            worker_total=int(event["worker_total"]),
        )


def _write_arena_result(
    args: argparse.Namespace,
    games: int,
    wins: dict[str, int],
    illegal: dict[str, int],
    timeouts: dict[str, int],
    elapsed_total: dict[str, float],
    move_count: dict[str, int],
    simulations_total: dict[str, int],
    workers: int,
    color_games: dict[str, int] | None = None,
    color_wins: dict[str, int] | None = None,
) -> None:
    _verify_arena_models(args)
    a_rate = wins[args.a_name] / games if games else 0.0
    b_rate = wins[args.b_name] / games if games else 0.0
    a_model_sha256 = _file_sha256(args.a_model)
    b_model_sha256 = _file_sha256(args.b_model)
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "games": games,
        "workers": workers,
        "fixed_simulations": getattr(args, "fixed_simulations", False),
        "opening_plies": getattr(args, "opening_plies", 0),
        "a_name": args.a_name,
        "b_name": args.b_name,
        "a_model_sha256": a_model_sha256,
        "b_model_sha256": b_model_sha256,
        "same_model": (
            a_model_sha256 is not None and a_model_sha256 == b_model_sha256
        ),
        "a_wins": wins[args.a_name],
        "b_wins": wins[args.b_name],
        "a_win_rate": a_rate,
        "b_win_rate": b_rate,
        "a_win_rate_wilson_95": _wilson_interval(wins[args.a_name], games),
        "a_black_games": (color_games or {}).get("black", 0),
        "a_white_games": (color_games or {}).get("white", 0),
        "a_black_wins": (color_wins or {}).get("black", 0),
        "a_white_wins": (color_wins or {}).get("white", 0),
        "a_black_win_rate": (color_wins or {}).get("black", 0) / max(1, (color_games or {}).get("black", 0)),
        "a_white_win_rate": (color_wins or {}).get("white", 0) / max(1, (color_games or {}).get("white", 0)),
        "a_illegal": illegal[args.a_name],
        "b_illegal": illegal[args.b_name],
        "a_timeouts": timeouts[args.a_name],
        "b_timeouts": timeouts[args.b_name],
        "a_avg_move_seconds": elapsed_total[args.a_name] / max(1, move_count[args.a_name]),
        "b_avg_move_seconds": elapsed_total[args.b_name] / max(1, move_count[args.b_name]),
        "a_avg_simulations": simulations_total[args.a_name] / max(1, move_count[args.a_name]),
        "b_avg_simulations": simulations_total[args.b_name] / max(1, move_count[args.b_name]),
        "a_simulations_per_second": simulations_total[args.a_name] / max(1.0e-9, elapsed_total[args.a_name]),
        "b_simulations_per_second": simulations_total[args.b_name] / max(1.0e-9, elapsed_total[args.b_name]),
        "a_inference_batch_size": args.a_inference_batch_size or args.inference_batch_size,
        "b_inference_batch_size": args.b_inference_batch_size or args.inference_batch_size,
    }
    _write_json_atomic(result_path, result)


def run_save_model(args: argparse.Namespace) -> int:
    path = Path(args.path)
    TinyPolicyValueNet().save(path)
    print(f"saved {path}")
    return 0


def _make_progress_printer(log_path: str | None) -> Callable[[dict[str, object]], None]:
    def emit(event: dict[str, object]) -> None:
        line = _format_progress_event(event)
        if not line:
            return
        print(line, flush=True)
        _append_log(log_path, line)

    return emit


def _format_progress_event(event: dict[str, object]) -> str:
    name = str(event.get("event", ""))
    worker_id = event.get("worker_id")
    worker_total = event.get("worker_total")
    prefix = ""
    if worker_id is not None and worker_total is not None:
        prefix = f"[w{int(worker_id) + 1}/{int(worker_total)}] "
    if name == "self_play_start":
        return (
            f"{prefix}self-play start "
            f"games={event['games']} simulations={event['simulations']} "
            f"time_limit={float(event['time_limit']):.3f}s cpuct={float(event['cpuct']):.2f} "
            f"max_plies={event['max_plies']} temperature_moves={event['temperature_moves']}"
            f" batch={event.get('inference_batch_size', 32)}"
            f" dirichlet=({float(event.get('dirichlet_alpha', 0.0)):.3g},"
            f"{float(event.get('dirichlet_epsilon', 0.0)):.3g})"
            f" clean_labels={bool(event.get('clean_label_search', False))}"
            f" label={event.get('label_simulations', 0)}@"
            f"{float(event.get('label_time_limit', 0.0)):.3f}s"
        )
    if name == "self_play_game":
        worker_game = event.get("worker_game")
        worker_game_text = f" worker_game={worker_game}" if worker_game is not None else ""
        timing = ""
        if event.get("rate") is not None:
            timing += f" rate={float(event['rate']) * 60.0:.2f} games/min"
        if event.get("eta") is not None:
            timing += f" eta={_format_seconds(float(event['eta']))}"
        return (
            f"{prefix}game {event['game']}/{event['games']} "
            f"winner={event['winner']} plies={event['plies']} samples={event['samples']} "
             f"confidence_avg={_percent(float(event['avg_confidence']))} "
             f"entropy={float(event.get('avg_policy_entropy', 0.0)):.3f} "
             f"label_entropy={float(event.get('avg_label_policy_entropy', 0.0)):.3f} "
             f"tactical={event.get('tactical_decisions', 0)} "
             f"override={event.get('tactical_overrides', 0)} "
             f"immediate_loss={_percent(float(event.get('tactical_immediate_loss_rate', 0.0)))} "
             f"search={_format_seconds(float(event['search_seconds']))} elapsed={_format_seconds(float(event['elapsed']))} "
             f"action_sim/move={int(event.get('action_completed_simulations', int(event.get('completed_simulations', 0)) - int(event.get('label_completed_simulations', 0)))) / max(1, int(event['plies'])):.1f} "
             f"label_sim/move={int(event.get('label_completed_simulations', 0)) / max(1, int(event['plies'])):.1f} "
             f"infer={event.get('inference_positions', 0)}/{event.get('inference_calls', 0)} "
             f"score={event['black_wins']}-{event['white_wins']} last={event['last_move']}"
             f" open={','.join(str(value) for value in event.get('opening_unique', []))}"
             f"{worker_game_text}{timing}"
        )
    if name == "self_play_end":
        return (
            f"{prefix}self-play done "
            f"games={event['games']} samples={event['samples']} "
             f"wins black={event['black_wins']} white={event['white_wins']} "
             f"entropy={float(event.get('avg_policy_entropy', 0.0)):.3f} "
             f"label_entropy={float(event.get('avg_label_policy_entropy', 0.0)):.3f} "
             f"override={event.get('tactical_overrides', 0)} "
             f"action_sim/move={int(event.get('action_completed_simulations', int(event.get('completed_simulations', 0)) - int(event.get('label_completed_simulations', 0)))) / max(1, int(event.get('plies', 0))):.1f} "
             f"label_sim/move={int(event.get('label_completed_simulations', 0)) / max(1, int(event.get('plies', 0))):.1f} "
             f"immediate_loss={_percent(float(event.get('tactical_immediate_loss_rate', 0.0)))} "
             f"open={','.join(str(value) for value in event.get('opening_unique', []))} "
             f"elapsed={_format_seconds(float(event['elapsed']))}"
        )
    if name == "endgame_relabel_progress":
        return (
            f"endgame relabel {event['processed']}/{event['total']} "
            f"eligible={event['eligible']} solved={event['solved']} "
            f"value_changed={event['value_changed']} policy_changed={event['policy_changed']}"
        )
    if name == "tactical_sanitize_progress":
        return (
            f"tactical sanitize {event['processed']}/{event['total']} "
            f"changed={event['changed']} elapsed={_format_seconds(float(event['elapsed']))}"
        )
    if name == "train_start":
        return (
            "train start "
            f"backend={event['backend']} device={event.get('device', 'cpu')} samples={event['samples']} "
            f"validation={event.get('validation_samples', 0)} "
            f"epochs={event['epochs']} batch_size={event['batch_size']} "
            f"batches_per_epoch={event['batches_per_epoch']} "
            f"lr={float(event['learning_rate']):.5g}->{float(event.get('minimum_learning_rate', 0.0)):.5g} "
            f"channels={event.get('channels', '-')} blocks={event.get('blocks', '-')} "
            f"heads={event.get('input_planes', '-')}/{event.get('policy_channels', '-')}/"
            f"{event.get('value_channels', '-')}/{event.get('value_hidden', '-')} "
             f"norm={event.get('norm_type', 'batch')} "
             f"balance_colors={event.get('balance_policy_colors', False)} "
             f"policy_color_weight={float(event.get('black_policy_weight', 0.0)):.0f}/"
             f"{float(event.get('white_policy_weight', 0.0)):.0f} "
             f"scale={float(event.get('policy_color_scale_black', 1.0)):.4f}/"
             f"{float(event.get('policy_color_scale_white', 1.0)):.4f} "
             f"balanced={float(event.get('balanced_black_policy_weight', 0.0)):.0f}/"
             f"{float(event.get('balanced_white_policy_weight', 0.0)):.0f} "
             f"checkpoint_loaded={bool(event.get('checkpoint_loaded', False))} "
             f"checkpoint={event.get('checkpoint', 'none')}"
            f" amp={event.get('amp', False)} gpu={event.get('gpu', 'none')}"
        )
    if name == "train_baseline":
        return (
            f"validation baseline={float(event['validation_loss']):.6f} "
            f"best={float(event['best_validation_loss']):.6f} epoch={event['best_epoch']}"
        )
    if name == "train_prepare":
        return (
            f"prepare[{event.get('phase', 'train')}] {event['samples_prepared']}/{event['samples_total']} "
            f"elapsed={_format_seconds(float(event['elapsed']))}"
        )
    if name == "train_transfer":
        return (
            f"train data transfer mode={event.get('mode', 'cpu')} "
            f"bytes={_format_bytes(int(event.get('bytes', 0)))} device={event.get('device', 'cpu')}"
        )
    if name == "train_batch":
        eta = event.get("eta")
        eta_text = f" eta={_format_seconds(float(eta))}" if eta is not None else " eta=estimating"
        return (
            f"batch {event['batch']}/{event['batches']} "
            f"loss={float(event['loss']):.5f} policy={float(event['policy_loss']):.5f} "
            f"value={float(event['value_loss']):.5f} "
            f"samples={event['samples_seen']}/{event['samples_total']} "
            f"speed={float(event['samples_per_sec']):.1f}/s "
            f"batch={_format_seconds(float(event['batch_elapsed']))}{eta_text}"
        )
    if name == "train_epoch":
        gpu_text = ""
        validation_text = ""
        if event.get("validation_loss") is not None:
            validation_text = (
                f" val={float(event['validation_loss']):.5f}"
                f"/{float(event['validation_policy_loss']):.5f}"
                f"/{float(event['validation_value_loss']):.5f}"
            )
        if "gpu_peak" in event:
            gpu_text = (
                f" gpu={_format_bytes(int(event['gpu_allocated']))}/"
                f"{_format_bytes(int(event['gpu_reserved']))} "
                f"peak={_format_bytes(int(event['gpu_peak']))}"
            )
            if event.get("gpu_utilization") is not None:
                gpu_text += f" util={int(event['gpu_utilization'])}%"
        return (
            f"epoch {event['epoch']}/{event['epochs']} "
            f"loss={float(event['loss']):.5f} policy={float(event['policy_loss']):.5f} "
            f"value={float(event['value_loss']):.5f} batches={event['batches']} "
            f"lr={float(event.get('learning_rate', 0.0)):.5g}{validation_text} "
            f"elapsed={_format_seconds(float(event['elapsed']))} "
            f"speed={float(event.get('samples_per_sec', 0.0)):.1f}/s"
            f" eta={_format_seconds(float(event['eta'])) if event.get('eta') is not None else 'estimating'}"
            f"{gpu_text}"
        )
    if name == "train_end":
        best_text = ""
        if event.get("best_validation_loss") is not None:
            best_text = (
                f" best_val={float(event['best_validation_loss']):.5f}"
                f" best_epoch={event.get('best_epoch', 0)}"
            )
        return (
            "train done "
            f"output={event['output']} loss={float(event['loss']):.5f} "
            f"policy={float(event.get('policy_loss', 0.0)):.5f} value={float(event.get('value_loss', 0.0)):.5f} "
            f"elapsed={_format_seconds(float(event['elapsed']))}{best_text}"
        )
    if name == "arena_game":
        rate = event.get("rate")
        eta = event.get("eta")
        timing = ""
        if rate is not None:
            timing += f" rate={float(rate) * 60.0:.2f} games/min"
        if eta is not None:
            timing += f" eta={_format_seconds(float(eta))}"
        game_id = event.get("game_id")
        game_id_text = f" id={game_id}" if game_id is not None else ""
        return (
            f"arena game {event['game']}/{event['games']}{game_id_text} winner={event['winner']} "
            f"plies={event['plies']} score {event['a_name']}={event['score_a']} "
            f"{event['b_name']}={event['score_b']}{timing}"
        )
    return ""


def _drain_progress_events(
    event_queue: object,
    log_path: str | None,
    quiet: bool,
    *,
    eta_tracker: _EtaTracker | None = None,
    total_games: int | None = None,
    progress_state: dict[str, object] | None = None,
) -> None:
    while True:
        try:
            event = event_queue.get_nowait()  # type: ignore[attr-defined]
        except Empty:
            break
        except Exception:
            break
        if (
            eta_tracker is not None
            and total_games is not None
            and progress_state is not None
            and event.get("event") == "self_play_game"
        ):
            worker_id = event.get("worker_id")
            local_game = event.get("game")
            seen = progress_state.get("seen")
            if worker_id is not None and local_game is not None and isinstance(seen, set):
                key = (int(worker_id), int(local_game))
                if key in seen:
                    continue
                seen.add(key)
            completed = int(progress_state.get("completed", 0)) + 1
            progress_state["completed"] = completed
            eta_tracker.add()
            event = dict(event)
            event["worker_game"] = event.get("game")
            event["game"] = completed
            event["games"] = total_games
            event["rate"] = eta_tracker.rate
            event["eta"] = eta_tracker.estimate(total_games - completed)
        line = _format_progress_event(event)
        if not line:
            continue
        if not quiet:
            print(line, flush=True)
        _append_log(log_path, line)


def _append_log(log_path: str | None, line: str) -> None:
    if not log_path:
        return
    path = Path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            fh.write(f"{stamp} {line}\n")
    except OSError:
        pass


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _wilson_interval(wins: int, games: int, z: float = 1.96) -> list[float]:
    if games <= 0:
        return [0.0, 1.0]
    rate = wins / games
    z2 = z * z
    denominator = 1.0 + z2 / games
    center = (rate + z2 / (2.0 * games)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / games + z2 / (4.0 * games * games))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _format_seconds(value: float) -> str:
    if value < 1.0:
        return f"{1000.0 * value:.0f}ms"
    return f"{value:.2f}s"


def _file_sha256(path: str | None) -> str | None:
    """Return a model fingerprint for reproducible Arena reports."""
    if not path:
        return None
    model_path = Path(path)
    if not model_path.exists() or not model_path.is_file():
        return None
    digest = hashlib.sha256()
    with model_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.2f}GiB"
    return f"{value / 1024**2:.0f}MiB"


def _value_to_winrate(value: float) -> float:
    return max(0.0, min(1.0, (value + 1.0) * 0.5))


def _side_name(to_play: int, a_is_black: bool, a_name: str, b_name: str) -> str:
    if to_play == BLACK:
        return a_name if a_is_black else b_name
    return b_name if a_is_black else a_name


def _arena_move_time_limit(
    per_move_limit: float,
    game_time_limit: float,
    time_reserve: float,
    spent: float,
    empty_points: int,
) -> float:
    from .protocol import _allocate_search_time

    return _allocate_search_time(per_move_limit, game_time_limit, time_reserve, spent, empty_points)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "engine"
    try:
        if command == "engine":
            return run_engine(args)
        if command == "benchmark":
            return run_benchmark(args)
        if command == "self-play":
            return run_self_play(args)
        if command == "generate-data":
            return run_generate_data(args)
        if command == "train":
            return run_train(args)
        if command == "upgrade-model":
            return run_upgrade_model(args)
        if command == "arena":
            return run_arena(args)
        if command == "save-model":
            return run_save_model(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {command}")
    return 2
