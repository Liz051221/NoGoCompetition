"""Portable SAU engine. Answer name? before importing inference libraries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def configuration(root: Path) -> dict:
    config = json.loads((root / "engine_config.json").read_text(encoding="utf-8-sig"))
    model = root / "models" / "champion.onnx"
    actual = hashlib.sha256(model.read_bytes()).hexdigest()
    if actual != config["model_sha256"]:
        raise ValueError("model checksum mismatch; extract the complete package again")
    return config


def make_engine(root: Path, config: dict):
    from nogo_ai.competition_runtime import CompetitionMCTS
    from nogo_ai.portable_onnx import PortableOnnxNetwork
    from nogo_ai.protocol import EngineConfig, SAUNoGoEngine

    network = PortableOnnxNetwork(root / "models" / "champion.onnx", config["cpu_threads"])
    if network.metadata.get("source_checkpoint_sha256") != config["source_checkpoint_sha256"]:
        raise ValueError("source model identity mismatch")
    searcher = CompetitionMCTS(
        network, simulations=config["simulations"], time_limit=config["time_limit"],
        cpuct=config["cpuct"], endgame_moves=config["endgame_moves"],
        endgame_nodes=config["endgame_nodes"], endgame_time_fraction=.35,
        root_tactical_weight=config["root_tactical_weight"],
        inference_batch_size=config["inference_batch_size"],
    )
    try:
        log_dir = root / "logs"
        log_dir.mkdir(exist_ok=True)
    except OSError:
        log_dir = Path(tempfile.gettempdir())
    os.environ["NOGO_AI_LOG"] = str(log_dir / f"nogo_engine_{os.getpid()}.log")
    return SAUNoGoEngine(searcher, EngineConfig(
        name=config["name"], game_time_limit=config["game_time_limit"],
        time_reserve=config["time_reserve"],
    ))


def main() -> int:
    # The platform commands are ASCII; UTF-8 also tolerates diagnostic BOMs
    # independently of the receiving computer's Windows locale.
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    default_root = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--move-seconds", type=float, help="Local GUI search cap; original config stays unchanged")
    parser.add_argument("--game-seconds", type=float, help="Local referee total clock")
    parser.add_argument("--reserve-seconds", type=float, help="Time kept out of neural search")
    parser.add_argument("--search-worker", action="store_true")
    options = parser.parse_args()
    root = options.root.resolve()
    # Reading the small configuration is cheap; model loading waits until play.
    config = json.loads((root / "engine_config.json").read_text(encoding="utf-8-sig"))
    if options.search_worker:
        from nogo_ai.competition_runtime import worker_loop
        worker_loop(make_engine(root, configuration(root)).searcher, sys.stdin, sys.stdout)
        return 0
    if options.move_seconds is not None:
        if not 0.1 <= options.move_seconds <= 60:
            parser.error("--move-seconds must be between 0.1 and 60")
        config["time_limit"] = options.move_seconds
    if options.game_seconds is not None:
        if not 0 < options.game_seconds <= 86400:
            parser.error("--game-seconds must be between 0 and 86400")
        config["game_time_limit"] = options.game_seconds
    if options.reserve_seconds is not None:
        if not 0 <= options.reserve_seconds < config["game_time_limit"]:
            parser.error("--reserve-seconds must be below the total game clock")
        config["time_reserve"] = options.reserve_seconds
    engine = None
    if options.self_test:
        from nogo_ai.rules import BLACK, new_board
        config = configuration(root)
        engine = make_engine(root, config)
        prediction = engine.searcher.network.predict(new_board(), BLACK)
        if abs(sum(prediction.priors) - 1.0) > 1e-5:
            raise ValueError("self-test policy normalization failed")
        print(json.dumps({"ok": True, "name": config["name"], "provider": "CPUExecutionProvider",
                          "source_checkpoint_sha256": config["source_checkpoint_sha256"],
                          "model_sha256": config["model_sha256"], "value": prediction.value,
                          "policy_sum": sum(prediction.priors)}), flush=True)
        return 0
    from nogo_ai.competition_runtime import CompetitionProtocol, GuardedSearcher, run_protocol
    from nogo_ai.protocol import EngineConfig
    command = [sys.executable]
    if not getattr(sys, "frozen", False):
        command += ["-m", "tools.portable_engine"]
    command += ["--root", str(root), "--search-worker"]
    try:
        log_dir = root / "logs"
        log_dir.mkdir(exist_ok=True)
    except OSError:
        log_dir = Path(tempfile.gettempdir())
    os.environ["NOGO_AI_LOG"] = str(log_dir / f"nogo_engine_{os.getpid()}.log")
    searcher = GuardedSearcher(command, config["time_limit"], log_dir)
    engine = CompetitionProtocol(searcher, EngineConfig(name=config["name"],
                                 game_time_limit=config["game_time_limit"], time_reserve=config["time_reserve"]))
    try:
        run_protocol(engine, sys.stdin, sys.stdout)
    finally:
        searcher.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"NoGo engine could not start: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
