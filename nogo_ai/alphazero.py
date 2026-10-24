from __future__ import annotations

import json
import hashlib
import math
import os
import random
import time
import copy
import concurrent.futures
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .endgame import EndgameSolver, move_allows_immediate_loss
from .mcts import NeuralMCTS
from .network import TinyPolicyValueNet
from .rules import BLACK, POINTS, Board, color_name, index_to_coord, legal_moves, new_board, play_move
from .symmetry import transform_board, transform_index, transform_policy

ProgressCallback = Callable[[dict[str, object]], None]


def _estimate_remaining_seconds(
    recent_durations: Iterable[float],
    remaining: int,
) -> float | None:
    """Estimate work time from a bounded, recent duration window."""
    if remaining <= 0:
        return 0.0
    durations = sorted(
        float(duration)
        for duration in recent_durations
        if math.isfinite(float(duration)) and float(duration) > 0.0
    )
    if not durations:
        return None
    middle = len(durations) // 2
    if len(durations) % 2:
        typical = durations[middle]
    else:
        typical = (durations[middle - 1] + durations[middle]) / 2.0
    return typical * remaining


@dataclass(slots=True)
class TrainingSample:
    board: Board
    to_play: int
    policy: list[float]
    value: float
    sample_weight: float = 1.0
    policy_weight: float = 1.0
    value_weight: float = 1.0

    def __post_init__(self) -> None:
        if (not all(math.isfinite(weight) and weight >= 0.0
                    for weight in (self.policy_weight, self.value_weight))
                or self.policy_weight + self.value_weight <= 0.0):
            raise ValueError("sample head weights must be finite, non-negative and not both zero")

    def to_json(self) -> dict[str, object]:
        payload = {
            "board": list(self.board),
            "to_play": self.to_play,
            "policy": self.policy,
            "value": self.value,
        }
        if self.sample_weight != 1.0:
            payload["sample_weight"] = self.sample_weight
        if self.policy_weight != 1.0:
            payload["policy_weight"] = self.policy_weight
        if self.value_weight != 1.0:
            payload["value_weight"] = self.value_weight
        return payload

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "TrainingSample":
        return cls(
            board=tuple(int(v) for v in data["board"]),  # type: ignore[index]
            to_play=int(data["to_play"]),
            policy=[float(v) for v in data["policy"]],  # type: ignore[index]
            value=float(data["value"]),
            sample_weight=max(0.01, float(data.get("sample_weight", 1.0))),
            policy_weight=float(data.get("policy_weight", 1.0)),
            value_weight=float(data.get("value_weight", 1.0)),
        )


def sample_manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".meta.json")


def parameter_fingerprint(parameters: dict[str, object] | None) -> str:
    payload = json.dumps(parameters or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def samples_fingerprint(samples: Iterable[TrainingSample]) -> str:
    """Return a stable digest for the exact labelled samples used to train.

    The digest intentionally includes board, side, policy, value, and sample
    and per-head weights in their stored order. It is used by training checkpoints to
    reject accidental resume with a different replay mix or changed labels.
    """
    digest = hashlib.sha256()
    for sample in samples:
        payload = json.dumps(
            sample.to_json(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()[:16]


def _write_manifest(
    path: Path,
    sample_count: int,
    data_sha256: str,
    metadata: dict[str, object] | None,
) -> None:
    manifest_path = sample_manifest_path(path)
    payload = {
        "format_version": 1,
        "sample_count": sample_count,
        "data_sha256": data_sha256,
        "parameter_fingerprint": parameter_fingerprint(metadata),
        "metadata": metadata or {},
    }
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)


def write_samples(
    path: str | Path,
    samples: Iterable[TrainingSample],
    metadata: dict[str, object] | None = None,
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8", newline="\n") as fh:
        for sample in samples:
            line = json.dumps(sample.to_json(), separators=(",", ":")) + "\n"
            fh.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    sample_manifest_path(path).unlink(missing_ok=True)
    os.replace(temporary, path)
    _write_manifest(path, count, digest.hexdigest(), metadata)
    return count


def write_sample_parts(
    path: str | Path,
    part_paths: Iterable[str | Path],
    metadata: dict[str, object] | None = None,
) -> int:
    """Atomically merge worker JSONL files and emit a verifiable manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    digest = hashlib.sha256()
    with temporary.open("wb") as out:
        for part_path in part_paths:
            with Path(part_path).open("rb") as part:
                for chunk in iter(lambda: part.read(1024 * 1024), b""):
                    out.write(chunk)
                    digest.update(chunk)
                    count += chunk.count(b"\n")
    sample_manifest_path(path).unlink(missing_ok=True)
    os.replace(temporary, path)
    _write_manifest(path, count, digest.hexdigest(), metadata)
    return count


def read_sample_manifest(path: str | Path) -> dict[str, object] | None:
    manifest_path = sample_manifest_path(path)
    if not manifest_path.exists():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid sample manifest: {manifest_path}")
    return data


def read_samples(path: str | Path) -> list[TrainingSample]:
    path = Path(path)
    samples: list[TrainingSample] = []
    digest = hashlib.sha256()
    with path.open("rb") as raw:
        data = raw.read()
    digest.update(data)
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(TrainingSample.from_json(json.loads(line)))
    manifest = read_sample_manifest(path)
    if manifest is not None:
        expected_count = int(manifest.get("sample_count", -1))
        expected_hash = str(manifest.get("data_sha256", ""))
        if expected_count != len(samples) or expected_hash != digest.hexdigest():
            raise ValueError(
                f"sample manifest mismatch for {path}: "
                f"count={len(samples)}/{expected_count} sha256={digest.hexdigest()[:16]}/{expected_hash[:16]}"
            )
    return samples


def augment_samples_with_symmetries(samples: list[TrainingSample]) -> list[TrainingSample]:
    augmented: list[TrainingSample] = []
    for sample in samples:
        for transform in range(8):
            augmented.append(
                TrainingSample(
                    board=transform_board(sample.board, transform),
                    to_play=sample.to_play,
                    policy=transform_policy(sample.policy, transform),
                    value=sample.value,
                    sample_weight=sample.sample_weight,
                    policy_weight=sample.policy_weight,
                    value_weight=sample.value_weight,
                )
            )
    return augmented


def sanitize_policy_targets(samples: list[TrainingSample]) -> dict[str, int]:
    """Validate and normalize policy labels against the actual NoGo rules.

    Search labels should already be legal distributions, but replay files and
    externally generated data can contain stale, illegal, negative, or
    unnormalized probabilities.  Keeping those entries would teach the model
    probability mass on moves that MCTS can never play.
    """
    changed = 0
    dropped_terminal = 0
    uniform_filled = 0
    for sample in samples:
        if len(sample.board) != POINTS:
            raise ValueError(f"invalid board length: {len(sample.board)}")
        if any(value not in (-1, 0, 1) for value in sample.board):
            raise ValueError("board contains a value other than -1, 0, or 1")
        if sample.to_play not in (BLACK, -BLACK):
            raise ValueError(f"invalid side to play: {sample.to_play}")
        if len(sample.policy) != POINTS:
            raise ValueError(f"invalid policy length: {len(sample.policy)}")
        if not math.isfinite(sample.value):
            raise ValueError("sample value is not finite")
        sample.value = max(-1.0, min(1.0, float(sample.value)))
        legal = legal_moves(sample.board, sample.to_play)
        if not legal:
            dropped_terminal += 1
            continue

        normalized = [0.0] * POINTS
        total = 0.0
        for move in legal:
            probability = float(sample.policy[move])
            if math.isfinite(probability) and probability > 0.0:
                normalized[move] = probability
                total += probability
        if total <= 0.0:
            uniform_filled += 1
            probability = 1.0 / len(legal)
            for move in legal:
                normalized[move] = probability
        else:
            normalized = [probability / total for probability in normalized]
        if any(abs(left - right) > 1.0e-7 for left, right in zip(sample.policy, normalized)):
            changed += 1
        sample.policy = normalized

    if dropped_terminal:
        samples[:] = [sample for sample in samples if legal_moves(sample.board, sample.to_play)]
    return {
        "changed": changed,
        "dropped_terminal": dropped_terminal,
        "uniform_filled": uniform_filled,
    }


def sanitize_tactical_policy_targets(
    samples: list[TrainingSample],
    progress: ProgressCallback | None = None,
) -> int:
    """Remove immediate-loss probability from late-game policy targets."""
    changed = 0
    started_at = time.perf_counter()
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        # Match MCTS: inspect positions with at least 36 stones on the board.
        if len(sample.board) - sample.board.count(0) < 36:
            if progress and (index == 1 or index % 1000 == 0 or index == total):
                progress({
                    "event": "tactical_sanitize_progress",
                    "processed": index,
                    "total": total,
                    "changed": changed,
                    "elapsed": time.perf_counter() - started_at,
                })
            continue
        legal = legal_moves(sample.board, sample.to_play)
        normalized, safe, was_changed = _sanitize_tactical_policy(
            sample.board, sample.to_play, sample.policy, legal
        )
        if not was_changed:
            if progress and (index == 1 or index % 1000 == 0 or index == total):
                progress({
                    "event": "tactical_sanitize_progress",
                    "processed": index,
                    "total": total,
                    "changed": changed,
                    "elapsed": time.perf_counter() - started_at,
                })
            continue
        sample.policy = normalized
        changed += 1
        if progress and (index == 1 or index % 1000 == 0 or index == total):
            progress({
                "event": "tactical_sanitize_progress",
                "processed": index,
                "total": total,
                "changed": changed,
                "elapsed": time.perf_counter() - started_at,
            })

    return changed


def _sanitize_tactical_policy(
    board: Board,
    to_play: int,
    policy: list[float],
    legal: list[int] | None = None,
) -> tuple[list[float], list[int], bool]:
    """Return a policy with immediate-loss moves removed when possible."""
    legal = legal if legal is not None else legal_moves(board, to_play)
    filled = len(board) - board.count(0)
    # Keep this threshold aligned with the late-game tactical scan in MCTS.
    # Replay data can come from an older searcher, so training must repair
    # unsafe labels even when the current search would already filter them.
    late_tactical = filled >= 36 and len(legal) <= 48
    if len(legal) <= 1 or (len(legal) > 24 and not late_tactical):
        return list(policy), legal, False
    # Most search labels are sparse. Check positive-mass moves first; zero
    # probability moves cannot affect the normalized target unless every
    # positive move is unsafe and a uniform safe fallback is required.
    # The visit distribution can contain dozens of one-visit tail moves. A
    # bounded high-mass scan keeps preprocessing tractable; if a dangerous
    # move is found, we expand the scan to all positive-mass moves below.
    positive = [
        move
        for move in legal
        if math.isfinite(float(policy[move])) and float(policy[move]) > 0.0
    ]
    if not positive:
        return list(policy), legal, False
    ordered_positive = sorted(positive, key=lambda move: float(policy[move]), reverse=True)
    scan = ordered_positive[:12]
    losing_positive = [
        move
        for move in scan
        if move_allows_immediate_loss(board, to_play, move)
    ]
    if not losing_positive:
        return list(policy), legal, False

    # A tactical hit in the high-mass portion means the remaining support is
    # worth checking too, so the repair does not leave a known unsafe tail.
    checked = set(scan)
    losing_positive.extend(
        move
        for move in positive
        if move not in checked and move_allows_immediate_loss(board, to_play, move)
    )
    losing_set = set(losing_positive)
    safe_positive = [move for move in positive if move not in losing_set]
    if safe_positive:
        safe_total = sum(max(0.0, float(policy[move])) for move in safe_positive)
        allowed = set(safe_positive)
        return [
            (max(0.0, float(value)) / safe_total) if move in allowed else 0.0
            for move, value in enumerate(policy)
        ], safe_positive, True

    safe = [
        move
        for move in legal
        if not move_allows_immediate_loss(board, to_play, move)
    ]
    if not safe:
        return list(policy), safe, False
    safe_probability = 1.0 / len(safe)
    safe_set = set(safe)
    return [
        safe_probability if move in safe_set else 0.0
        for move in range(POINTS)
    ], safe, True


def _normalize_legal_policy(policy: list[float], legal: list[int]) -> list[float]:
    """Return a finite distribution supported exactly on the legal moves."""
    if len(policy) != POINTS:
        raise ValueError(f"policy length must be {POINTS}, got {len(policy)}")
    legal_moves_set = sorted({move for move in legal if 0 <= move < POINTS})
    normalized = [0.0] * POINTS
    total = 0.0
    for move in legal_moves_set:
        probability = float(policy[move])
        if math.isfinite(probability) and probability > 0.0:
            normalized[move] = probability
            total += probability
    if not legal_moves_set:
        return normalized
    if total <= 0.0:
        probability = 1.0 / len(legal_moves_set)
        for move in legal_moves_set:
            normalized[move] = probability
        return normalized
    return [probability / total for probability in normalized]


def _policy_color_balance_scales(
    samples: list[TrainingSample],
    winner_policy_weight: float,
    loser_policy_weight: float,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return policy-only color scales and their unscaled effective weights.

    NoGo has a strong first-player bias. When winner/loser policy weights are
    asymmetric, balancing only the raw sample counts is insufficient: the
    weighted policy supervision must be balanced by side to move. Value labels
    intentionally do not use these scales.
    """
    totals = {BLACK: 0.0, -BLACK: 0.0}
    for sample in samples:
        outcome_weight = winner_policy_weight if sample.value > 0.0 else loser_policy_weight
        totals[sample.to_play] += sample.sample_weight * sample.policy_weight * outcome_weight
    positive_totals = [total for total in totals.values() if total > 0.0]
    scales = {BLACK: 1.0, -BLACK: 1.0}
    if positive_totals:
        target_total = sum(positive_totals) / len(positive_totals)
        for color, total in totals.items():
            if total > 0.0:
                scales[color] = target_total / total
    return scales, totals


def relabel_exact_endgames(
    samples: list[TrainingSample],
    max_legal_moves: int = 12,
    max_nodes: int = 200000,
    exact_weight: float = 2.0,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """Replace noisy late-game labels with exact NoGo minimax results.

    Proven losing moves are also removed from policy targets. Samples whose
    search budget is exhausted keep their original value target.
    """
    solver = EndgameSolver(max_legal_moves=max_legal_moves, max_nodes=max_nodes)
    stats = {"eligible": 0, "solved": 0, "value_changed": 0, "policy_changed": 0}
    sample_total = len(samples)

    def emit(index: int) -> None:
        if progress and (index == 1 or index % 1000 == 0 or index == sample_total):
            progress(
                {
                    "event": "endgame_relabel_progress",
                    "processed": index,
                    "total": sample_total,
                    **stats,
                }
            )

    for index, sample in enumerate(samples, start=1):
        legal = legal_moves(sample.board, sample.to_play)
        if not legal or len(legal) > max_legal_moves:
            emit(index)
            continue
        stats["eligible"] += 1
        result = solver.solve(sample.board, sample.to_play)
        if result is None:
            emit(index)
            continue

        if result.value != 0.0:
            stats["solved"] += 1
            if sample.value != result.value:
                sample.value = result.value
                stats["value_changed"] += 1
            sample.sample_weight = max(sample.sample_weight, exact_weight)

        allowed = [move for move in legal if move not in result.losing_moves]
        if not allowed:
            emit(index)
            continue
        policy_total = sum(max(0.0, sample.policy[move]) for move in allowed)
        if policy_total <= 0.0:
            emit(index)
            continue
        if any(sample.policy[move] > 0.0 for move in result.losing_moves):
            allowed_set = set(allowed)
            sample.policy = [
                max(0.0, probability) / policy_total if move in allowed_set else 0.0
                for move, probability in enumerate(sample.policy)
            ]
            stats["policy_changed"] += 1
        emit(index)
    return stats


def _relabel_endgame_chunk(
    chunk: list[TrainingSample],
    max_legal_moves: int,
    max_nodes: int,
    exact_weight: float,
) -> tuple[list[TrainingSample], dict[str, int]]:
    stats = relabel_exact_endgames(
        chunk,
        max_legal_moves=max_legal_moves,
        max_nodes=max_nodes,
        exact_weight=exact_weight,
    )
    return chunk, stats


def relabel_exact_endgames_parallel(
    samples: list[TrainingSample],
    max_legal_moves: int = 12,
    max_nodes: int = 200000,
    exact_weight: float = 2.0,
    workers: int = 1,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """Relabel independent sample ranges in CPU worker processes."""
    worker_count = min(max(1, workers), max(1, len(samples)))
    if worker_count == 1:
        return relabel_exact_endgames(
            samples,
            max_legal_moves=max_legal_moves,
            max_nodes=max_nodes,
            exact_weight=exact_weight,
            progress=progress,
        )

    chunk_size = (len(samples) + worker_count - 1) // worker_count
    ranges = [
        (start, min(len(samples), start + chunk_size))
        for start in range(0, len(samples), chunk_size)
    ]
    totals = {"eligible": 0, "solved": 0, "value_changed": 0, "policy_changed": 0}
    processed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_ranges = {
            executor.submit(
                _relabel_endgame_chunk,
                samples[start:end],
                max_legal_moves,
                max_nodes,
                exact_weight,
            ): (start, end)
            for start, end in ranges
        }
        for future in concurrent.futures.as_completed(future_ranges):
            start, end = future_ranges[future]
            chunk, stats = future.result()
            samples[start:end] = chunk
            processed += end - start
            for key in totals:
                totals[key] += int(stats[key])
            if progress:
                progress(
                    {
                        "event": "endgame_relabel_progress",
                        "processed": processed,
                        "total": len(samples),
                        **totals,
                    }
                )
    return totals


def split_training_samples(
    samples: list[TrainingSample],
    validation_fraction: float,
    seed: int | None = None,
    *,
    disjoint_positions: bool = False,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0.0 or len(samples) < 2:
        return samples, []
    validation_count = min(len(samples) - 1, max(1, round(len(samples) * validation_fraction)))

    # Self-play files contain each trajectory in move order. Splitting samples
    # independently leaks adjacent positions from the same game into both
    # sets and makes validation loss look better than true generalization.
    # A trajectory continues only when exactly one additional stone appears.
    groups: list[list[int]] = []
    current: list[int] = []
    previous: TrainingSample | None = None
    for index, sample in enumerate(samples):
        continues = previous is not None and _is_trajectory_successor(previous, sample)
        if current and not continues:
            groups.append(current)
            current = []
        current.append(index)
        previous = sample
    if current:
        groups.append(current)

    # A single trajectory cannot provide a leakage-free validation set.
    if len(groups) < 2:
        if disjoint_positions:
            raise ValueError("independent validation requires more than one trajectory")
        return samples, []

    random.Random(seed).shuffle(groups)
    validation_indices: set[int] = set()
    for group in groups:
        if len(validation_indices) >= validation_count:
            break
        if len(validation_indices) + len(group) >= len(samples):
            continue
        validation_indices.update(group)
    training = [sample for index, sample in enumerate(samples) if index not in validation_indices]
    validation = [sample for index, sample in enumerate(samples) if index in validation_indices]
    if disjoint_positions:
        training_keys = {canonical_training_position(sample) for sample in training}
        validation = [
            sample for sample in validation
            if canonical_training_position(sample) not in training_keys
        ]
        if not validation:
            raise ValueError("no independent validation positions remain; provide more diverse games")
    return training, validation


_POSITION_SYMMETRY_GATHERS = tuple(
    transform_board(tuple(range(POINTS)), symmetry) for symmetry in range(8)
)


def canonical_training_position(sample: TrainingSample) -> bytes:
    """Identify inputs equivalent under D4 and player-relative stone encoding."""
    relative = bytes(stone * sample.to_play + 1 for stone in sample.board)
    return min(
        bytes(relative[index] for index in gather)
        for gather in _POSITION_SYMMETRY_GATHERS
    )


def _is_trajectory_successor(previous: TrainingSample, current: TrainingSample) -> bool:
    if current.to_play != -previous.to_play:
        return False
    added = 0
    for before, after in zip(previous.board, current.board):
        if before == after:
            continue
        if before != 0 or after != previous.to_play:
            return False
        added += 1
    return added == 1


def generate_self_play_samples(
    network: object,
    games: int,
    simulations: int,
    time_limit: float,
    cpuct: float,
    max_plies: int,
    seed: int | None,
    temperature_moves: int = 12,
    dirichlet_alpha: float = 0.0,
    dirichlet_epsilon: float = 0.0,
    root_tactical_weight: float = 0.0,
    inference_batch_size: int = 32,
    opening_temperature: float = 1.0,
    opening_uniform_mix: float = 0.0,
    endgame_moves: int = 16,
    endgame_nodes: int = 200000,
    endgame_time_fraction: float = 0.35,
    clean_label_search: bool = False,
    label_simulations: int | None = None,
    label_time_limit: float | None = None,
    progress: ProgressCallback | None = None,
    fixed_simulations: bool = False,
    label_fixed_simulations: bool = False,
    initial_moves: tuple[int, ...] = (),
    record_search: bool = False,
    require_terminal: bool = False,
) -> tuple[list[TrainingSample], dict[str, object]]:
    if temperature_moves < 0:
        raise ValueError("temperature_moves must be non-negative")
    if not math.isfinite(opening_temperature) or opening_temperature <= 0.0:
        raise ValueError("opening_temperature must be finite and positive")
    if not math.isfinite(opening_uniform_mix) or not 0.0 <= opening_uniform_mix <= 1.0:
        raise ValueError("opening_uniform_mix must be between zero and one")
    if label_simulations is not None and label_simulations <= 0:
        raise ValueError("label_simulations must be positive")
    if label_fixed_simulations and not clean_label_search:
        raise ValueError("fixed label simulations require clean_label_search")
    if len(initial_moves) >= max_plies:
        raise ValueError("initial moves must leave room for self-play before max_plies")
    if label_time_limit is not None and (
        not math.isfinite(label_time_limit) or label_time_limit <= 0.0
    ):
        raise ValueError("label_time_limit must be finite and positive")
    effective_label_simulations = max(1, int(label_simulations or simulations))
    effective_label_time_limit = max(0.05, float(label_time_limit or time_limit))
    rng = random.Random(seed)
    samples: list[TrainingSample] = []
    black_wins = white_wins = 0
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
    tactical_decisions = 0
    tactical_immediate_losses = 0
    tactical_overrides = 0
    game_records: list[dict[str, object]] = []
    # Keep actual action coverage separate from the clean policy labels. This
    # makes it possible to detect an opening distribution collapsing to one
    # coordinate even when the stored labels look well formed.
    opening_move_counts: list[dict[str, int]] = [
        {} for _ in range(min(4, max(0, max_plies)))
    ]
    started_at = time.perf_counter()

    if progress:
        progress(
            {
                "event": "self_play_start",
                "games": games,
                "simulations": simulations,
                "time_limit": time_limit,
                "cpuct": cpuct,
                "max_plies": max_plies,
                "temperature_moves": temperature_moves,
                "opening_temperature": opening_temperature,
                "opening_uniform_mix": opening_uniform_mix,
                "dirichlet_alpha": dirichlet_alpha,
                "dirichlet_epsilon": dirichlet_epsilon,
                "inference_batch_size": inference_batch_size,
                "endgame_moves": endgame_moves,
                "endgame_nodes": endgame_nodes,
                "endgame_time_fraction": endgame_time_fraction,
                "clean_label_search": clean_label_search,
                "fixed_simulations": fixed_simulations,
                "label_fixed_simulations": label_fixed_simulations,
                "initial_moves": [index_to_coord(move) for move in initial_moves],
                "label_simulations": effective_label_simulations if clean_label_search else 0,
                "label_time_limit": effective_label_time_limit if clean_label_search else 0.0,
            }
        )

    for game in range(1, games + 1):
        game_started_at = time.perf_counter()
        board = new_board()
        to_play = BLACK
        moves: list[int] = []
        search_records: list[dict[str, object]] = []
        for move in initial_moves:
            board = play_move(board, to_play, move)
            to_play = -to_play
            moves.append(move)
        for ply, move in enumerate(initial_moves[:len(opening_move_counts)]):
            coordinate = index_to_coord(move)
            opening_move_counts[ply][coordinate] = opening_move_counts[ply].get(coordinate, 0) + 1
        trajectory: list[tuple[Board, int, list[float]]] = []
        winner = -to_play
        confidences: list[float] = []
        game_policy_entropy = 0.0
        game_tactical_decisions = 0
        game_tactical_immediate_losses = 0
        game_tactical_overrides = 0
        search_seconds = 0.0
        completed_simulations = 0
        label_search_seconds = 0.0
        label_completed_simulations = 0
        label_confidences: list[float] = []
        label_policy_entropy = 0.0
        last_move: int | None = None
        searcher = NeuralMCTS(
            network=network,  # type: ignore[arg-type]
            simulations=simulations,
            time_limit=time_limit,
            cpuct=cpuct,
            seed=rng.randrange(1 << 30),
            root_dirichlet_alpha=dirichlet_alpha,
            root_exploration_fraction=dirichlet_epsilon,
            root_tactical_weight=root_tactical_weight,
            endgame_moves=endgame_moves,
            endgame_nodes=endgame_nodes,
            endgame_time_fraction=endgame_time_fraction,
            inference_batch_size=inference_batch_size,
            fixed_simulations=fixed_simulations,
        )
        label_searcher: NeuralMCTS | None = None
        if clean_label_search:
            label_searcher = NeuralMCTS(
                network=network,  # type: ignore[arg-type]
                simulations=effective_label_simulations,
                time_limit=effective_label_time_limit,
                cpuct=cpuct,
                seed=rng.randrange(1 << 30),
                root_dirichlet_alpha=0.0,
                root_exploration_fraction=0.0,
                root_tactical_weight=root_tactical_weight,
                endgame_moves=endgame_moves,
                endgame_nodes=endgame_nodes,
                endgame_time_fraction=endgame_time_fraction,
                inference_batch_size=inference_batch_size,
                fixed_simulations=label_fixed_simulations,
            )
            # The trees must remain independent, but network predictions are
            # pure for a (board, side) pair. Reusing the action search cache
            # avoids evaluating the same root and leaf positions twice.
            if hasattr(searcher, "prediction_cache") and hasattr(label_searcher, "prediction_cache"):
                label_searcher.prediction_cache = searcher.prediction_cache

        for ply in range(len(initial_moves), max_plies):
            if not legal_moves(board, to_play):
                winner = -to_play
                break

            result = searcher.search(
                board,
                to_play,
                add_root_noise=ply < temperature_moves,
            )
            legal_before_move = legal_moves(board, to_play)
            policy_for_action, safe_tactical_moves, policy_was_filtered = _sanitize_tactical_policy(
                board,
                to_play,
                list(result.visit_policy),
                legal_before_move,
            )
            policy_for_action = _normalize_legal_policy(policy_for_action, legal_before_move)
            label_policy = policy_for_action
            label_result = None
            if clean_label_search:
                # The action search may contain root Dirichlet noise and is
                # intentionally exploratory. Generate the stored target with
                # an independent, noise-free search so exploration does not
                # become permanent policy supervision.
                label_result = label_searcher.search(  # type: ignore[union-attr]
                    board,
                    to_play,
                    add_root_noise=False,
                )
                label_policy, _safe_label_moves, _label_was_filtered = _sanitize_tactical_policy(
                    board,
                    to_play,
                    list(label_result.visit_policy),
                    legal_before_move,
                )
                label_policy = _normalize_legal_policy(label_policy, legal_before_move)
                label_confidences.append(label_result.confidence)
                label_policy_entropy += -sum(
                    probability * math.log(max(probability, 1.0e-12))
                    for probability in label_policy
                    if probability > 0.0
                )
                label_search_seconds += label_result.elapsed
                label_completed_simulations += label_result.visits
            confidences.append(result.confidence)
            entropy = -sum(
                probability * math.log(max(probability, 1.0e-12))
                for probability in policy_for_action
                if probability > 0.0
            )
            game_policy_entropy += entropy
            total_policy_entropy += entropy
            search_seconds += result.elapsed
            completed_simulations += result.visits
            move = (
                _choose_from_policy(
                    policy_for_action,
                    rng,
                    temperature=opening_temperature,
                    uniform_mix=opening_uniform_mix,
                    legal=legal_before_move,
                )
                if ply < temperature_moves
                else result.move
            )
            if move is None:
                winner = -to_play
                break

            # MCTS already filters tactical losses at the root. Keep this final
            # guard because opening sampling and partial endgame searches can
            # otherwise reintroduce a dominated move into the trajectory.
            if policy_was_filtered and safe_tactical_moves and move not in safe_tactical_moves:
                move = max(
                    safe_tactical_moves,
                    key=lambda candidate: (policy_for_action[candidate], -candidate),
                )
                tactical_overrides += 1
                game_tactical_overrides += 1

            # Measure, but do not alter, the generated move. Restricting this
            # diagnostic to tactical positions keeps large self-play runs fast.
            if len(legal_before_move) <= 24:
                tactical_decisions += 1
                game_tactical_decisions += 1
                if move_allows_immediate_loss(board, to_play, move):
                    tactical_immediate_losses += 1
                    game_tactical_immediate_losses += 1

            last_move = move
            moves.append(move)
            if record_search:
                label = label_result if label_result is not None else result
                search_records.append({
                    "ply": ply, "move": index_to_coord(move),
                    "to_play": to_play, "legal_moves": len(legal_before_move),
                    "action_simulations": result.visits,
                    "action_reason": getattr(result, "reason", "unknown"),
                    "label_simulations": label.visits,
                    "label_reason": getattr(label, "reason", "unknown"),
                    "label_value": getattr(label, "value", None),
                    "label_seconds": label.elapsed,
                    "label_reused_tree": getattr(label, "reused_tree", False),
                })
            if ply < len(opening_move_counts):
                coordinate = index_to_coord(move)
                opening_move_counts[ply][coordinate] = opening_move_counts[ply].get(coordinate, 0) + 1
            trajectory.append((board, to_play, list(label_policy)))
            board = play_move(board, to_play, move)
            to_play = -to_play

        # A ply cap is a safety terminal condition for data generation. Keep
        # its result consistent with the normal no-legal-move terminal rule.
        else:
            winner = -to_play

        terminal = not legal_moves(board, to_play)
        if require_terminal and not terminal:
            raise RuntimeError("self-play ended before a true terminal state; no outcome labels were accepted")
        if record_search:
            game_records.append({
                "game": game, "winner": winner, "terminal": terminal,
                "opening_moves": [index_to_coord(move) for move in initial_moves],
                "moves": [index_to_coord(move) for move in moves],
                "search_records": search_records,
            })

        total_plies += len(trajectory)
        total_label_policy_entropy += label_policy_entropy
        total_label_search_seconds += label_search_seconds
        total_label_completed_simulations += label_completed_simulations
        total_label_inference_calls += label_searcher.inference_calls if label_searcher is not None else 0
        total_label_inference_positions += label_searcher.inference_positions if label_searcher is not None else 0
        total_label_inference_seconds += label_searcher.inference_seconds if label_searcher is not None else 0.0
        total_search_seconds += search_seconds + label_search_seconds
        total_completed_simulations += completed_simulations + label_completed_simulations
        total_inference_calls += searcher.inference_calls + (label_searcher.inference_calls if label_searcher is not None else 0)
        total_inference_positions += searcher.inference_positions + (label_searcher.inference_positions if label_searcher is not None else 0)
        total_inference_seconds += searcher.inference_seconds + (label_searcher.inference_seconds if label_searcher is not None else 0.0)

        if winner == BLACK:
            black_wins += 1
        else:
            white_wins += 1

        for state, state_player, policy in trajectory:
            value = 1.0 if state_player == winner else -1.0
            samples.append(TrainingSample(state, state_player, policy, value))

        if progress:
            progress(
                {
                    "event": "self_play_game",
                    "game": game,
                    "games": games,
                    "winner": color_name(winner),
                    "plies": len(trajectory),
                    "samples": len(samples),
                    "black_wins": black_wins,
                    "white_wins": white_wins,
                    "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                    "avg_policy_entropy": game_policy_entropy / max(1, len(confidences)),
                    "avg_label_confidence": (
                        sum(label_confidences) / len(label_confidences)
                        if label_confidences else 0.0
                    ),
                    "avg_label_policy_entropy": label_policy_entropy / max(1, len(confidences)),
                    "clean_label_search": clean_label_search,
                    "tactical_decisions": game_tactical_decisions,
                    "tactical_immediate_losses": game_tactical_immediate_losses,
                    "tactical_overrides": game_tactical_overrides,
                    "tactical_immediate_loss_rate": game_tactical_immediate_losses / max(1, game_tactical_decisions),
                    "search_seconds": search_seconds + label_search_seconds,
                    "action_search_seconds": search_seconds,
                    "label_search_seconds": label_search_seconds,
                    "completed_simulations": completed_simulations + label_completed_simulations,
                    "action_completed_simulations": completed_simulations,
                    "label_completed_simulations": label_completed_simulations,
                    "elapsed": time.perf_counter() - game_started_at,
                    "last_move": index_to_coord(last_move) if last_move is not None else "none",
                    "inference_calls": searcher.inference_calls + (label_searcher.inference_calls if label_searcher is not None else 0),
                    "inference_positions": searcher.inference_positions + (label_searcher.inference_positions if label_searcher is not None else 0),
                    "inference_seconds": searcher.inference_seconds + (label_searcher.inference_seconds if label_searcher is not None else 0.0),
                    "opening_unique": [len(counts) for counts in opening_move_counts],
                }
            )

    if progress:
        elapsed = time.perf_counter() - started_at
        progress(
            {
                "event": "self_play_end",
                "games": games,
                "samples": len(samples),
                "black_wins": black_wins,
                "white_wins": white_wins,
                "plies": total_plies,
                "search_seconds": total_search_seconds,
                "completed_simulations": total_completed_simulations,
                "action_completed_simulations": (
                    total_completed_simulations - total_label_completed_simulations
                ),
                "inference_calls": total_inference_calls,
                "inference_positions": total_inference_positions,
                "inference_seconds": total_inference_seconds,
                "avg_policy_entropy": total_policy_entropy / max(1, total_plies),
                "avg_label_policy_entropy": total_label_policy_entropy / max(1, total_plies),
                "clean_label_search": clean_label_search,
                "label_simulations": effective_label_simulations if clean_label_search else 0,
                "label_time_limit": effective_label_time_limit if clean_label_search else 0.0,
                "label_search_seconds": total_label_search_seconds,
                "label_completed_simulations": total_label_completed_simulations,
                "label_inference_calls": total_label_inference_calls,
                "label_inference_positions": total_label_inference_positions,
                "label_inference_seconds": total_label_inference_seconds,
                "opening_move_counts": opening_move_counts,
                "opening_unique": [len(counts) for counts in opening_move_counts],
                "tactical_decisions": tactical_decisions,
                "tactical_immediate_losses": tactical_immediate_losses,
                "tactical_overrides": tactical_overrides,
                "tactical_immediate_loss_rate": tactical_immediate_losses / max(1, tactical_decisions),
                "elapsed": elapsed,
            }
        )
    else:
        elapsed = time.perf_counter() - started_at

    return samples, {
        "black": black_wins,
        "white": white_wins,
        "games": games,
        "plies": total_plies,
        "search_seconds": total_search_seconds,
        "completed_simulations": total_completed_simulations,
        "inference_calls": total_inference_calls,
        "inference_positions": total_inference_positions,
        "inference_seconds": total_inference_seconds,
        "policy_entropy": total_policy_entropy,
        "label_policy_entropy": total_label_policy_entropy,
        "clean_label_search": clean_label_search,
        "label_simulations": effective_label_simulations if clean_label_search else 0,
        "label_time_limit": effective_label_time_limit if clean_label_search else 0.0,
        "label_search_seconds": total_label_search_seconds,
        "label_completed_simulations": total_label_completed_simulations,
        "label_inference_calls": total_label_inference_calls,
        "label_inference_positions": total_label_inference_positions,
        "label_inference_seconds": total_label_inference_seconds,
        "opening_move_counts": opening_move_counts,
        "opening_unique": [len(counts) for counts in opening_move_counts],
        "tactical_decisions": tactical_decisions,
        "tactical_immediate_losses": tactical_immediate_losses,
        "tactical_overrides": tactical_overrides,
        "tactical_immediate_loss_rate": tactical_immediate_losses / max(1, tactical_decisions),
        "elapsed": elapsed,
        "fixed_simulations": fixed_simulations,
        "label_fixed_simulations": label_fixed_simulations,
        "game_records": game_records,
    }


def train_torch(
    samples: list[TrainingSample],
    output: str | Path,
    validation_samples: list[TrainingSample] | None = None,
    checkpoint: str | Path | None = None,
    resume_training: str | Path | None = None,
    epochs: int = 4,
    batch_size: int = 64,
    learning_rate: float = 1.0e-3,
    device: str = "auto",
    channels: int = 64,
    blocks: int = 4,
    input_planes: int = 4,
    policy_channels: int = 2,
    value_channels: int = 1,
    value_hidden: int | None = None,
    norm_type: str = "batch",
    policy_head: str = "legacy",
    minimum_learning_rate: float = 0.0,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    winner_policy_weight: float = 1.0,
    loser_policy_weight: float = 1.0,
    balance_policy_colors: bool = False,
    random_symmetries: bool = False,
    checkpoint_every: int = 0,
    seed: int | None = None,
    amp: bool = True,
    data_transfer: str = "auto",
    distill_checkpoint: str | Path | None = None,
    distill_policy_weight: float = 0.0,
    distill_value_weight: float = 0.0,
    freeze_batchnorm_stats: bool = False,
    data_fingerprint: str | None = None,
    validation_fingerprint: str | None = None,
    progress: ProgressCallback | None = None,
    micro_batch_size: int = 0,
) -> dict[str, float]:
    if not samples:
        raise ValueError("no training samples")
    if minimum_learning_rate < 0.0 or minimum_learning_rate > learning_rate:
        raise ValueError("minimum_learning_rate must be between zero and learning_rate")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if type(micro_batch_size) is not int or micro_batch_size < 0:
        raise ValueError("micro_batch_size must be a nonnegative integer")
    if min(policy_weight, value_weight, winner_policy_weight, loser_policy_weight) <= 0.0:
        raise ValueError("loss weights must be positive")
    if min(distill_policy_weight, distill_value_weight) < 0.0:
        raise ValueError("distillation weights must be non-negative")
    if checkpoint and resume_training:
        raise ValueError("checkpoint and resume_training are mutually exclusive")
    if data_transfer not in {"auto", "pinned", "gpu"}:
        raise ValueError("data_transfer must be auto, pinned, or gpu")
    validation_samples = validation_samples or []

    from .torch_model import BOARD_SIZE, POINTS, TorchPolicyValueNet, encode_board_planes, resolve_device, _import_torch

    torch, _ = _import_torch()
    if seed is not None:
        torch.manual_seed(seed)
    device = resolve_device(device)
    model_checkpoint = resume_training or checkpoint
    net = TorchPolicyValueNet(
        checkpoint=model_checkpoint,
        device=device,
        channels=channels,
        blocks=blocks,
        input_planes=input_planes,
        policy_channels=policy_channels,
        value_channels=value_channels,
        value_hidden=value_hidden,
        norm_type=norm_type,
        policy_head=policy_head,
    )
    # A fine-tuning run must never silently fall back to random initialization.
    # Keep this invariant close to model construction so the failure is
    # immediate, rather than discovered later through a weak Arena result.
    checkpoint_loaded = bool(model_checkpoint and net.checkpoint_data is not None)
    if model_checkpoint and not checkpoint_loaded:
        raise RuntimeError(
            f"training checkpoint was requested but not loaded: {model_checkpoint}"
        )
    torch = net.torch
    if micro_batch_size and micro_batch_size < batch_size and net.norm_type == "batch" and not freeze_batchnorm_stats:
        raise ValueError("micro batches require frozen BatchNorm statistics")

    def set_batchnorm_stats_frozen() -> None:
        """Keep running statistics fixed while leaving BN affine weights trainable."""
        if not freeze_batchnorm_stats:
            return
        for module in net.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    teacher_model = None
    if distill_policy_weight or distill_value_weight:
        if distill_checkpoint:
            teacher_net = TorchPolicyValueNet(checkpoint=distill_checkpoint, device=device)
            architecture = (
                teacher_net.channels,
                teacher_net.blocks,
                teacher_net.input_planes,
                teacher_net.policy_channels,
                teacher_net.value_channels,
                teacher_net.value_hidden,
                teacher_net.norm_type,
                teacher_net.policy_head,
                teacher_net.phase_plane,
            )
            candidate_architecture = (
                net.channels,
                net.blocks,
                net.input_planes,
                net.policy_channels,
                net.value_channels,
                net.value_hidden,
                net.norm_type,
                net.policy_head,
                net.phase_plane,
            )
            if architecture != candidate_architecture:
                raise ValueError("distillation checkpoint architecture does not match the candidate")
            teacher_model = teacher_net.model
        else:
            teacher_model = copy.deepcopy(net.model)
            if resume_training:
                saved_teacher = (net.checkpoint_data.get("training_state") or {}).get("teacher_model")
                if saved_teacher is None:
                    raise ValueError("resume checkpoint lacks the original implicit distillation teacher")
                teacher_model.load_state_dict(saved_teacher)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

    def cpu_model_state(model: Any) -> dict[str, Any]:
        return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    teacher_state = cpu_model_state(teacher_model) if teacher_model is not None else None
    teacher_sha256 = None
    if teacher_state is not None:
        digest = hashlib.sha256()
        for name, tensor in sorted(teacher_state.items()):
            digest.update(name.encode("utf-8"))
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        teacher_sha256 = digest.hexdigest()

    # Loading an explicit teacher constructs a module before loading weights.
    # Seed afterwards so adding distillation does not change data order or
    # random symmetries in a controlled comparison.
    if seed is not None:
        torch.manual_seed(seed)

    use_amp = bool(amp and str(device).startswith("cuda"))
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    optimizer = torch.optim.AdamW(net.model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=minimum_learning_rate,
    )
    last_loss = policy_loss = value_loss = 0.0
    avg_loss = avg_policy_loss = avg_value_loss = 0.0
    validation_loss = validation_policy_loss = validation_value_loss = 0.0
    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    baseline_validation_loss = None
    start_epoch = 1
    resume_state = None
    if resume_training:
        checkpoint_data = net.checkpoint_data
        resume_state = (
            checkpoint_data.get("training_state")
            if isinstance(checkpoint_data, dict)
            else None
        )
        if not isinstance(resume_state, dict):
            raise ValueError("resume checkpoint does not contain training_state")
        saved_config = resume_state.get("config", {})
        if not isinstance(saved_config, dict):
            raise ValueError("resume checkpoint has an invalid training config")
        expected = {
            "samples": len(samples),
            "validation_samples": len(validation_samples),
            "batch_size": batch_size,
            "target_epochs": epochs,
            "learning_rate": learning_rate,
            "minimum_learning_rate": minimum_learning_rate,
            "policy_weight": policy_weight,
            "value_weight": value_weight,
            "winner_policy_weight": winner_policy_weight,
            "loser_policy_weight": loser_policy_weight,
            "balance_policy_colors": balance_policy_colors,
            "random_symmetries": random_symmetries,
            "amp": use_amp,
            "data_transfer_requested": data_transfer,
            "channels": net.channels,
            "blocks": net.blocks,
            "input_planes": net.input_planes,
            "policy_channels": net.policy_channels,
            "value_channels": net.value_channels,
            "value_hidden": net.value_hidden,
            "norm_type": net.norm_type,
            "policy_head": net.policy_head,
            "phase_plane": net.phase_plane,
            "freeze_batchnorm_stats": freeze_batchnorm_stats,
            "distill_policy_weight": distill_policy_weight,
            "distill_value_weight": distill_value_weight,
            "teacher_sha256": teacher_sha256,
            "micro_batch_size": micro_batch_size,
        }
        for key, value in expected.items():
            # Checkpoints written before these options existed are equivalent
            # to both options being disabled.
            saved = (
                saved_config.get(key, False)
                if key in {"freeze_batchnorm_stats", "balance_policy_colors"}
                else saved_config.get(key)
            )
            if key in {"distill_policy_weight", "distill_value_weight"}:
                saved = saved_config.get(key, 0.0)
            if key == "micro_batch_size":
                saved = saved_config.get(key, 0)
            if isinstance(value, bool):
                matches = bool(saved) == value
            elif isinstance(value, float):
                matches = saved is not None and math.isclose(float(saved), value, rel_tol=0.0, abs_tol=1.0e-15)
            else:
                matches = saved == value
            if not matches:
                raise ValueError(
                    f"resume checkpoint {key} mismatch: "
                    f"saved={saved!r} current={value!r}"
                )
        for key, current in (
            ("data_fingerprint", data_fingerprint),
            ("validation_fingerprint", validation_fingerprint),
        ):
            saved = saved_config.get(key)
            if current is not None and str(saved) != str(current):
                raise ValueError(
                    f"resume checkpoint {key} mismatch: "
                    f"saved={saved!r} current={current!r}"
                )
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        if resume_state.get("scaler"):
            scaler.load_state_dict(resume_state["scaler"])
        start_epoch = int(resume_state["completed_epoch"]) + 1
        best_validation_loss = float(resume_state.get("best_validation_loss", float("inf")))
        best_epoch = int(resume_state.get("best_epoch", 0))
        if validation_samples:
            if resume_state.get("format_version", 1) < 2:
                raise ValueError("legacy validation resume uses a different loss reduction; restart from checkpoint")
            best_model_state = resume_state.get("best_model")
            if best_model_state is None:
                raise ValueError("resume checkpoint lacks best validation model weights")
            baseline_validation_loss = resume_state.get("baseline_validation_loss")
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        if str(device).startswith("cuda") and resume_state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_state["cuda_rng_state_all"]]
            )
        if start_epoch > epochs:
            raise ValueError(
                f"resume checkpoint already completed epoch {start_epoch - 1} of {epochs}"
            )
    batches_per_epoch = (len(samples) + batch_size - 1) // batch_size
    started_at = time.perf_counter()
    policy_color_scale = {BLACK: 1.0, -BLACK: 1.0}
    policy_totals = {BLACK: 0.0, -BLACK: 0.0}
    if balance_policy_colors:
        policy_color_scale, policy_totals = _policy_color_balance_scales(
            samples,
            winner_policy_weight,
            loser_policy_weight,
        )
    effective_policy_weight = {BLACK: 0.0, -BLACK: 0.0}
    for sample in samples:
        outcome_weight = winner_policy_weight if sample.value > 0.0 else loser_policy_weight
        effective_policy_weight[sample.to_play] += sample.sample_weight * sample.policy_weight * outcome_weight
    balanced_policy_weight = {
        color: effective_policy_weight[color] * policy_color_scale[color]
        for color in (BLACK, -BLACK)
    }

    if progress:
        progress(
            {
                "event": "train_start",
                "backend": "torch",
                "samples": len(samples),
                "epochs": epochs,
                "batch_size": batch_size,
                "micro_batch_size": micro_batch_size,
                "batches_per_epoch": batches_per_epoch,
                "learning_rate": learning_rate,
                "minimum_learning_rate": minimum_learning_rate,
                "device": device,
                "channels": net.channels,
                "blocks": net.blocks,
                "input_planes": net.input_planes,
                "policy_channels": net.policy_channels,
                "value_channels": net.value_channels,
                "value_hidden": net.value_hidden,
                "norm_type": net.norm_type,
                "policy_head": net.policy_head,
                "phase_plane": net.phase_plane,
                "freeze_batchnorm_stats": freeze_batchnorm_stats,
                "policy_weight": policy_weight,
                "value_weight": value_weight,
                "winner_policy_weight": winner_policy_weight,
                "loser_policy_weight": loser_policy_weight,
                "balance_policy_colors": balance_policy_colors,
                "black_policy_weight": effective_policy_weight[BLACK],
                "white_policy_weight": effective_policy_weight[-BLACK],
                "policy_color_scale_black": policy_color_scale[BLACK],
                "policy_color_scale_white": policy_color_scale[-BLACK],
                "balanced_black_policy_weight": balanced_policy_weight[BLACK],
                "balanced_white_policy_weight": balanced_policy_weight[-BLACK],
                "random_symmetries": random_symmetries,
                "checkpoint": str(checkpoint) if checkpoint else "none",
                "resume_training": str(resume_training) if resume_training else "none",
                "checkpoint_loaded": checkpoint_loaded,
                "start_epoch": start_epoch,
                "validation_samples": len(validation_samples),
                "checkpoint_every": checkpoint_every,
                "amp": use_amp,
                "data_transfer": data_transfer,
                "distill_checkpoint": str(distill_checkpoint) if distill_checkpoint else "current checkpoint",
                "distill_policy_weight": distill_policy_weight,
                "distill_value_weight": distill_value_weight,
                "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "none",
            }
        )

    def prepare_tensors(
        source: list[TrainingSample],
        phase: str,
    ) -> tuple[Any, Any, Any, Any, Any]:
        prepare_started_at = time.perf_counter()
        count = len(source)
        pin_memory = bool(str(device).startswith("cuda"))
        inputs = torch.empty(
            (count, net.input_planes, BOARD_SIZE, BOARD_SIZE),
            # The optional phase plane is fractional.  uint8 would silently
            # turn every value in (0, 1) into zero during preprocessing.
            dtype=torch.float32,
            pin_memory=pin_memory,
        )
        policies = torch.empty((count, POINTS), dtype=torch.float32, pin_memory=pin_memory)
        values = torch.empty(count, dtype=torch.float32, pin_memory=pin_memory)
        weights = torch.empty((count, 2), dtype=torch.float32, pin_memory=pin_memory)
        sides = torch.empty(count, dtype=torch.int8, pin_memory=pin_memory)
        prepare_chunk = 4096
        report_interval = max(prepare_chunk, ((count + 9) // 10 // prepare_chunk) * prepare_chunk)
        for start in range(0, count, prepare_chunk):
            end = min(count, start + prepare_chunk)
            chunk = source[start:end]
            inputs[start:end] = torch.tensor(
                [
                    encode_board_planes(
                        sample.board,
                        sample.to_play,
                        net.input_planes,
                        include_phase=net.phase_plane,
                    )
                    for sample in chunk
                ],
                dtype=torch.float32,
            )
            policies[start:end] = torch.tensor([sample.policy for sample in chunk], dtype=torch.float32)
            values[start:end] = torch.tensor([sample.value for sample in chunk], dtype=torch.float32)
            weights[start:end] = torch.tensor(
                [[sample.sample_weight * sample.policy_weight,
                  sample.sample_weight * sample.value_weight] for sample in chunk], dtype=torch.float32
            )
            sides[start:end] = torch.tensor(
                [sample.to_play for sample in chunk], dtype=torch.int8
            )
            if progress and (end == count or end % report_interval < prepare_chunk):
                progress(
                    {
                        "event": "train_prepare",
                        "phase": phase,
                        "samples_prepared": end,
                        "samples_total": count,
                        "elapsed": time.perf_counter() - prepare_started_at,
                    }
                )
        return inputs, policies, values, weights, sides

    sample_count = len(samples)
    input_store, policy_store, value_store, sample_weight_store, side_store = prepare_tensors(samples, "train")
    validation_input_store, validation_policy_store, validation_value_store, validation_weight_store, validation_side_store = prepare_tensors(
        validation_samples,
        "validation",
    )
    samples.clear()
    validation_samples.clear()

    tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            input_store,
            policy_store,
            value_store,
            sample_weight_store,
            side_store,
            validation_input_store,
            validation_policy_store,
            validation_value_store,
            validation_weight_store,
            validation_side_store,
        )
    )
    resolved_transfer = "cpu"
    if str(device).startswith("cuda"):
        resolved_transfer = data_transfer
        if data_transfer == "auto":
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
            preload_limit = min(2 * 1024**3, int(free_bytes * 0.20))
            resolved_transfer = "gpu" if tensor_bytes <= preload_limit else "pinned"
        if resolved_transfer == "gpu":
            input_store = input_store.to(device=device, non_blocking=True)
            policy_store = policy_store.to(device=device, non_blocking=True)
            value_store = value_store.to(device=device, non_blocking=True)
            sample_weight_store = sample_weight_store.to(device=device, non_blocking=True)
            side_store = side_store.to(device=device, non_blocking=True)
            validation_input_store = validation_input_store.to(device=device, non_blocking=True)
            validation_policy_store = validation_policy_store.to(device=device, non_blocking=True)
            validation_value_store = validation_value_store.to(device=device, non_blocking=True)
            validation_weight_store = validation_weight_store.to(device=device, non_blocking=True)
            validation_side_store = validation_side_store.to(device=device, non_blocking=True)
            torch.cuda.synchronize()
        else:
            resolved_transfer = "pinned"
    storage_device = device if resolved_transfer == "gpu" else "cpu"
    if progress:
        progress(
            {
                "event": "train_transfer",
                "mode": resolved_transfer,
                "bytes": tensor_bytes,
                "device": device,
            }
        )

    symmetry_gather = None
    if random_symmetries:
        gather_rows: list[list[int]] = []
        for transform in range(8):
            target_to_source = [0] * POINTS
            for source in range(POINTS):
                target_to_source[transform_index(source, transform)] = source
            gather_rows.append(target_to_source)
        symmetry_gather = torch.tensor(gather_rows, dtype=torch.long, device=device)

    def apply_random_symmetry(x: Any, policy: Any) -> tuple[Any, Any]:
        if symmetry_gather is None or len(x) == 0:
            return x, policy
        choices = torch.randint(0, 8, (len(x),), device=device)
        flat_x = x.flatten(start_dim=2)
        transformed_x = torch.empty_like(flat_x)
        transformed_policy = torch.empty_like(policy)
        for transform in range(8):
            selected = torch.nonzero(choices == transform, as_tuple=False).flatten()
            if len(selected) == 0:
                continue
            gather = symmetry_gather[transform]
            transformed_x[selected] = flat_x[selected].index_select(2, gather)
            transformed_policy[selected] = policy[selected].index_select(1, gather)
        return transformed_x.view_as(x), transformed_policy

    def evaluate_validation() -> tuple[float, float, float]:
        # Accumulate each weighted objective globally. Batch means weighted by
        # sample count change the result when sample weights/boundaries differ.
        net.model.eval()
        policy_numerator = value_numerator = 0.0
        policy_denominator = value_denominator = 0.0
        with torch.no_grad():
            eval_batch_size = min(batch_size, micro_batch_size) if micro_batch_size else batch_size
            for start in range(0, len(validation_input_store), eval_batch_size):
                x = validation_input_store[start : start + eval_batch_size].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=resolved_transfer == "pinned",
                )
                target_policy = validation_policy_store[start : start + eval_batch_size].to(
                    device=device, non_blocking=resolved_transfer == "pinned"
                )
                target_value = validation_value_store[start : start + eval_batch_size].to(
                    device=device, non_blocking=resolved_transfer == "pinned"
                )
                target_sample_weight = validation_weight_store[start : start + eval_batch_size].to(
                    device=device, non_blocking=resolved_transfer == "pinned"
                )
                target_policy_weight = target_sample_weight[:, 0]
                target_sample_weight = target_sample_weight[:, 1]
                target_to_play = validation_side_store[start : start + eval_batch_size].to(
                    device=device, non_blocking=resolved_transfer == "pinned"
                )
                logits, pred_value = net.model(x)
                legal_mask = x[:, 3].reshape(len(x), -1).to(dtype=torch.bool)
                policy_per_sample = -(
                    target_policy
                    * torch.log_softmax(logits.masked_fill(~legal_mask, -1.0e4), dim=1)
                ).sum(dim=1)
                policy_sample_weight = torch.where(
                    target_value > 0.0,
                    torch.full_like(target_value, winner_policy_weight),
                    torch.full_like(target_value, loser_policy_weight),
                ) * target_policy_weight
                if balance_policy_colors:
                    policy_sample_weight = policy_sample_weight * torch.where(
                        target_to_play == BLACK,
                        torch.full_like(target_sample_weight, policy_color_scale[BLACK]),
                        torch.full_like(target_sample_weight, policy_color_scale[-BLACK]),
                    )
                value_per_sample = (pred_value - target_value).square()
                policy_numerator += (policy_per_sample.double() * policy_sample_weight.double()).sum().item()
                policy_denominator += policy_sample_weight.double().sum().item()
                value_numerator += (value_per_sample.double() * target_sample_weight.double()).sum().item()
                value_denominator += target_sample_weight.double().sum().item()
        policy_mean = policy_numerator / policy_denominator if policy_denominator else 0.0
        value_mean = value_numerator / value_denominator if value_denominator else 0.0
        return policy_weight * policy_mean + value_weight * value_mean, policy_mean, value_mean

    if len(validation_input_store):
        if resume_state is None:
            baseline_validation_loss, _, _ = evaluate_validation()
            best_validation_loss = baseline_validation_loss
            best_model_state = cpu_model_state(net.model)
        # Ensure a new output path contains the selected model even if every
        # subsequent epoch regresses. Epoch snapshots remain resumable.
        current_state = cpu_model_state(net.model)
        net.model.load_state_dict(best_model_state)
        net.save(output)
        net.model.load_state_dict(current_state)
        if progress:
            progress({"event": "train_baseline", "validation_loss": baseline_validation_loss,
                      "best_validation_loss": best_validation_loss, "best_epoch": best_epoch})

    net.model.train()
    set_batchnorm_stats_frozen()
    recent_epoch_durations: deque[float] = deque(maxlen=8)
    for epoch in range(start_epoch, epochs + 1):
        epoch_started_at = time.perf_counter()
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        order = torch.randperm(sample_count, device=storage_device)
        loss_total = policy_total = value_total = 0.0
        seen = 0
        batch_count = 0
        recent_batch_durations: deque[float] = deque(maxlen=32)
        batch_report_interval = max(1, batches_per_epoch // 8)
        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            batch_started_at = time.perf_counter()
            non_blocking = resolved_transfer == "pinned"
            x = input_store[indices].to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
            target_policy = policy_store[indices].to(device=device, non_blocking=non_blocking)
            target_value = value_store[indices].to(device=device, non_blocking=non_blocking)
            target_sample_weight = sample_weight_store[indices].to(
                device=device,
                non_blocking=non_blocking,
            )
            target_policy_weight = target_sample_weight[:, 0]
            target_sample_weight = target_sample_weight[:, 1]
            target_to_play = side_store[indices].to(
                device=device,
                non_blocking=non_blocking,
            )
            x, target_policy = apply_random_symmetry(x, target_policy)

            optimizer.zero_grad(set_to_none=True)
            if micro_batch_size:
                from .training_microbatch import backward_microbatches

                policy_sample_weight = torch.where(
                    target_value > 0.0,
                    torch.full_like(target_value, winner_policy_weight),
                    torch.full_like(target_value, loser_policy_weight),
                ) * target_policy_weight
                if balance_policy_colors:
                    policy_sample_weight = policy_sample_weight * torch.where(
                        target_to_play == BLACK,
                        torch.full_like(target_sample_weight, policy_color_scale[BLACK]),
                        torch.full_like(target_sample_weight, policy_color_scale[-BLACK]),
                    )
                loss, policy_term, value_term = backward_microbatches(
                    torch, net.model, teacher_model, scaler, x, target_policy, target_value,
                    policy_sample_weight, target_sample_weight,
                    micro_batch_size=micro_batch_size,
                    policy_weight=policy_weight, value_weight=value_weight,
                    distill_policy_weight=distill_policy_weight,
                    distill_value_weight=distill_value_weight, use_amp=use_amp,
                )
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits, pred_value = net.model(x)
                    # Match inference: policy probability mass must only cover legal moves.
                    legal_mask = x[:, 3].reshape(len(x), -1).to(dtype=torch.bool)
                    log_probs = torch.log_softmax(logits.masked_fill(~legal_mask, -1.0e4), dim=1)
                    policy_per_sample = -(target_policy * log_probs).sum(dim=1)
                    policy_sample_weight = torch.where(
                        target_value > 0.0,
                        torch.full_like(target_value, winner_policy_weight),
                        torch.full_like(target_value, loser_policy_weight),
                    ) * target_policy_weight
                    if balance_policy_colors:
                        policy_sample_weight = policy_sample_weight * torch.where(
                            target_to_play == BLACK,
                            torch.full_like(target_sample_weight, policy_color_scale[BLACK]),
                            torch.full_like(target_sample_weight, policy_color_scale[-BLACK]),
                        )
                    policy_term = (
                        (policy_per_sample * policy_sample_weight).sum() / policy_sample_weight.sum()
                        if bool(policy_sample_weight.sum() > 0)
                        else torch.zeros((), device=device)
                    )
                    value_per_sample = torch.nn.functional.mse_loss(
                        pred_value, target_value, reduction="none"
                    )
                    value_term = (
                        (value_per_sample * target_sample_weight).sum() / target_sample_weight.sum()
                        if bool(target_sample_weight.sum() > 0)
                        else torch.zeros((), device=device)
                    )
                    distill_policy_term = torch.zeros((), device=device)
                    distill_value_term = torch.zeros((), device=device)
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_logits, teacher_value = teacher_model(x)
                            teacher_log_probs = torch.log_softmax(
                                teacher_logits.masked_fill(~legal_mask, -1.0e4), dim=1
                            )
                            teacher_probs = teacher_log_probs.exp()
                        if distill_policy_weight > 0:
                            distill_policy_term = torch.nn.functional.kl_div(
                                log_probs, teacher_probs, reduction="none"
                            ).sum(dim=1).mean()
                        if distill_value_weight > 0:
                            distill_value_term = torch.nn.functional.mse_loss(
                                pred_value, teacher_value.detach()
                            )
                    loss = (
                        policy_weight * policy_term
                        + value_weight * value_term
                        + distill_policy_weight * distill_policy_term
                        + distill_value_weight * distill_value_term
                    )

                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            last_loss = float(loss.detach().cpu().item())
            policy_loss = float(policy_term.detach().cpu().item())
            value_loss = float(value_term.detach().cpu().item())
            batch_weight = len(indices)
            loss_total += last_loss * batch_weight
            policy_total += policy_loss * batch_weight
            value_total += value_loss * batch_weight
            seen += batch_weight
            batch_count += 1
            batch_elapsed = time.perf_counter() - batch_started_at
            recent_batch_durations.append(batch_elapsed)

            if progress and (batch_count == 1 or batch_count % batch_report_interval == 0 or batch_count == batches_per_epoch):
                epoch_elapsed = time.perf_counter() - epoch_started_at
                samples_per_sec = seen / epoch_elapsed if epoch_elapsed > 0 else 0.0
                remaining_batches = batches_per_epoch - batch_count
                eta = (
                    _estimate_remaining_seconds(recent_batch_durations, remaining_batches)
                    if len(recent_batch_durations) >= 4
                    else None
                )
                progress(
                    {
                        "event": "train_batch",
                        "backend": "torch",
                        "epoch": epoch,
                        "epochs": epochs,
                        "batch": batch_count,
                        "batches": batches_per_epoch,
                        "loss": last_loss,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                        "samples_seen": seen,
                        "samples_total": sample_count,
                        "samples_per_sec": samples_per_sec,
                        "eta": eta,
                        "batch_elapsed": batch_elapsed,
                    }
                )

        if seen:
            avg_loss = loss_total / seen
            avg_policy_loss = policy_total / seen
            avg_value_loss = value_total / seen
        save_best = False
        if len(validation_input_store):
            validation_loss, validation_policy_loss, validation_value_loss = evaluate_validation()
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_model_state = cpu_model_state(net.model)
                save_best = True
            net.model.train()
            set_batchnorm_stats_frozen()

        scheduler.step()
        training_state = {
            "format_version": 2,
            "completed_epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else []
            ),
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "best_model": best_model_state,
            "baseline_validation_loss": baseline_validation_loss,
            "teacher_model": teacher_state,
            "config": {
                "samples": sample_count,
                "validation_samples": len(validation_input_store),
                "batch_size": batch_size,
                "micro_batch_size": micro_batch_size,
                "target_epochs": epochs,
                "learning_rate": learning_rate,
                "minimum_learning_rate": minimum_learning_rate,
                "policy_weight": policy_weight,
                "value_weight": value_weight,
                "winner_policy_weight": winner_policy_weight,
                "loser_policy_weight": loser_policy_weight,
                "balance_policy_colors": balance_policy_colors,
                "random_symmetries": random_symmetries,
                "amp": use_amp,
                "data_transfer_requested": data_transfer,
                "data_transfer": resolved_transfer,
                "channels": net.channels,
                "blocks": net.blocks,
                "input_planes": net.input_planes,
                "policy_channels": net.policy_channels,
                "value_channels": net.value_channels,
                "value_hidden": net.value_hidden,
                "norm_type": net.norm_type,
                "policy_head": net.policy_head,
                "phase_plane": net.phase_plane,
                "freeze_batchnorm_stats": freeze_batchnorm_stats,
                "distill_policy_weight": distill_policy_weight,
                "distill_value_weight": distill_value_weight,
                "teacher_sha256": teacher_sha256,
                "data_fingerprint": data_fingerprint,
                "validation_fingerprint": validation_fingerprint,
            },
        }
        if save_best:
            net.save(output, training_state=training_state)
        if checkpoint_every and (epoch % checkpoint_every == 0 or epoch == epochs):
            output_path = Path(output)
            epoch_path = output_path.with_name(
                f"{output_path.stem}.epoch_{epoch:03d}{output_path.suffix}"
            )
            net.save(epoch_path, training_state=training_state)
        if not len(validation_input_store) and epoch == epochs:
            net.save(output, training_state=training_state)
        # Include validation and checkpoint I/O in the epoch estimate because
        # both are part of the time the caller must wait for training.
        epoch_elapsed = time.perf_counter() - epoch_started_at
        recent_epoch_durations.append(epoch_elapsed)
        total_remaining_epochs = epochs - epoch
        total_eta = _estimate_remaining_seconds(recent_epoch_durations, total_remaining_epochs)
        samples_per_sec = seen / epoch_elapsed if epoch_elapsed > 0 else 0.0
        if progress:
            gpu_memory = {}
            if str(device).startswith("cuda"):
                gpu_utilization = None
                try:
                    gpu_utilization = int(torch.cuda.utilization(device))
                except (AttributeError, RuntimeError, OSError, ModuleNotFoundError):
                    pass
                gpu_memory = {
                    "gpu_allocated": torch.cuda.memory_allocated(),
                    "gpu_reserved": torch.cuda.memory_reserved(),
                    "gpu_peak": torch.cuda.max_memory_allocated(),
                }
                if gpu_utilization is not None:
                    gpu_memory["gpu_utilization"] = gpu_utilization
            progress(
                {
                    "event": "train_epoch",
                    "backend": "torch",
                    "epoch": epoch,
                    "epochs": epochs,
                    "batches": batch_count,
                    "loss": avg_loss,
                    "policy_loss": avg_policy_loss,
                    "value_loss": avg_value_loss,
                    "validation_loss": validation_loss if len(validation_input_store) else None,
                    "validation_policy_loss": validation_policy_loss if len(validation_input_store) else None,
                    "validation_value_loss": validation_value_loss if len(validation_input_store) else None,
                    "learning_rate": current_learning_rate,
                    "elapsed": epoch_elapsed,
                    "eta": total_eta,
                    "samples_per_sec": samples_per_sec,
                    "data_transfer": resolved_transfer,
                    **gpu_memory,
                }
            )

    net.model.eval()
    elapsed = time.perf_counter() - started_at
    if progress:
        progress(
            {
                "event": "train_end",
                "backend": "torch",
                "output": str(output),
                "loss": avg_loss,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "validation_loss": validation_loss if len(validation_input_store) else None,
                "best_validation_loss": best_validation_loss if len(validation_input_store) else None,
                "best_epoch": best_epoch,
                "elapsed": elapsed,
            }
        )
    return {
        "loss": last_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "avg_loss": avg_loss,
        "avg_policy_loss": avg_policy_loss,
        "avg_value_loss": avg_value_loss,
        "validation_loss": validation_loss,
        "validation_policy_loss": validation_policy_loss,
        "validation_value_loss": validation_value_loss,
        "best_validation_loss": best_validation_loss if len(validation_input_store) else 0.0,
        "baseline_validation_loss": baseline_validation_loss if baseline_validation_loss is not None else 0.0,
        "best_epoch": float(best_epoch),
        "elapsed": elapsed,
    }


def train_tiny_value_head(
    samples: list[TrainingSample],
    output: str | Path,
    epochs: int = 3,
    learning_rate: float = 0.03,
    progress: ProgressCallback | None = None,
) -> dict[str, float]:
    """Small fallback trainer for environments without PyTorch.

    It updates only the final value-head layer of TinyPolicyValueNet. The full
    AlphaZero-style policy/value trainer is `train_torch`.
    """

    from .network import _tanh, value_features

    if any(sample.policy_weight != 1.0 or sample.value_weight != 1.0 for sample in samples):
        raise ValueError("sample head weights require the torch trainer")
    net = TinyPolicyValueNet()
    last_loss = 0.0
    started_at = time.perf_counter()
    if progress:
        progress(
            {
                "event": "train_start",
                "backend": "tiny",
                "samples": len(samples),
                "epochs": epochs,
                "batch_size": 1,
                "batches_per_epoch": len(samples),
                "learning_rate": learning_rate,
                "device": "cpu",
            }
        )
    for epoch in range(1, epochs + 1):
        epoch_started_at = time.perf_counter()
        loss_total = 0.0
        for sample in samples:
            features = value_features(sample.board, sample.to_play)
            hidden = [
                _tanh(sum(w * v for w, v in zip(row, features)) + bias)
                for row, bias in zip(net.weights["value_w1"], net.weights["value_b1"])
            ]
            pred = _tanh(sum(w * h for w, h in zip(net.weights["value_w2"], hidden)) + net.weights["value_b2"])
            error = pred - sample.value
            last_loss = error * error
            grad = 2.0 * error * (1.0 - pred * pred)
            for i, h in enumerate(hidden):
                net.weights["value_w2"][i] -= learning_rate * grad * h
            net.weights["value_b2"] -= learning_rate * grad
            loss_total += last_loss
        avg_loss = loss_total / len(samples) if samples else 0.0
        epoch_elapsed = time.perf_counter() - epoch_started_at
        if progress:
            progress(
                {
                    "event": "train_epoch",
                    "backend": "tiny",
                    "epoch": epoch,
                    "epochs": epochs,
                    "batches": len(samples),
                    "loss": avg_loss,
                    "policy_loss": 0.0,
                    "value_loss": avg_loss,
                    "elapsed": epoch_elapsed,
                    "samples_per_sec": len(samples) / epoch_elapsed if epoch_elapsed > 0 else 0.0,
                }
            )
    net.save(output)
    elapsed = time.perf_counter() - started_at
    if progress:
        progress(
            {
                "event": "train_end",
                "backend": "tiny",
                "output": str(output),
                "loss": last_loss,
                "value_loss": last_loss,
                "elapsed": elapsed,
            }
        )
    return {"loss": last_loss, "elapsed": elapsed}


def describe_samples(samples: list[TrainingSample]) -> str:
    if not samples:
        return "no samples"
    positives = sum(1 for sample in samples if sample.value > 0)
    first_move = max(range(POINTS), key=lambda idx: samples[0].policy[idx])
    return (
        f"samples={len(samples)} positive={positives} negative={len(samples)-positives} "
        f"first_policy_peak={index_to_coord(first_move)}"
    )


def _choose_from_policy(
    policy: list[float],
    rng: random.Random,
    temperature: float = 1.0,
    uniform_mix: float = 0.0,
    legal: Iterable[int] | None = None,
) -> int | None:
    """Sample a legal-support policy for early self-play moves.

    Temperature changes action selection only; the original MCTS visit policy
    remains the training target.  The optional uniform mix broadens openings
    over *all* rule-legal moves, including legal moves that received zero MCTS
    visits in a small search.  Without the explicit legal list the function
    keeps its old positive-support behavior for callers outside self-play.
    """
    if not policy:
        return None
    if legal is None:
        positive_support = [
            index for index, value in enumerate(policy)
            if value > 0.0 and math.isfinite(value)
        ]
        if not positive_support:
            return None
        candidates = range(len(policy))
    else:
        candidates = sorted({
            index for index in legal
            if 0 <= index < len(policy)
        })
        if not candidates:
            return None
    if temperature <= 0.0:
        return max(candidates, key=lambda index: float(policy[index]))
    temperature = max(0.05, float(temperature))
    uniform_mix = max(0.0, min(1.0, float(uniform_mix)))
    if legal is None:
        support = positive_support
    else:
        support = sorted({
            index for index in legal
            if 0 <= index < len(policy)
        })
    if not support:
        return None
    weights = [0.0] * len(policy)
    inverse_temperature = 1.0 / temperature
    transformed_total = 0.0
    for index in support:
        raw_value = float(policy[index])
        value = max(0.0, raw_value) ** inverse_temperature if math.isfinite(raw_value) else 0.0
        weights[index] = value
        transformed_total += value
    if transformed_total <= 0.0:
        return support[rng.randrange(len(support))]
    uniform_weight = transformed_total / len(support)
    for index in support:
        weights[index] = (1.0 - uniform_mix) * weights[index] + uniform_mix * uniform_weight
    threshold = rng.random() * sum(weights)
    running = 0.0
    for idx, value in enumerate(weights):
        running += value
        if running >= threshold:
            return idx
    return support[-1]
