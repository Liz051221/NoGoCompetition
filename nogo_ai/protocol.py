from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TextIO

from .mcts import NeuralMCTS
from .rules import BLACK, Board, coord_to_index, index_to_coord, is_legal_move, new_board, parse_color, play_move


@dataclass(slots=True)
class EngineConfig:
    name: str = "PyNNNoGo"
    verbose: bool = False
    game_time_limit: float = 900.0
    time_reserve: float = 30.0


class SAUNoGoEngine:
    def __init__(self, searcher: NeuralMCTS, config: EngineConfig | None = None) -> None:
        self.searcher = searcher
        self.per_move_time_limit = searcher.time_limit
        self.config = config or EngineConfig()
        self.board: Board = new_board()
        self.side = BLACK
        self.last_board_before_own_move: Board | None = None
        self.last_own_move: int | None = None
        self.banned_once: set[int] = set()
        self.search_seconds = 0.0

    def run(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for raw_line in stdin:
            clean_line = _sanitize_line(raw_line)
            self._log(f"< raw={raw_line.strip()!r} clean={clean_line.strip()!r}")
            response = self.handle_line(clean_line)
            if response:
                self._log(f"> {response!r}")
                print(response, file=stdout, flush=True)
            if clean_line.strip().lower() == "quit":
                break

    def handle_line(self, raw_line: str) -> str | None:
        parts = _sanitize_line(raw_line).strip().split()
        if not parts:
            return None

        command = parts[0].lower()
        try:
            if command == "name?":
                return f"name {self.config.name}"
            if command == "new":
                return self._handle_new(parts)
            if command == "move":
                return self._handle_move(parts)
            if command == "error":
                return self._handle_error()
            if command == "end":
                self.banned_once.clear()
                return None
            if command == "quit":
                return None
        except Exception as exc:  # Keep protocol alive; diagnostics go to stderr.
            self._log(f"protocol error for {raw_line.strip()!r}: {exc}")
            return self._safe_move_response()
        return None

    def _handle_new(self, parts: list[str]) -> str | None:
        if len(parts) < 2:
            raise ValueError("missing side")
        self.board = new_board()
        self.side = parse_color(parts[1])
        self.last_board_before_own_move = None
        self.last_own_move = None
        self.banned_once.clear()
        self.search_seconds = 0.0
        if self.side == BLACK:
            return self._choose_and_play()
        return None

    def _handle_move(self, parts: list[str]) -> str:
        if len(parts) >= 2:
            move_text = parts[1]
            if move_text.lower() not in {"go", "pass"}:
                move = coord_to_index(move_text)
                enemy = -self.side
                if is_legal_move(self.board, enemy, move):
                    self.board = play_move(self.board, enemy, move)
                else:
                    self._log(f"ignored illegal opponent move {move_text}")
        return self._choose_and_play()

    def _handle_error(self) -> str:
        if self.last_board_before_own_move is not None:
            self.board = self.last_board_before_own_move
        if self.last_own_move is not None:
            self.banned_once.add(self.last_own_move)
        return self._choose_and_play()

    def _choose_and_play(self) -> str:
        decision_started_at = time.perf_counter()
        move_budget = _allocate_search_time(
            per_move_limit=self.per_move_time_limit,
            game_time_limit=self.config.game_time_limit,
            time_reserve=self.config.time_reserve,
            spent=self.search_seconds,
            empty_points=self.board.count(0),
        )
        self.searcher.time_limit = move_budget
        result = self.searcher.search(self.board, self.side)
        move = result.move
        used_fallback = False
        if move is None or move in self.banned_once or not is_legal_move(self.board, self.side, move):
            move = self._fallback_move()
            used_fallback = True
        if move is None:
            self.search_seconds += time.perf_counter() - decision_started_at
            self._log("no legal move available")
            return "move AA"

        self.banned_once.discard(move)
        self.last_board_before_own_move = self.board
        self.last_own_move = move
        self.board = play_move(self.board, self.side, move)
        decision_elapsed = time.perf_counter() - decision_started_at
        self.search_seconds += decision_elapsed
        coord = index_to_coord(move)
        fallback_text = " fallback=1" if used_fallback else ""
        self._log(
            f"move {coord}{fallback_text}, visits={result.visits}, value={result.value:+.3f}, "
            f"confidence={_percent(result.confidence)}, margin={_percent(result.margin)}, "
            f"reason={result.reason}, "
            f"elapsed={_format_seconds(decision_elapsed)}, search={_format_seconds(result.elapsed)}, "
            f"budget={_format_seconds(move_budget)}, game_used={_format_seconds(self.search_seconds)}, "
            f"tree_reused={int(result.reused_tree)}, "
            f"top={_format_candidates(result)}"
        )
        return f"move {coord}"

    def _fallback_move(self) -> int | None:
        for move in range(len(self.board)):
            if move in self.banned_once:
                continue
            if is_legal_move(self.board, self.side, move):
                return move
        for move in range(len(self.board)):
            if is_legal_move(self.board, self.side, move):
                return move
        return None

    def _safe_move_response(self) -> str:
        move = self._fallback_move()
        if move is None:
            return "move AA"
        self.last_board_before_own_move = self.board
        self.last_own_move = move
        self.board = play_move(self.board, self.side, move)
        return f"move {index_to_coord(move)}"

    def _log(self, message: str) -> None:
        log_path = os.environ.get("NOGO_AI_LOG")
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    fh.write(f"{stamp} {message}\n")
            except OSError:
                pass
        if self.config.verbose:
            print(f"[nogo-ai] {message}", file=sys.stderr, flush=True)


def _sanitize_line(raw_line: str) -> str:
    return raw_line.replace("\x00", "").lstrip("\ufeff")


def _allocate_search_time(
    per_move_limit: float,
    game_time_limit: float,
    time_reserve: float,
    spent: float,
    empty_points: int,
) -> float:
    per_move_limit = max(0.05, per_move_limit)
    if game_time_limit <= 0.0:
        return per_move_limit
    usable_total = max(0.0, game_time_limit - max(0.0, time_reserve))
    remaining = max(0.0, usable_total - max(0.0, spent))
    remaining_own_turns = max(1, (max(0, empty_points) + 1) // 2)
    fair_share = remaining / remaining_own_turns
    return max(0.05, min(per_move_limit, fair_share))


def _format_candidates(result: object) -> str:
    candidates = getattr(result, "top_candidates", [])
    if not candidates:
        return "-"
    parts: list[str] = []
    for candidate in candidates[:5]:
        parts.append(
            f"{index_to_coord(candidate.move)}:"
            f"{candidate.visits}/{_percent(candidate.visit_share)}/q{candidate.q:+.2f}/"
            f"p{_percent(candidate.prior)}/h{candidate.tactical:.2f}"
        )
    return " ".join(parts)


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _format_seconds(value: float) -> str:
    if value < 1.0:
        return f"{1000.0 * value:.0f}ms"
    return f"{value:.2f}s"
