"""Small macOS-friendly local referee for two SAU stdin/stdout engines."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nogo_ai.rules import BLACK, WHITE, coord_to_index, index_to_coord, is_legal_move, legal_moves, new_board, play_move


@dataclass
class EngineProcess:
    path: Path
    root: Path
    label: str
    process: subprocess.Popen[str] | None = None
    messages: queue.Queue[str | None] | None = None

    def start(self) -> None:
        self.messages = queue.Queue()
        self.process = subprocess.Popen(
            [str(self.path), "--root", str(self.root)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
            cwd=self.path.parent,
        )
        assert self.process.stdout is not None
        channel = self.messages
        def read_lines() -> None:
            for line in self.process.stdout:
                channel.put(line.rstrip("\r\n"))
            channel.put(None)
        threading.Thread(target=read_lines, daemon=True).start()

    def command(self, line: str, timeout: float) -> str:
        self.send(line)
        assert self.messages is not None
        try:
            answer = self.messages.get(timeout=max(0.01, timeout))
        except queue.Empty as error:
            raise TimeoutError(f"{self.label} exceeded {timeout:.3f}s") from error
        if answer is None:
            raise RuntimeError(f"{self.label} exited before replying")
        return answer

    def send(self, line: str) -> None:
        assert self.process is not None and self.process.stdin is not None and self.messages is not None
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin:
                self.process.stdin.write("quit\n")
                self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT / "NoGoEngine")
    parser.add_argument("--total-seconds", type=float, default=900.0)
    parser.add_argument("--reserve-seconds", type=float, default=180.0)
    parser.add_argument("--move-cap-seconds", type=float, default=60.0)
    args = parser.parse_args()
    usable = max(0.0, args.total_seconds - args.reserve_seconds)
    engines = {BLACK: EngineProcess(args.engine.resolve(), ROOT, "black"),
               WHITE: EngineProcess(args.engine.resolve(), ROOT, "white")}
    for engine in engines.values():
        engine.start()
    spent = {BLACK: 0.0, WHITE: 0.0}
    board, side, move_no = new_board(), BLACK, 0
    result: dict[str, object] = {"platform": "macos-local-sau-referee", "moves": []}
    try:
        for color, engine in engines.items():
            expected = "name NoGoChampionSafe"
            if engine.command("name?", 5.0) != expected:
                raise RuntimeError(f"{engine.label} failed name handshake")
        engines[WHITE].send("new white")
        while legal_moves(board, side):
            if move_no == 0:
                command = "new black"
            else:
                command = "move " + last_coord
            budget = max(0.05, min(args.move_cap_seconds, usable - spent[side]))
            started = time.perf_counter()
            response = engines[side].command(command, budget + 0.25)
            elapsed = time.perf_counter() - started
            spent[side] += elapsed
            if spent[side] > usable + 0.25:
                raise TimeoutError(f"{('black' if side == BLACK else 'white')} total clock exhausted")
            parts = response.split()
            if len(parts) != 2 or parts[0].lower() != "move":
                raise RuntimeError(f"invalid response from {('black' if side == BLACK else 'white')}: {response!r}")
            move = coord_to_index(parts[1])
            if not is_legal_move(board, side, move):
                raise RuntimeError(f"illegal move {parts[1]} from {('black' if side == BLACK else 'white')}")
            board = play_move(board, side, move)
            last_coord = index_to_coord(move)
            move_no += 1
            result["moves"].append({"number": move_no, "side": "black" if side == BLACK else "white",
                                    "coord": last_coord, "seconds": round(elapsed, 3)})
            print(f"{move_no:3d}. {'black' if side == BLACK else 'white'} {last_coord} ({elapsed:.2f}s)", flush=True)
            side = -side
        winner = "white" if side == BLACK else "black"
        result.update({"winner": winner, "loser": "black" if winner == "white" else "white",
                       "move_count": move_no, "spent_seconds": spent, "passed": True})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        result.update({"passed": False, "error": str(error), "spent_seconds": spent, "move_count": move_no})
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr, flush=True)
        return 2
    finally:
        for engine in engines.values():
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
