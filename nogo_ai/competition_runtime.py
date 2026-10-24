"""Deployment-only clock isolation; training/search implementations stay frozen."""
from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from types import SimpleNamespace

from .mcts import NeuralMCTS
from .protocol import SAUNoGoEngine
from .rules import legal_moves


class EvictionSafeLRU(OrderedDict):
    def move_to_end(self, key, last=True):
        # A prediction retained in a batch-local result may already have been
        # evicted while inserting other predictions from that same batch.
        if key in self:
            super().move_to_end(key, last=last)


class CompetitionMCTS(NeuralMCTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_cache = EvictionSafeLRU(self.prediction_cache)
        self.progress_callback = None
        self._progress_at = 0.0

    def _simulate_batch(self, root, batch_size, deadline):
        count = super()._simulate_batch(root, batch_size, deadline)
        now = time.perf_counter()
        if self.progress_callback and root.edges and now >= self._progress_at:
            best = max(root.edges.values(), key=lambda edge: (edge.visits, edge.q, edge.prior))
            self.progress_callback({"move": best.move, "visits": root.visits, "value": best.q})
            self._progress_at = now + 0.5
        return count


def worker_loop(searcher, stdin, stdout):
    def emit(message):
        stdout.write(json.dumps(message, allow_nan=False) + "\n")
        stdout.flush()
    emit({"kind": "ready"})
    for line in stdin:
        request = json.loads(line)
        if request.get("kind") == "quit":
            return
        identity = request["id"]
        searcher.time_limit = request["time_limit"]
        searcher.progress_callback = lambda data: emit({"kind": "progress", "id": identity, **data})
        try:
            result = searcher.search(tuple(request["board"]), request["to_play"])
            emit({"kind": "result", "id": identity,
                  **{name: getattr(result, name) for name in
                     ("move", "visits", "value", "confidence", "margin", "elapsed", "reused_tree", "reason")}})
        except Exception as error:
            emit({"kind": "error", "id": identity, "error": repr(error)})
            # An interrupted virtual-loss batch may leave the tree modified.
            return


class GuardedSearcher:
    """The protocol process never runs neural inference or tactical search."""
    def __init__(self, command, time_limit, log_directory, *, environment=None):
        self.command = list(command)
        self.time_limit = float(time_limit)
        self.remaining_game_seconds = float("inf")
        self.log_directory = Path(log_directory)
        self.environment = environment
        self.process = None
        self.messages = None
        self.identity = 0
        self.last_reason = None
        self._reapers = []
        self._ready = False
        self._job = None

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        if self.process is not None or self._job is not None:
            self._retire()
        self.log_directory.mkdir(parents=True, exist_ok=True)
        diagnostic = self.log_directory / f"worker_{os.getpid()}.log"
        self.messages = queue.Queue()
        self._ready = False
        if os.name == "nt":
            from .competition_process import SearchJob
            self._job = SearchJob()
        with diagnostic.open("a", encoding="utf-8") as errors:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                text=True, encoding="utf-8", env=self.environment,
                creationflags=(subprocess.CREATE_NO_WINDOW | 4) if os.name == "nt" else 0)
        if self._job is not None:
            self._job.attach_and_resume(self.process)
        channel, output = self.messages, self.process.stdout
        def reader():
            try:
                for line in output:
                    try:
                        channel.put(json.loads(line))
                    except (ValueError, TypeError):
                        channel.put({"kind": "invalid_output"})
            finally:
                output.close()
                channel.put({"kind": "closed"})
        threading.Thread(target=reader, daemon=True).start()

    def _retire(self):
        process, self.process = self.process, None
        job, self._job = self._job, None
        if job is not None:
            job.terminate()
        if process is None:
            if job is not None:
                job.close()
            return
        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            pass
        # Reaping and pipe cleanup cannot delay a competition response.
        def reap():
            try:
                process.wait(timeout=5)
            finally:
                if process.stdin:
                    process.stdin.close()
                if job is not None:
                    job.wait_and_close()
        thread = threading.Thread(target=reap, daemon=True)
        self._reapers = [item for item in self._reapers if item.is_alive()]
        self._reapers.append(thread)
        thread.start()

    def close(self):
        self._retire()
        # Shutdown can wait for its own children, while search() never waits
        # for worker cleanup before returning a move.
        for thread in self._reapers:
            thread.join(timeout=5)
        self._reapers.clear()

    def search(self, board, to_play):
        started = time.perf_counter()
        legal = legal_moves(board, to_play)
        best = {"move": legal[0] if legal else None, "visits": 0, "value": 0.0}
        limit = max(0.0, min(self.time_limit, self.remaining_game_seconds))
        # Leave explicit room for state updates and flushing the response.
        deadline = started + max(0.0, limit - min(0.10, limit * 0.1))
        reason = "clock_exhausted"
        if legal and deadline - time.perf_counter() >= 0.05:
            try:
                self._start()
                while not self._ready and time.perf_counter() < deadline:
                    try:
                        message = self.messages.get(timeout=max(0.0, deadline - time.perf_counter()))
                    except queue.Empty:
                        break
                    if message.get("kind") != "ready":
                        raise ValueError("worker failed before initialization completed")
                    self._ready = True
                if not self._ready:
                    raise TimeoutError("worker initialization exceeded move budget")
                self.identity += 1
                request = {"id": self.identity, "board": list(board), "to_play": to_play,
                           "time_limit": max(0.001, deadline - time.perf_counter() - 0.05)}
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                reason = "hard_timeout"
                while time.perf_counter() < deadline:
                    try:
                        message = self.messages.get(timeout=max(0.0, deadline - time.perf_counter()))
                    except queue.Empty:
                        break
                    if message.get("kind") in {"closed", "invalid_output"}:
                        reason = "worker_failure"
                        break
                    if message.get("id") != self.identity:
                        continue
                    if message.get("kind") in {"progress", "result"} and message.get("move") in legal:
                        best = message
                    if message.get("kind") == "result":
                        reason = "worker_" + message.get("reason", "mcts")
                        break
                    if message.get("kind") == "error":
                        reason = "worker_error"
                        break
            except TimeoutError:
                reason = "hard_timeout"
            except (OSError, ValueError, TypeError):
                reason = "worker_failure"
        if not reason.startswith("worker_") or reason in {"worker_error", "worker_failure"}:
            self._retire()
        self.last_reason = reason
        return SimpleNamespace(
            move=best.get("move"), visits=best.get("visits", 0), value=best.get("value", 0.0),
            confidence=best.get("confidence", 0.0), margin=best.get("margin", 0.0),
            elapsed=time.perf_counter() - started, top_candidates=[],
            reused_tree=best.get("reused_tree", False), reason=reason)


class CompetitionProtocol(SAUNoGoEngine):
    def handle_line(self, line):
        parts = line.replace("\x00", "").lstrip("\ufeff").split()
        timed = bool(parts) and parts[0].lower() in {"new", "move", "error"}
        started = time.perf_counter()
        previous = 0.0 if parts and parts[0].lower() == "new" else self.search_seconds
        try:
            return super().handle_line(line)
        finally:
            if timed:
                # Includes exception/fallback work, state updates and logging.
                self.search_seconds = previous + time.perf_counter() - started

    def _choose_and_play(self):
        usable = max(0.0, self.config.game_time_limit - max(0.0, self.config.time_reserve))
        remaining = max(0.0, usable - self.search_seconds)
        if self.config.game_time_limit > 0 and remaining < 0.05:
            self._log("search skipped: usable game clock exhausted")
            return self._safe_move_response()
        self.searcher.remaining_game_seconds = remaining if self.config.game_time_limit > 0 else float("inf")
        return super()._choose_and_play()


def run_protocol(engine, stdin, stdout):
    for raw in stdin:
        command = raw.replace("\x00", "").lstrip("\ufeff").strip()
        engine._log(f"received {command!r}")
        response = engine.handle_line(command)
        if response:
            before_flush = time.perf_counter()
            stdout.write(response + "\n")
            stdout.flush()
            engine.search_seconds += time.perf_counter() - before_flush
            # This marker occurs after flush, unlike the legacy pre-print log.
            engine._log(f"sent {response!r}; game_used={engine.search_seconds:.3f}s")
        if command.lower() == "quit":
            break
