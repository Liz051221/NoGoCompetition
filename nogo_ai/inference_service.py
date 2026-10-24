from __future__ import annotations

import time
from collections import deque
from queue import Empty
from typing import Any

from .network import Prediction
from .rules import Board


class RemotePolicyValueNet:
    """Synchronous MCTS client for a shared GPU inference process."""

    def __init__(
        self,
        worker_id: int,
        request_queue: Any,
        response_queue: Any,
        timeout: float = 180.0,
    ) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.timeout = timeout
        self.device = "shared-cuda"
        self._request_id = 0

    def predict(self, board: Board, color: int) -> Prediction:
        return self.predict_batch([(board, color)])[0]

    def predict_batch(self, positions: list[tuple[Board, int]]) -> list[Prediction]:
        if not positions:
            return []
        self._request_id += 1
        request_id = self._request_id
        self.request_queue.put((self.worker_id, request_id, positions))
        try:
            response_id, payload, error = self.response_queue.get(timeout=self.timeout)
        except Empty as exc:
            raise RuntimeError("shared inference server timed out") from exc
        if error is not None:
            raise RuntimeError(f"shared inference server failed: {error}")
        if response_id != request_id:
            raise RuntimeError(
                f"shared inference response mismatch: expected {request_id}, got {response_id}"
            )
        return [Prediction(priors, value) for priors, value in payload]


def run_inference_server(
    request_queue: Any,
    response_queues: list[Any],
    status_queue: Any,
    model: str | None,
    device: str,
    channels: int,
    blocks: int,
    inference_amp: bool,
    inference_tf32: bool,
    max_batch_size: int = 64,
    max_wait_ms: float = 1.0,
) -> None:
    """Serve policy/value requests from several CPU MCTS workers."""
    from .torch_model import TorchPolicyValueNet

    maximum = max(1, max_batch_size)
    wait_seconds = max(0.0, max_wait_ms) / 1000.0
    deferred: deque[tuple[int, int, list[tuple[Board, int]]]] = deque()
    calls = positions_total = 0
    inference_seconds = 0.0
    try:
        network = TorchPolicyValueNet(
            checkpoint=model,
            device=device,
            channels=channels,
            blocks=blocks,
            inference_amp=inference_amp,
            inference_tf32=inference_tf32,
        )
        network.predict_batch([((0,) * 81, 1)])
        network.torch.cuda.synchronize() if str(network.device).startswith("cuda") else None
        status_queue.put(("ready", None))

        stopping = False
        while not stopping:
            if deferred:
                first = deferred.popleft()
            else:
                first = request_queue.get()
                if first is None:
                    break
            requests = [first]
            position_count = len(first[2])
            deadline = time.perf_counter() + wait_seconds
            while position_count < maximum:
                timeout = deadline - time.perf_counter()
                if timeout <= 0.0:
                    break
                try:
                    request = request_queue.get(timeout=timeout)
                except Empty:
                    break
                if request is None:
                    stopping = True
                    break
                request_size = len(request[2])
                if requests and position_count + request_size > maximum:
                    deferred.append(request)
                    break
                requests.append(request)
                position_count += request_size

            combined = [position for _worker, _request, batch in requests for position in batch]
            started = time.perf_counter()
            predictions = network.predict_batch(combined)
            inference_seconds += time.perf_counter() - started
            calls += 1
            positions_total += len(combined)
            offset = 0
            for worker_id, request_id, batch in requests:
                count = len(batch)
                payload = [
                    (prediction.priors, prediction.value)
                    for prediction in predictions[offset : offset + count]
                ]
                response_queues[worker_id].put((request_id, payload, None))
                offset += count
        status_queue.put(
            (
                "stopped",
                {
                    "calls": calls,
                    "positions": positions_total,
                    "average_batch": positions_total / max(1, calls),
                    "inference_seconds": inference_seconds,
                },
            )
        )
    except BaseException as exc:
        message = f"{type(exc).__name__}: {exc}"
        for response_queue in response_queues:
            try:
                response_queue.put((-1, [], message))
            except Exception:
                pass
        try:
            status_queue.put(("error", message))
        except Exception:
            pass
        raise
