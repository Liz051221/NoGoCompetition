"""CPU-only inference for the portable release; no PyTorch dependency."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .network import Prediction
from .rules import BOARD_SIZE, POINTS, Board, legal_moves


class PortableOnnxNetwork:
    def __init__(self, model: str | Path, threads: int = 2) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(int(threads), os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.metadata = self.session.get_modelmeta().custom_metadata_map
        if self.session.get_inputs()[0].shape[1:] != [5, BOARD_SIZE, BOARD_SIZE]:
            raise ValueError("portable model must have five 9x9 input planes")
        if self.metadata.get("phase_plane") != "false":
            raise ValueError("portable model encoding metadata mismatch")

    @staticmethod
    def encode(positions: list[tuple[Board, int]], legal_lists: list[list[int]]) -> np.ndarray:
        count = len(positions)
        planes = np.zeros((count, 5, POINTS), dtype=np.float32)
        boards = np.asarray([board for board, _ in positions], dtype=np.int8)
        colors = np.asarray([color for _, color in positions], dtype=np.int8)[:, None]
        planes[:, 0] = boards == colors
        planes[:, 1] = boards == -colors
        planes[:, 2] = boards == 0
        for row, ((board, color), legal) in enumerate(zip(positions, legal_lists)):
            planes[row, 3, legal] = 1.0
            planes[row, 4, legal_moves(board, -color)] = 1.0
        return planes.reshape(count, 5, BOARD_SIZE, BOARD_SIZE)

    def predict(self, board: Board, color: int) -> Prediction:
        return self.predict_batch([(board, color)])[0]

    def predict_batch(self, positions: list[tuple[Board, int]]) -> list[Prediction]:
        if not positions:
            return []
        legal_lists = [legal_moves(board, color) for board, color in positions]
        active = [index for index, legal in enumerate(legal_lists) if legal]
        result = [Prediction([0.0] * POINTS, -1.0) for _ in positions]
        if not active:
            return result
        selected = [positions[index] for index in active]
        selected_legal = [legal_lists[index] for index in active]
        x = self.encode(selected, selected_legal)
        logits, values = self.session.run(["policy_logits", "value"], {"board": x})
        if not np.isfinite(logits).all() or not np.isfinite(values).all():
            raise ValueError("model produced non-finite predictions")
        for row, index in enumerate(active):
            legal = legal_lists[index]
            masked = logits[row, legal]
            probabilities = np.exp(masked - np.max(masked))
            probabilities /= probabilities.sum()
            policy = [0.0] * POINTS
            for move, probability in zip(legal, probabilities):
                policy[move] = float(probability)
            result[index] = Prediction(policy, float(values[row]))
        return result
