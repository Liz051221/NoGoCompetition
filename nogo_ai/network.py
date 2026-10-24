from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rules import (
    BLACK,
    BOARD_SIZE,
    EMPTY,
    POINTS,
    Board,
    group_and_liberties,
    groups,
    is_legal_move,
    legal_moves,
    neighbors,
    play_move,
    stone_count,
)


POLICY_FEATURES = 11
POLICY_HIDDEN = 6
VALUE_FEATURES = 10
VALUE_HIDDEN = 5


def _dot(weights: list[float], values: list[float]) -> float:
    return sum(w * v for w, v in zip(weights, values))


def _tanh(value: float) -> float:
    return math.tanh(max(-20.0, min(20.0, value)))


def _softmax_masked(logits: list[float], legal: list[int]) -> list[float]:
    probs = [0.0] * POINTS
    if not legal:
        return probs
    max_logit = max(logits[idx] for idx in legal)
    total = 0.0
    for idx in legal:
        value = math.exp(max(-60.0, min(60.0, logits[idx] - max_logit)))
        probs[idx] = value
        total += value
    if total <= 0.0:
        even = 1.0 / len(legal)
        for idx in legal:
            probs[idx] = even
        return probs
    return [value / total for value in probs]


def default_weights() -> dict[str, Any]:
    return {
        "version": 1,
        "policy_w1": [
            [0.0, 2.2, 0.2, 0.0, 0.0, 0.0, 0.4, 0.0, -0.2, -0.3, 0.0],
            [-0.6, 0.0, 0.0, 0.5, 0.0, 0.7, 2.1, 0.0, -2.4, 0.0, 0.0],
            [-0.3, 0.0, 0.0, 1.7, 0.2, 0.0, 0.9, 0.0, -1.6, 0.1, 0.0],
            [-0.5, 0.0, -0.1, 0.0, 1.2, 0.0, 0.3, 2.0, -0.9, 0.2, 0.0],
            [0.1, 0.3, 1.0, 0.2, 0.0, 0.4, 0.6, 0.0, -0.4, 1.2, 0.0],
            [0.0, -0.2, 0.6, 0.1, 0.7, 0.2, 0.5, 0.6, -1.0, 0.8, 0.0],
        ],
        "policy_b1": [0.0, 0.0, 0.0, 0.0, -0.3, 0.0],
        "policy_w2": [0.35, 1.05, 0.75, 0.65, 0.25, 0.40],
        "policy_b2": 0.0,
        "value_w1": [
            [0.0, 2.4, -2.2, 0.4, 0.0, 0.6, -0.6, 0.0, 0.0, 0.1],
            [0.0, 1.4, -1.3, 0.1, 0.0, 1.2, -1.0, -0.2, 0.2, 0.0],
            [0.0, 0.6, -0.5, 0.8, 0.0, 0.3, -0.2, 0.0, 0.0, -0.1],
            [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -0.8, 0.8, 0.2],
            [0.0, 0.8, -0.8, 0.2, -0.4, 0.7, -0.7, -0.2, 0.2, 0.5],
        ],
        "value_b1": [0.0, 0.0, 0.0, 0.0, 0.0],
        "value_w2": [0.75, 0.65, 0.35, 0.35, 0.55],
        "value_b2": 0.0,
    }


@dataclass(slots=True)
class Prediction:
    priors: list[float]
    value: float


class TinyPolicyValueNet:
    """A compact policy-value neural network for 9x9 NoGo.

    The policy head is shared across legal moves. The value head evaluates the
    whole board from the side-to-move perspective. The default weights are small
    and heuristic-seeded so the engine is useful before external training.
    """

    def __init__(self, weights: dict[str, Any] | None = None) -> None:
        self.weights = weights if weights is not None else default_weights()
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> "TinyPolicyValueNet":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.weights, fh, indent=2)

    def predict(self, board: Board, color: int) -> Prediction:
        legal = legal_moves(board, color)
        if not legal:
            return Prediction([0.0] * POINTS, -1.0)

        logits = [-1.0e9] * POINTS
        for move in legal:
            logits[move] = self._policy_logit(board, color, move)
        priors = _softmax_masked(logits, legal)
        value = self._value(board, color)
        return Prediction(priors, value)

    def _policy_logit(self, board: Board, color: int, move: int) -> float:
        features = move_features(board, color, move)
        hidden = [
            _tanh(_dot(row, features) + bias)
            for row, bias in zip(self.weights["policy_w1"], self.weights["policy_b1"])
        ]
        return _dot(self.weights["policy_w2"], hidden) + float(self.weights["policy_b2"])

    def _value(self, board: Board, color: int) -> float:
        features = value_features(board, color)
        hidden = [
            _tanh(_dot(row, features) + bias)
            for row, bias in zip(self.weights["value_w1"], self.weights["value_b1"])
        ]
        return _tanh(_dot(self.weights["value_w2"], hidden) + float(self.weights["value_b2"]))

    def _validate(self) -> None:
        if len(self.weights["policy_w1"]) != POLICY_HIDDEN:
            raise ValueError("invalid policy hidden width")
        if any(len(row) != POLICY_FEATURES for row in self.weights["policy_w1"]):
            raise ValueError("invalid policy feature width")
        if len(self.weights["policy_b1"]) != POLICY_HIDDEN:
            raise ValueError("invalid policy bias width")
        if len(self.weights["policy_w2"]) != POLICY_HIDDEN:
            raise ValueError("invalid policy output width")
        if len(self.weights["value_w1"]) != VALUE_HIDDEN:
            raise ValueError("invalid value hidden width")
        if any(len(row) != VALUE_FEATURES for row in self.weights["value_w1"]):
            raise ValueError("invalid value feature width")
        if len(self.weights["value_b1"]) != VALUE_HIDDEN:
            raise ValueError("invalid value bias width")
        if len(self.weights["value_w2"]) != VALUE_HIDDEN:
            raise ValueError("invalid value output width")


def move_features(board: Board, color: int, move: int) -> list[float]:
    x = move % BOARD_SIZE
    y = move // BOARD_SIZE
    center = 1.0 - (abs(x - 4) + abs(y - 4)) / 8.0
    edge = min(x, y, BOARD_SIZE - 1 - x, BOARD_SIZE - 1 - y) / 4.0
    filled = 1.0 - (sum(1 for stone in board if stone == EMPTY) / POINTS)

    after = play_move(board, color, move)
    _, own_liberties = group_and_liberties(after, move)
    own_adj = opp_adj = empty_adj = 0
    enemy_min_liberties = 4
    checked: set[int] = set()

    for nb in neighbors(move):
        if board[nb] == color:
            own_adj += 1
        elif board[nb] == -color:
            opp_adj += 1
            if nb not in checked:
                group, liberties = group_and_liberties(after, nb)
                checked.update(group)
                enemy_min_liberties = min(enemy_min_liberties, len(liberties))
        else:
            empty_adj += 1

    own_lib_score = min(len(own_liberties), 8) / 8.0
    pressure = (4.0 - min(enemy_min_liberties, 4)) / 4.0 if opp_adj else 0.0
    self_atari = 1.0 if len(own_liberties) <= 1 else 0.0

    return [
        1.0,
        center,
        edge,
        own_adj / 4.0,
        opp_adj / 4.0,
        empty_adj / 4.0,
        own_lib_score,
        pressure,
        self_atari,
        filled,
        1.0 if color == BLACK else -1.0,
    ]


def value_features(board: Board, color: int) -> list[float]:
    legal_self = len(legal_moves(board, color)) / POINTS
    legal_opp = len(legal_moves(board, -color)) / POINTS
    stone_balance = (stone_count(board, color) - stone_count(board, -color)) / POINTS
    empty_ratio = sum(1 for stone in board if stone == EMPTY) / POINTS
    progress = 1.0 - empty_ratio

    own_groups = groups(board, color)
    opp_groups = groups(board, -color)
    own_lib_avg = _avg_liberties(own_groups)
    opp_lib_avg = _avg_liberties(opp_groups)

    return [
        1.0,
        legal_self,
        legal_opp,
        stone_balance,
        empty_ratio,
        own_lib_avg,
        opp_lib_avg,
        min(len(own_groups), 20) / 20.0,
        min(len(opp_groups), 20) / 20.0,
        progress,
    ]


def _avg_liberties(group_data: list[tuple[set[int], set[int]]]) -> float:
    if not group_data:
        return 0.0
    return min(sum(len(libs) for _, libs in group_data) / len(group_data), 8.0) / 8.0

