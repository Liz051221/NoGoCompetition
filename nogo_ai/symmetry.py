from __future__ import annotations

from .rules import BOARD_SIZE, POINTS, Board, index_to_xy, xy_to_index


def transform_index(index: int, transform: int) -> int:
    x, y = index_to_xy(index)
    last = BOARD_SIZE - 1
    if transform == 0:
        nx, ny = x, y
    elif transform == 1:
        nx, ny = last - y, x
    elif transform == 2:
        nx, ny = last - x, last - y
    elif transform == 3:
        nx, ny = y, last - x
    elif transform == 4:
        nx, ny = last - x, y
    elif transform == 5:
        nx, ny = x, last - y
    elif transform == 6:
        nx, ny = y, x
    elif transform == 7:
        nx, ny = last - y, last - x
    else:
        raise ValueError(f"unknown symmetry transform: {transform}")
    return xy_to_index(nx, ny)


def transform_board(board: Board, transform: int) -> Board:
    values = [0] * POINTS
    for idx, stone in enumerate(board):
        values[transform_index(idx, transform)] = stone
    return tuple(values)


def transform_policy(policy: list[float], transform: int) -> list[float]:
    if len(policy) != POINTS:
        raise ValueError(f"policy length must be {POINTS}, got {len(policy)}")
    values = [0.0] * POINTS
    for idx, value in enumerate(policy):
        values[transform_index(idx, transform)] = value
    return values

