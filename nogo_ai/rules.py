from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

BOARD_SIZE = 9
POINTS = BOARD_SIZE * BOARD_SIZE
EMPTY = 0
BLACK = 1
WHITE = -1


@dataclass(frozen=True, slots=True)
class Move:
    index: int

    @property
    def x(self) -> int:
        return self.index % BOARD_SIZE

    @property
    def y(self) -> int:
        return self.index // BOARD_SIZE

    def to_coord(self) -> str:
        return index_to_coord(self.index)


Board = tuple[int, ...]


def new_board() -> Board:
    return (EMPTY,) * POINTS


def opponent(color: int) -> int:
    return -color


def color_name(color: int) -> str:
    return "black" if color == BLACK else "white"


def parse_color(value: str) -> int:
    value = value.strip().lower()
    if value == "black":
        return BLACK
    if value == "white":
        return WHITE
    raise ValueError(f"unknown side: {value!r}")


def xy_to_index(x: int, y: int) -> int:
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        raise ValueError(f"coordinate out of board: {x}, {y}")
    return y * BOARD_SIZE + x


def index_to_xy(index: int) -> tuple[int, int]:
    if not (0 <= index < POINTS):
        raise ValueError(f"index out of board: {index}")
    return index % BOARD_SIZE, index // BOARD_SIZE


def coord_to_index(coord: str) -> int:
    coord = coord.strip().upper()
    if len(coord) != 2:
        raise ValueError(f"invalid coordinate: {coord!r}")
    return xy_to_index(ord(coord[0]) - ord("A"), ord(coord[1]) - ord("A"))


def index_to_coord(index: int) -> str:
    x, y = index_to_xy(index)
    return f"{chr(ord('A') + x)}{chr(ord('A') + y)}"


def neighbors(index: int) -> Iterable[int]:
    x, y = index_to_xy(index)
    if x > 0:
        yield index - 1
    if x + 1 < BOARD_SIZE:
        yield index + 1
    if y > 0:
        yield index - BOARD_SIZE
    if y + 1 < BOARD_SIZE:
        yield index + BOARD_SIZE


def group_and_liberties(board: Board, start: int) -> tuple[set[int], set[int]]:
    color = board[start]
    if color == EMPTY:
        return set(), {start}

    group: set[int] = set()
    liberties: set[int] = set()
    queue: deque[int] = deque([start])
    group.add(start)

    while queue:
        point = queue.popleft()
        for nb in neighbors(point):
            stone = board[nb]
            if stone == EMPTY:
                liberties.add(nb)
            elif stone == color and nb not in group:
                group.add(nb)
                queue.append(nb)

    return group, liberties


def play_move(board: Board, color: int, move: int) -> Board:
    if not is_legal_move(board, color, move):
        raise ValueError(f"illegal NoGo move: {index_to_coord(move)}")
    return play_legal_move(board, color, move)


def play_legal_move(board: Board, color: int, move: int) -> Board:
    """Place a stone when the caller has already established legality."""
    mutable = list(board)
    mutable[move] = color
    return tuple(mutable)


@lru_cache(maxsize=20_000)
def _is_legal_move_cached(board: Board, color: int, move: int) -> bool:
    if not (0 <= move < POINTS) or board[move] != EMPTY:
        return False

    mutable = list(board)
    mutable[move] = color
    next_board = tuple(mutable)

    _, own_liberties = group_and_liberties(next_board, move)
    if not own_liberties:
        return False

    enemy = opponent(color)
    checked: set[int] = set()
    for nb in neighbors(move):
        if next_board[nb] != enemy or nb in checked:
            continue
        group, liberties = group_and_liberties(next_board, nb)
        checked.update(group)
        if not liberties:
            return False

    return True


def is_legal_move(board: Board, color: int, move: int) -> bool:
    """Return whether ``move`` is legal, with a bounded position cache.

    MCTS asks the same legality questions through several tactical filters,
    endgame probes, and network input encoding. The board is immutable, so a
    bounded cache is safe and avoids repeating the flood-fill work.
    """
    return _is_legal_move_cached(board, color, move)


@lru_cache(maxsize=10_000)
def _legal_moves_cached(board: Board, color: int) -> tuple[int, ...]:
    return tuple(
        idx for idx, stone in enumerate(board)
        if stone == EMPTY and _is_legal_move_cached(board, color, idx)
    )


def legal_moves(board: Board, color: int) -> list[int]:
    return list(_legal_moves_cached(board, color))


def has_legal_move(board: Board, color: int) -> bool:
    return bool(_legal_moves_cached(board, color))


def stone_count(board: Board, color: int) -> int:
    return sum(1 for stone in board if stone == color)


def empty_count(board: Board) -> int:
    return sum(1 for stone in board if stone == EMPTY)


def groups(board: Board, color: int) -> list[tuple[set[int], set[int]]]:
    seen: set[int] = set()
    result: list[tuple[set[int], set[int]]] = []
    for idx, stone in enumerate(board):
        if stone != color or idx in seen:
            continue
        group, liberties = group_and_liberties(board, idx)
        seen.update(group)
        result.append((group, liberties))
    return result


def board_to_lines(board: Board) -> list[str]:
    chars = {BLACK: "X", WHITE: "O", EMPTY: "."}
    return [
        "".join(chars[board[xy_to_index(x, y)]] for x in range(BOARD_SIZE))
        for y in range(BOARD_SIZE)
    ]
