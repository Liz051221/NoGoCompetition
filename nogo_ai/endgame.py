from __future__ import annotations

from dataclasses import dataclass, field
import time

from .rules import Board, legal_moves, neighbors, play_legal_move


@dataclass(frozen=True, slots=True)
class TacticalMove:
    move: int
    value: float
    reason: str
    move_values: tuple[tuple[int, int], ...] = ()

    def value_for(self, move: int) -> int | None:
        return next((value for candidate, value in self.move_values if candidate == move), None)

    @property
    def losing_moves(self) -> set[int]:
        return {move for move, value in self.move_values if value < 0}

    @property
    def winning_moves(self) -> set[int]:
        return {move for move, value in self.move_values if value > 0}


@dataclass(slots=True)
class EndgameSolver:
    max_legal_moves: int = 12
    max_nodes: int = 50000
    cache_limit: int = 50000
    legal_cache_limit: int = 50000
    cache: dict[tuple[Board, int], bool] = field(default_factory=dict)
    legal_cache: dict[tuple[Board, int], list[int]] = field(default_factory=dict)
    _nodes: int = 0
    _deadline: float | None = None

    def solve(
        self,
        board: Board,
        to_play: int,
        deadline: float | None = None,
    ) -> TacticalMove | None:
        legal = self._legal_moves(board, to_play)
        if not legal or len(legal) > self.max_legal_moves:
            return None

        self._nodes = 0
        self._deadline = deadline
        winning_moves: list[int] = []
        losing_moves: list[int] = []
        unknown_moves: list[int] = []
        ordered = self._order_moves(board, to_play, legal)
        for move in ordered:
            try:
                opponent_wins = self._is_winning(play_legal_move(board, to_play, move), -to_play)
            except SearchBudgetExceeded:
                unknown_moves.append(move)
                continue
            if opponent_wins:
                losing_moves.append(move)
            else:
                winning_moves.append(move)

        move_values = tuple(
            (move, 1 if move in winning_moves else -1 if move in losing_moves else 0)
            for move in legal
        )
        if winning_moves:
            return TacticalMove(winning_moves[0], 1.0, "endgame_win", move_values)
        if len(losing_moves) == len(legal):
            return TacticalMove(losing_moves[0], -1.0, "endgame_loss", move_values)
        if unknown_moves:
            # Partial proofs are still useful: a known losing move must not be
            # selected while an unresolved alternative remains.
            return TacticalMove(unknown_moves[0], 0.0, "endgame_partial", move_values)
        return None

    def _is_winning(self, board: Board, to_play: int) -> bool:
        self._nodes += 1
        if self._nodes > self.max_nodes or (
            self._deadline is not None and time.perf_counter() >= self._deadline
        ):
            raise SearchBudgetExceeded

        key = (board, to_play)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        legal = self._legal_moves(board, to_play)
        if not legal:
            self._store(self.cache, key, False, self.cache_limit)
            return False

        for move in self._order_moves(board, to_play, legal):
            next_board = play_legal_move(board, to_play, move)
            # A move that leaves the opponent without a move is an immediate
            # win and avoids traversing the rest of the subtree.
            if not self._legal_moves(next_board, -to_play):
                self._store(self.cache, key, True, self.cache_limit)
                return True
            if not self._is_winning(next_board, -to_play):
                self._store(self.cache, key, True, self.cache_limit)
                return True

        self._store(self.cache, key, False, self.cache_limit)
        return False

    def _legal_moves(self, board: Board, to_play: int) -> list[int]:
        key = (board, to_play)
        legal = self.legal_cache.get(key)
        if legal is None:
            legal = legal_moves(board, to_play)
            self._store(self.legal_cache, key, legal, self.legal_cache_limit)
        return legal

    @staticmethod
    def _store(cache: dict, key: object, value: object, limit: int) -> None:
        """Keep exact memoization bounded; eviction never changes correctness."""
        if limit > 0 and len(cache) >= limit and key not in cache:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def _order_moves(self, board: Board, to_play: int, legal: list[int]) -> list[int]:
        """Try forcing and low-opponent-mobility moves first.

        This mirrors the strong legacy solver's ordering and makes bounded
        minimax useful even when the full endgame tree is too large.
        """
        scored: list[tuple[int, int, int]] = []
        for move in legal:
            after = play_legal_move(board, to_play, move)
            opponent_moves = self._legal_moves(after, -to_play)
            scored.append((len(opponent_moves), -len(self._legal_moves(after, to_play)), move))
        return [move for _opp, _self, move in sorted(scored)]


class SearchBudgetExceeded(Exception):
    pass


def root_mobility_prior(
    board: Board,
    to_play: int,
    legal: list[int] | None = None,
    max_legal_moves: int = 32,
    min_filled: int = 32,
    deadline: float | None = None,
    max_candidates: int | None = None,
    max_opponent_replies: int | None = None,
) -> dict[int, float]:
    """Rank root moves by worst-case two-ply mobility.

    NoGo positions are often decided by a reply that collapses the mover's
    next-turn legal set.  This is the useful part of the legacy MCTS heuristic
    that is complementary to a policy network.  The result is normalized to
    ``[0, 1]`` and can therefore be blended with network priors.
    """
    legal = legal if legal is not None else legal_moves(board, to_play)
    filled = len(board) - board.count(0)
    if not legal or len(legal) > max_legal_moves or filled < min_filled:
        return {}

    # Evaluate every root move by default.  The old implementation examined
    # only 5-6 moves and a handful of replies, which made the result dependent
    # on network ordering and could miss the only safe move in a dense ending.
    candidate_limit = len(legal) if max_candidates is None else max(1, max_candidates)
    opponent_limit = None if max_opponent_replies is None else max(1, max_opponent_replies)
    preliminary: dict[int, float] = {}
    opponent_by_move: dict[int, list[int]] = {}
    for move in legal:
        if deadline is not None and time.perf_counter() >= deadline:
            return {}
        after_move = play_legal_move(board, to_play, move)
        opponent_moves = legal_moves(after_move, -to_play)
        opponent_by_move[move] = opponent_moves
        if not opponent_moves:
            preliminary[move] = 1000.0
            continue
        degree = sum(board[neighbor] == 0 for neighbor in neighbors(move))
        preliminary[move] = (
            -float(len(opponent_moves)) + 0.05 * degree
        )

    selected = sorted(legal, key=lambda move: preliminary[move], reverse=True)[:candidate_limit]
    collapse_scores: dict[int, float] = {}
    for move in selected:
        if preliminary[move] >= 1000.0:
            collapse_scores[move] = 1000.0
            continue
        after_move = play_legal_move(board, to_play, move)
        opponent_moves = opponent_by_move[move]
        take = len(opponent_moves) if opponent_limit is None else min(opponent_limit, len(opponent_moves))
        offset = move % len(opponent_moves)
        worst = 99
        for index in range(take):
            if deadline is not None and time.perf_counter() >= deadline:
                return _normalize_scores(preliminary)
            if take == len(opponent_moves):
                reply_index = (offset + index) % len(opponent_moves)
            else:
                # Evenly spread a partial sample without repeating replies
                # when the stride and reply count are not coprime.
                reply_index = (offset + (index * len(opponent_moves)) // take) % len(opponent_moves)
            opponent_move = opponent_moves[reply_index]
            after_reply = play_legal_move(after_move, -to_play, opponent_move)
            worst = min(worst, len(legal_moves(after_reply, to_play)))
            if worst == 0:
                break
        collapse_scores[move] = float(worst) - 0.12 * len(opponent_moves)

    preliminary_prior = _normalize_scores(preliminary)
    collapse_prior = _normalize_scores(collapse_scores)
    result = {
        move: (
            0.20 * preliminary_prior.get(move, 0.0)
            + 0.80 * collapse_prior.get(move, 0.0)
            if move in collapse_scores
            else 0.10 * preliminary_prior.get(move, 0.0)
        )
        for move in legal
    }
    return result if any(value > 0.0 for value in result.values()) else {}


def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high <= low + 1.0e-12:
        return {}
    return {move: (score - low) / (high - low) for move, score in scores.items()}


def find_immediate_win(board: Board, to_play: int, legal: list[int] | None = None) -> int | None:
    for move in legal if legal is not None else legal_moves(board, to_play):
        next_board = play_legal_move(board, to_play, move)
        if not legal_moves(next_board, -to_play):
            return move
    return None


def move_allows_immediate_loss(board: Board, to_play: int, move: int) -> bool:
    next_board = play_legal_move(board, to_play, move)
    return find_immediate_win(next_board, -to_play) is not None


def move_allows_forced_two_step_loss(
    board: Board,
    to_play: int,
    move: int,
    opponent_limit: int,
    reply_limit: int,
) -> bool:
    """Return whether the opponent can force a win within its next two moves."""
    after_move = play_legal_move(board, to_play, move)
    opponent = -to_play
    candidates: list[tuple[int, int, Board]] = []
    for opponent_move in legal_moves(after_move, opponent):
        after_opponent = play_legal_move(after_move, opponent, opponent_move)
        replies = legal_moves(after_opponent, to_play)
        if not replies:
            return True
        candidates.append((len(replies), opponent_move, after_opponent))

    candidates.sort(key=lambda item: item[0])
    for reply_count, _opponent_move, after_opponent in candidates[:opponent_limit]:
        if reply_count > reply_limit:
            break
        all_replies_lose = True
        for reply in legal_moves(after_opponent, to_play):
            after_reply = play_legal_move(after_opponent, to_play, reply)
            if not legal_moves(after_reply, opponent) or find_immediate_win(after_reply, opponent) is None:
                all_replies_lose = False
                break
        if all_replies_lose:
            return True
    return False
