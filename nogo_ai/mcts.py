from __future__ import annotations

import math
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .endgame import (
    EndgameSolver,
    find_immediate_win,
    move_allows_forced_two_step_loss,
    move_allows_immediate_loss,
    root_mobility_prior,
)
from .network import TinyPolicyValueNet
from .network import Prediction
from .rules import Board, legal_moves, play_legal_move


@dataclass(slots=True)
class Edge:
    move: int
    prior: float
    child: "Node | None" = None
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class Node:
    board: Board
    to_play: int
    visits: int = 0
    edges: dict[int, Edge] = field(default_factory=dict)
    expanded: bool = False
    root_tactical_prior: dict[int, float] | None = None
    forced_move: int | None = None


@dataclass(slots=True)
class SearchCandidate:
    move: int
    visits: int
    q: float
    prior: float
    visit_share: float
    tactical: float = 0.0


@dataclass(slots=True)
class SearchResult:
    move: int | None
    visits: int
    value: float
    policy: list[tuple[int, int, float]]
    visit_policy: list[float]
    confidence: float = 0.0
    margin: float = 0.0
    elapsed: float = 0.0
    top_candidates: list[SearchCandidate] = field(default_factory=list)
    reused_tree: bool = False
    reason: str = "mcts"


@dataclass(slots=True)
class _PendingSimulation:
    leaf: Node
    path: list[tuple[Node, Edge]]
    terminal_value: float | None = None


class NeuralMCTS:
    def __init__(
        self,
        network: TinyPolicyValueNet,
        simulations: int = 160,
        time_limit: float = 1.2,
        cpuct: float = 1.45,
        seed: int | None = None,
        tactical_moves: int = 24,
        endgame_moves: int = 16,
        endgame_nodes: int = 200000,
        endgame_time_fraction: float = 0.35,
        root_dirichlet_alpha: float = 0.0,
        root_exploration_fraction: float = 0.0,
        root_tactical_weight: float = 0.0,
        root_tactical_selection_weight: float | None = None,
        prediction_cache_limit: int = 50000,
        inference_batch_size: int = 32,
        virtual_loss: float = 1.0,
        fixed_simulations: bool = False,
    ) -> None:
        self.network = network
        self.simulations = max(1, simulations)
        self.time_limit = max(0.05, time_limit)
        self.fixed_simulations = fixed_simulations
        self.cpuct = cpuct
        self.random = random.Random(seed)
        self.tactical_moves = tactical_moves
        self.endgame_solver = EndgameSolver(max_legal_moves=endgame_moves, max_nodes=endgame_nodes)
        if not math.isfinite(endgame_time_fraction) or not 0.0 <= endgame_time_fraction <= 1.0:
            raise ValueError("endgame_time_fraction must be between zero and one")
        self.endgame_time_fraction = float(endgame_time_fraction)
        self.prediction_cache: OrderedDict[tuple[Board, int], Prediction] = OrderedDict()
        self.inference_calls = 0
        self.inference_positions = 0
        self.inference_seconds = 0.0
        self.prediction_cache_hits = 0
        self.prediction_cache_limit = max(1, prediction_cache_limit)
        self.inference_batch_size = max(
            1,
            inference_batch_size if callable(getattr(network, "predict_batch", None)) else 1,
        )
        self.virtual_loss = max(0.0, virtual_loss)
        self.root_dirichlet_alpha = max(0.0, root_dirichlet_alpha)
        self.root_exploration_fraction = max(0.0, min(1.0, root_exploration_fraction))
        self.root_tactical_weight = max(0.0, min(1.0, root_tactical_weight))
        selection_weight = (
            self.root_tactical_weight
            if root_tactical_selection_weight is None
            else root_tactical_selection_weight
        )
        self.root_tactical_selection_weight = max(0.0, min(1.0, selection_weight))
        self._last_root: Node | None = None
        self._noise_root: Node | None = None

    def search(
        self,
        board: Board,
        to_play: int,
        add_root_noise: bool = True,
    ) -> SearchResult:
        start = time.perf_counter()
        deadline = math.inf if self.fixed_simulations else start + self.time_limit
        legal = legal_moves(board, to_play)
        if not legal:
            return SearchResult(
                None, 0, -1.0, [], [0.0] * len(board),
                elapsed=time.perf_counter() - start,
                reason="terminal",
            )
        # A direct winning move is a complete proof in NoGo: after the move
        # the opponent has no legal reply.  The old fixed-size gate skipped
        # this proof whenever a dense position had more than 24 legal moves.
        # Keep the cheap early-game gate, but scan every legal move once the
        # board is sufficiently full, where tactical misses are most costly.
        filled = len(board) - board.count(0)
        scan_immediate = (
            len(legal) <= self.tactical_moves
            or (
                (self.fixed_simulations or self.time_limit >= 0.25)
                and filled >= 36
                and len(legal) <= 48
            )
        )
        immediate = find_immediate_win(board, to_play, legal) if scan_immediate else None
        if immediate is not None:
            return self._forced_result(
                move=immediate,
                visits=0,
                value=1.0,
                legal=legal,
                elapsed=time.perf_counter() - start,
                reason="immediate_win",
            )

        if len(legal) == 1:
            visit_policy = [0.0] * len(board)
            visit_policy[legal[0]] = 1.0
            candidate = SearchCandidate(legal[0], 1, 0.0, 1.0, 1.0)
            return SearchResult(
                legal[0],
                1,
                0.0,
                [(legal[0], 1, 0.0)],
                visit_policy,
                confidence=1.0,
                margin=1.0,
                elapsed=time.perf_counter() - start,
                top_candidates=[candidate],
                reason="single_legal",
            )

        # Exact endgame search is valuable, but it must not consume the whole
        # move budget. Without a separate deadline, an unresolved proof can
        # leave the root unexpanded and make the fallback choose randomly.
        endgame = None
        if self.fixed_simulations and self.endgame_time_fraction > 0.0:
            # In diagnostic mode the solver uses its node cap exclusively.
            endgame = self.endgame_solver.solve(board, to_play)
        elif self.endgame_time_fraction > 0.0:
            solver_started = time.perf_counter()
            remaining = max(0.0, deadline - solver_started)
            solver_budget = min(
                remaining * self.endgame_time_fraction,
                max(0.0, remaining - 0.005),
            )
            if solver_budget > 0.0:
                endgame = self.endgame_solver.solve(
                    board,
                    to_play,
                    deadline=solver_started + solver_budget,
                )
        if endgame is not None and endgame.value > 0.0:
            return self._forced_result(
                move=endgame.move,
                visits=0,
                value=endgame.value,
                legal=legal,
                elapsed=time.perf_counter() - start,
                reason=endgame.reason,
            )

        root, reused_tree = self._reuse_root(board, to_play)
        completed = 0
        tactical_prior: dict[int, float] = {}
        # Expand the root once, prepare its tactical prior, and inject
        # Dirichlet noise at most once for this root node. A reused root may
        # be searched again by protocol/recovery code, so a per-node marker is
        # required instead of blindly perturbing its priors on every search.
        if time.perf_counter() < deadline:
            self._simulate(root)
            completed = 1
            # A proof found while expanding a dense late-game root must not be
            # diluted by Dirichlet noise or by network priors.
            if root.forced_move is not None and root.forced_move in legal:
                return self._forced_result(
                    move=root.forced_move,
                    visits=completed,
                    value=1.0,
                    legal=legal,
                    elapsed=time.perf_counter() - start,
                    reason="immediate_win",
                )
            tactical_prior = self._apply_root_tactical_prior(root, board, to_play, legal, deadline)
            if add_root_noise and self._noise_root is not root:
                self._add_root_exploration_noise(root)
                self._noise_root = root

        while completed < self.simulations and time.perf_counter() < deadline:
            batch_size = min(self.inference_batch_size, self.simulations - completed)
            added = self._simulate_batch(root, batch_size, deadline)
            if added <= 0:
                break
            completed += added

        if not root.edges:
            # A partial exact search still has a move-ordering suggestion. Use
            # it when the search budget was exhausted before root expansion;
            # random fallback is especially harmful in late-game positions.
            fallback = (
                endgame.move
                if endgame is not None and endgame.move in legal
                else self.random.choice(legal)
            )
            visit_policy = [0.0] * len(board)
            visit_policy[fallback] = 1.0
            candidate = SearchCandidate(fallback, 0, 0.0, 0.0, 1.0)
            return SearchResult(
                fallback,
                completed,
                0.0,
                [(fallback, 0, 0.0)],
                visit_policy,
                confidence=0.0,
                margin=0.0,
                elapsed=time.perf_counter() - start,
                top_candidates=[candidate],
                reason="fallback",
            )

        ranked = sorted(
            root.edges.values(),
            key=lambda edge: (edge.visits, edge.q, edge.prior),
            reverse=True,
        )
        if endgame is not None:
            proven_safe = [edge for edge in ranked if edge.move not in endgame.losing_moves]
            if proven_safe:
                ranked = proven_safe
            elif endgame.value < 0.0:
                # Every move is proven losing. Report the exact root value and
                # select deterministically; network saturation must not turn a
                # solved loss into a false +1 evaluation.
                chosen = next((edge for edge in ranked if edge.move == endgame.move), ranked[0])
                return self._forced_result(
                    move=chosen.move,
                    visits=completed,
                    value=-1.0,
                    legal=legal,
                    elapsed=time.perf_counter() - start,
                    reason=endgame.reason,
                )
        # In a dense late position, an otherwise strong network move can still
        # hand the opponent a forced immediate win. Match the wider tactical
        # scan used at the root: filter these dominated moves before choosing
        # the visit winner, but keep the cheap early-game gate.
        late_tactical = filled >= 36 and len(legal) <= 48
        if len(legal) <= self.tactical_moves or late_tactical:
            safe_ranked = [edge for edge in ranked if not move_allows_immediate_loss(board, to_play, edge.move)]
            if safe_ranked:
                ranked = safe_ranked
        filled = len(board) - board.count(0)
        if len(legal) <= 18 or (filled >= 50 and len(legal) <= 24):
            opponent_limit = 81 if len(legal) <= 10 else 12
            reply_limit = 4 if len(legal) <= 10 else 3
            two_step_safe = [
                edge
                for edge in ranked
                if not move_allows_forced_two_step_loss(
                    board,
                    to_play,
                    edge.move,
                    opponent_limit=opponent_limit,
                    reply_limit=reply_limit,
                )
            ]
            if two_step_safe:
                ranked = two_step_safe

        selection_weight = self.root_tactical_selection_weight if tactical_prior else 0.0
        if selection_weight > 0.0:
            max_visits = max(1, max(edge.visits for edge in ranked))
            ranked.sort(
                key=lambda edge: (
                    (1.0 - selection_weight) * (edge.visits / max_visits)
                    + selection_weight * tactical_prior.get(edge.move, 0.0),
                    edge.visits,
                    edge.q,
                    edge.prior,
                ),
                reverse=True,
            )
        best = ranked[0]
        policy = [(edge.move, edge.visits, edge.q) for edge in ranked[:10]]
        eligible_edges = ranked
        max_visits = max(1, max(edge.visits for edge in eligible_edges))
        effective_visits = {
            edge.move: (
                (1.0 - selection_weight) * edge.visits
                + selection_weight * max_visits * tactical_prior.get(edge.move, 0.0)
            )
            for edge in eligible_edges
        }
        total_visits = sum(effective_visits.values())
        visit_policy = [0.0] * len(board)
        shares: dict[int, float] = {}
        if total_visits > 0:
            for edge in eligible_edges:
                share = effective_visits[edge.move] / total_visits
                visit_policy[edge.move] = share
                shares[edge.move] = share
        else:
            total_prior = sum(edge.prior for edge in eligible_edges)
            for edge in eligible_edges:
                share = edge.prior / total_prior if total_prior else 0.0
                visit_policy[edge.move] = share
                shares[edge.move] = share

        top_candidates = [
            SearchCandidate(
                move=edge.move,
                visits=edge.visits,
                q=edge.q,
                prior=edge.prior,
                visit_share=shares.get(edge.move, 0.0),
                tactical=tactical_prior.get(edge.move, 0.0),
            )
            for edge in ranked[:10]
        ]
        confidence = top_candidates[0].visit_share if top_candidates else 0.0
        second = top_candidates[1].visit_share if len(top_candidates) > 1 else 0.0
        margin = max(0.0, confidence - second)
        return SearchResult(
            best.move,
            completed,
            best.q,
            policy,
            visit_policy,
            confidence=confidence,
            margin=margin,
            elapsed=time.perf_counter() - start,
            top_candidates=top_candidates,
            reused_tree=reused_tree,
            reason="mcts",
        )

    def _reuse_root(self, board: Board, to_play: int) -> tuple[Node, bool]:
        previous = self._last_root
        if previous is not None:
            if previous.board == board and previous.to_play == to_play:
                return previous, True
            children = [edge.child for edge in previous.edges.values() if edge.child is not None]
            for child in children:
                if child.board == board and child.to_play == to_play:
                    self._last_root = child
                    return child, True
            for child in children:
                for edge in child.edges.values():
                    grandchild = edge.child
                    if grandchild is not None and grandchild.board == board and grandchild.to_play == to_play:
                        self._last_root = grandchild
                        return grandchild, True

        root = Node(board=board, to_play=to_play)
        self._last_root = root
        return root, False

    def _apply_root_tactical_prior(
        self,
        root: Node,
        board: Board,
        to_play: int,
        legal: list[int],
        deadline: float,
    ) -> dict[int, float]:
        if root.root_tactical_prior is not None:
            return root.root_tactical_prior
        if (
            max(self.root_tactical_weight, self.root_tactical_selection_weight) <= 0.0
            or (not self.fixed_simulations and self.time_limit < 1.5)
        ):
            return {}
        tactical_deadline = math.inf if self.fixed_simulations else min(
            deadline,
            time.perf_counter() + min(1.0, max(0.10, self.time_limit * 0.10)),
        )
        tactical = root_mobility_prior(
            board,
            to_play,
            legal=legal,
            deadline=tactical_deadline,
        )
        if not tactical:
            return {}
        root.root_tactical_prior = tactical
        weight = self.root_tactical_weight
        if weight <= 0.0:
            return tactical
        blended = {
            move: (1.0 - weight) * edge.prior + weight * tactical.get(move, 0.0)
            for move, edge in root.edges.items()
        }
        total = sum(blended.values())
        if total <= 0.0:
            return tactical
        for move, edge in root.edges.items():
            edge.prior = blended[move] / total
        return tactical

    def _simulate(self, node: Node) -> float:
        if not node.expanded:
            return self._expand(node)

        if not node.edges:
            node.visits += 1
            return -1.0

        edge = self._select_edge(node)
        if edge.child is None:
            next_board = play_legal_move(node.board, node.to_play, edge.move)
            edge.child = Node(board=next_board, to_play=-node.to_play)

        child_value = self._simulate(edge.child)
        value = -child_value

        edge.visits += 1
        edge.value_sum += value
        node.visits += 1
        return value

    def _simulate_batch(self, root: Node, batch_size: int, deadline: float) -> int:
        """Collect and expand several independent leaves in one network call.

        Virtual loss temporarily diverts later paths away from a leaf already
        selected in this batch. It is removed during backup, so it does not
        change the final tree values.
        """
        if batch_size <= 1 or not callable(getattr(self.network, "predict_batch", None)):
            if time.perf_counter() >= deadline:
                return 0
            self._simulate(root)
            return 1

        pending: list[_PendingSimulation] = []
        unique_leaves: list[Node] = []
        leaf_positions: dict[tuple[Board, int], int] = {}
        for _ in range(batch_size):
            if time.perf_counter() >= deadline and pending:
                break
            item = self._collect_simulation(root)
            pending.append(item)
            if item.terminal_value is None:
                key = (item.leaf.board, item.leaf.to_play)
                if key not in leaf_positions:
                    leaf_positions[key] = len(unique_leaves)
                    unique_leaves.append(item.leaf)

        if not pending:
            return 0

        predictions: list[Prediction] = []
        if unique_leaves:
            positions = [(leaf.board, leaf.to_play) for leaf in unique_leaves]
            predictions = self._predict_batch(positions)
            if len(predictions) != len(unique_leaves):
                raise RuntimeError("predict_batch returned an incorrect number of predictions")

        leaf_values: dict[tuple[Board, int], float] = {}
        for leaf, prediction in zip(unique_leaves, predictions):
            key = (leaf.board, leaf.to_play)
            leaf_values[key] = self._expand_with_prediction(leaf, prediction)

        seen_leaves: set[tuple[Board, int]] = set()
        for item in pending:
            if item.terminal_value is not None:
                value = item.terminal_value
            else:
                key = (item.leaf.board, item.leaf.to_play)
                value = leaf_values[key]
                # A repeated leaf is expanded once but was reached by each
                # selected path in this batch.
                if key in seen_leaves:
                    item.leaf.visits += 1
                seen_leaves.add(key)
            self._backup_virtual_path(item.path, value)
        return len(pending)

    def _collect_simulation(self, root: Node) -> _PendingSimulation:
        node = root
        path: list[tuple[Node, Edge]] = []
        while node.expanded:
            if not node.edges:
                node.visits += 1
                return _PendingSimulation(node, path, -1.0)
            edge = self._select_edge(node)
            edge.visits += 1
            edge.value_sum -= self.virtual_loss
            node.visits += 1
            path.append((node, edge))
            if edge.child is None:
                edge.child = Node(
                    board=play_legal_move(node.board, node.to_play, edge.move),
                    to_play=-node.to_play,
                )
            node = edge.child
        return _PendingSimulation(node, path)

    def _backup_virtual_path(self, path: list[tuple[Node, Edge]], value: float) -> None:
        for _parent, edge in reversed(path):
            value = -value
            # ``value`` enters this function from the leaf side-to-move
            # perspective. Each edge stores value from its parent player's
            # perspective, matching the scalar _simulate() backup above.
            edge.value_sum += self.virtual_loss + value

    def _expand(self, node: Node) -> float:
        if not legal_moves(node.board, node.to_play):
            node.expanded = True
            node.visits += 1
            return -1.0
        prediction = self._predict(node.board, node.to_play)
        return self._expand_with_prediction(node, prediction)

    def _expand_with_prediction(self, node: Node, prediction: Prediction) -> float:
        legal = legal_moves(node.board, node.to_play)
        if not legal:
            node.expanded = True
            node.visits += 1
            return -1.0
        priors = {
            move: (
                max(0.0, float(prediction.priors[move]))
                if math.isfinite(float(prediction.priors[move]))
                else 0.0
            )
            for move in legal
            if move < len(prediction.priors)
        }
        if len(priors) != len(legal):
            priors = {move: 0.0 for move in legal}
        prior_total = sum(priors.values())
        if prior_total <= 0.0 and legal:
            prior_total = float(len(legal))
            priors = {move: 1.0 for move in legal}
        for move in legal:
            # MCTS remains rule-safe even if a custom network emits an
            # illegal or all-zero prior vector.
            node.edges[move] = Edge(move=move, prior=priors[move] / prior_total)
        node.expanded = True
        node.visits += 1
        filled = len(node.board) - node.board.count(0)
        scan_immediate = (
            len(node.edges) <= self.tactical_moves
            or (filled >= 36 and len(node.edges) <= 48)
        )
        if node.edges and scan_immediate:
            immediate = find_immediate_win(node.board, node.to_play, list(node.edges))
            if immediate is not None:
                node.forced_move = immediate
                # Keep the proof visible to both scalar and batched MCTS. The
                # selector also checks forced_move, so this remains decisive
                # after root exploration noise is applied.
                for move, edge in node.edges.items():
                    edge.prior = 1.0 if move == immediate else 0.0
                return 1.0
        value = float(prediction.value)
        return value if math.isfinite(value) else 0.0

    def _select_edge(self, node: Node) -> Edge:
        if node.forced_move is not None:
            forced = node.edges.get(node.forced_move)
            if forced is not None:
                return forced
        parent_visits = max(1, node.visits)
        sqrt_parent = math.sqrt(parent_visits)

        def score(edge: Edge) -> float:
            explore = self.cpuct * edge.prior * sqrt_parent / (1 + edge.visits)
            return edge.q + explore

        best_score = -1.0e9
        best_edges: list[Edge] = []
        for edge in node.edges.values():
            value = score(edge)
            if value > best_score + 1.0e-12:
                best_score = value
                best_edges = [edge]
            elif abs(value - best_score) <= 1.0e-12:
                best_edges.append(edge)
        return self.random.choice(best_edges)

    def _predict(self, board: Board, to_play: int) -> Prediction:
        key = (board, to_play)
        prediction = self.prediction_cache.get(key)
        if prediction is None:
            started = time.perf_counter()
            prediction = self.network.predict(board, to_play)
            self.inference_calls += 1
            self.inference_positions += 1
            self.inference_seconds += time.perf_counter() - started
            if len(self.prediction_cache) >= self.prediction_cache_limit:
                self.prediction_cache.popitem(last=False)
            self.prediction_cache[key] = prediction
        else:
            self.prediction_cache_hits += 1
            self.prediction_cache.move_to_end(key)
        return prediction

    def _predict_batch(self, positions: list[tuple[Board, int]]) -> list[Prediction]:
        result: list[Prediction | None] = [None] * len(positions)
        missing: list[tuple[int, Board, int]] = []
        for index, (board, to_play) in enumerate(positions):
            cached = self.prediction_cache.get((board, to_play))
            if cached is None:
                missing.append((index, board, to_play))
            else:
                self.prediction_cache_hits += 1
                result[index] = cached
        if missing:
            predict_batch = getattr(self.network, "predict_batch", None)
            started = time.perf_counter()
            if not callable(predict_batch):
                predictions = [self._predict(board, to_play) for _, board, to_play in missing]
            else:
                predictions = predict_batch([(board, to_play) for _, board, to_play in missing])
                self.inference_calls += 1
                self.inference_positions += len(missing)
                self.inference_seconds += time.perf_counter() - started
            for (index, board, to_play), prediction in zip(missing, predictions):
                if len(self.prediction_cache) >= self.prediction_cache_limit:
                    self.prediction_cache.popitem(last=False)
                self.prediction_cache[(board, to_play)] = prediction
                result[index] = prediction
        for index, (board, to_play) in enumerate(positions):
            if result[index] is not None:
                self.prediction_cache.move_to_end((board, to_play))
        return [prediction for prediction in result if prediction is not None]

    def _add_root_exploration_noise(self, root: Node) -> None:
        if (
            not root.edges
            or self.root_dirichlet_alpha <= 0.0
            or self.root_exploration_fraction <= 0.0
        ):
            return
        edges = list(root.edges.values())
        noise = [self.random.gammavariate(self.root_dirichlet_alpha, 1.0) for _ in edges]
        total = sum(noise)
        if total <= 0.0:
            return
        fraction = self.root_exploration_fraction
        for edge, sample in zip(edges, noise):
            edge.prior = (1.0 - fraction) * edge.prior + fraction * (sample / total)

    def _forced_result(
        self,
        move: int,
        visits: int,
        value: float,
        legal: list[int],
        elapsed: float,
        reason: str,
    ) -> SearchResult:
        visit_policy = [0.0] * 81
        visit_policy[move] = 1.0
        candidate = SearchCandidate(move, visits, value, 1.0, 1.0)
        return SearchResult(
            move,
            visits,
            value,
            [(move, visits, value)],
            visit_policy,
            confidence=1.0,
            margin=1.0,
            elapsed=elapsed,
            top_candidates=[candidate],
            reason=reason,
        )
