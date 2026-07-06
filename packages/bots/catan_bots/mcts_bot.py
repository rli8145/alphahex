from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from catan_bots.base import Bot
from catan_bots.value_network import (
    DEFAULT_VALUE_NETWORK_PATH,
    ValueNetwork,
    action_policy_label,
    extract_state_features,
    load_value_network,
)
from catan_engine.actions import Action, ActionType, Phase
from catan_engine.resources import Resource
from catan_engine.rules import (
    CITY_COST,
    DEV_CARD_COST,
    ROAD_COST,
    SETTLEMENT_COST,
    apply_action,
    can_build_road,
    can_place_settlement,
    get_legal_actions,
    maritime_trade_ratio,
)
from catan_engine.scoring import calculate_longest_road, total_vp, visible_vp
from catan_engine.state import GameState


@dataclass(frozen=True)
class EvaluationWeights:
    own_vp: float = 18.0
    opponent_vp: float = -13.0
    own_resources: float = 0.35
    opponent_resources: float = -0.2
    resource_diversity: float = 1.2
    production: float = 1.0
    port: float = 1.0
    expansion: float = 0.7
    road_length: float = 0.9
    own_knights: float = 1.5
    opponent_knights: float = -1.1
    dev_cards: float = 0.7
    new_dev_cards: float = 0.35
    visible_vp_deficit: float = -2.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvaluationWeights":
        if not data:
            return cls()
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: float(value) for key, value in data.items() if key in valid})

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


DEFAULT_WEIGHTS = EvaluationWeights()
DEFAULT_WEIGHTS_PATH = Path(__file__).with_name("mcts_weights.json")
_PIP_WEIGHTS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
_RESOURCE_BASE_VALUE = {
    Resource.LUMBER: 1.05,
    Resource.BRICK: 1.05,
    Resource.WOOL: 0.9,
    Resource.GRAIN: 1.25,
    Resource.ORE: 1.35,
}
_SETUP_SECOND_SETTLEMENT_STEPS = {4, 6}


@dataclass(frozen=True)
class _SearchBudget:
    iterations: int
    rollout_depth: int
    branch_limit: int


def load_trained_weights(path: str | Path | None = None) -> EvaluationWeights:
    target = Path(path) if path is not None else DEFAULT_WEIGHTS_PATH
    if not target.exists():
        return DEFAULT_WEIGHTS
    try:
        import json

        data = json.loads(target.read_text(encoding="utf-8"))
        return EvaluationWeights.from_dict(data.get("weights", data))
    except (OSError, ValueError, TypeError):
        return DEFAULT_WEIGHTS


def save_trained_weights(
    weights: EvaluationWeights,
    path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    import json

    target = Path(path) if path is not None else DEFAULT_WEIGHTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"schema_version": 1, "bot": "mcts", "weights": weights.to_dict()}
    if metadata:
        payload["training"] = metadata
    temp_target = target.with_name(f".{target.name}.tmp")
    temp_target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_target.replace(target)
    return target


def _checkpoint_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


class MCTSBot(Bot):
    name = "mcts"

    def __init__(
        self,
        iterations: int = 4,
        rollout_depth: int = 6,
        exploration: float = 1.35,
        branch_limit: int = 6,
        weights: EvaluationWeights | None = None,
        value_network: ValueNetwork | None = None,
        use_value_network: bool = True,
        root_temperature: float = 0.0,
        temperature_turns: int | None = None,
        root_dirichlet_alpha: float = 0.3,
        root_dirichlet_epsilon: float = 0.0,
    ) -> None:
        self.iterations = iterations
        self.rollout_depth = rollout_depth
        self.exploration = exploration
        self.branch_limit = branch_limit
        self.root_temperature = max(0.0, root_temperature)
        self.temperature_turns = temperature_turns
        self.root_dirichlet_alpha = max(0.0, root_dirichlet_alpha)
        self.root_dirichlet_epsilon = max(0.0, min(1.0, root_dirichlet_epsilon))
        self.last_policy_target: str | None = None
        self.last_policy_distribution: dict[str, float] | None = None
        self.weights = weights or load_trained_weights()
        if value_network is not None:
            self.value_network = value_network
        elif use_value_network:
            self.value_network = load_value_network(require_serving_ready=True)
        else:
            self.value_network = None
        self._reload_value_network = value_network is None and use_value_network
        self._value_network_mtime = _checkpoint_mtime(DEFAULT_VALUE_NETWORK_PATH)

    def choose_action(self, observation: dict, legal_actions: list[Action], rng: random.Random) -> Action:
        self.last_policy_target = None
        self.last_policy_distribution = None
        self._refresh_value_network()
        if len(legal_actions) == 1:
            self._record_policy({action_policy_label(legal_actions[0]): 1.0})
            return legal_actions[0]

        state = observation.get("_state")
        if state is None and "state" in observation:
            state = GameState.from_dict(observation["state"])
        if state is None:
            action = rng.choice(legal_actions)
            self._record_policy({action_policy_label(action): 1.0})
            return action

        player_id = observation["player_id"]
        state = state.clone()
        state.action_log = []
        if state.phase == Phase.DISCARD:
            action = _least_valuable_discard(legal_actions)
            self._record_policy({action_policy_label(action): 1.0})
            return action
        if state.phase in {Phase.ROLL, Phase.STEAL}:
            self._record_policy({action_policy_label(legal_actions[0]): 1.0})
            return legal_actions[0]

        budget = _effective_budget(
            state,
            player_id,
            legal_actions,
            iterations=self.iterations,
            rollout_depth=self.rollout_depth,
            branch_limit=self.branch_limit,
        )
        root_candidates = _candidates_with_priors(
            state,
            player_id,
            legal_actions,
            budget.branch_limit,
            self.weights,
            self.value_network,
            rng,
        )
        root_candidates = _with_dirichlet_noise(
            root_candidates,
            self._active_root_dirichlet_epsilon(state),
            self.root_dirichlet_alpha,
            rng,
        )
        if len(root_candidates) == 1:
            self._record_policy({action_policy_label(root_candidates[0][0]): 1.0})
            return root_candidates[0][0]

        root = _Node(state=state.clone(), player_id=player_id, untried_actions=list(root_candidates))
        for _ in range(max(0, budget.iterations)):
            node = root
            search_state = state.clone()

            while not node.untried_actions and node.children:
                node = node.select_child(self.exploration, rng)
                search_state = node.state.clone()

            if node.untried_actions and search_state.phase != Phase.GAME_OVER:
                index = max(range(len(node.untried_actions)), key=lambda i: node.untried_actions[i][1])
                action, prior = node.untried_actions.pop(index)
                search_state = apply_action(search_state, action, rng)
                child_legal_actions = get_legal_actions(search_state)
                child_budget = _effective_budget(
                    search_state,
                    player_id,
                    child_legal_actions,
                    iterations=self.iterations,
                    rollout_depth=self.rollout_depth,
                    branch_limit=self.branch_limit,
                )
                child_candidates = _candidates_with_priors(
                    search_state,
                    player_id,
                    child_legal_actions,
                    child_budget.branch_limit,
                    self.weights,
                    self.value_network,
                    rng,
                )
                node = node.add_child(action, prior, search_state, child_candidates)

            reward = _rollout(
                search_state,
                player_id,
                budget.rollout_depth,
                budget.branch_limit,
                self.weights,
                self.value_network,
                rng,
            )
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        policy_distribution = _root_label_distribution(root.children)
        if policy_distribution is None:
            policy_distribution = _candidate_label_distribution(root_candidates)
        self._record_policy(policy_distribution)

        temperature = self._active_root_temperature(state)
        if temperature > 0.0:
            return _sample_root_action(root.children, root_candidates, temperature, rng)
        if root.children:
            return max(root.children, key=lambda child: (child.visits, child.value / max(1, child.visits))).action
        return max(root_candidates, key=lambda item: item[1])[0]

    def _refresh_value_network(self) -> None:
        if not self._reload_value_network:
            return
        mtime = _checkpoint_mtime(DEFAULT_VALUE_NETWORK_PATH)
        if mtime is None or mtime == self._value_network_mtime:
            return
        self.value_network = load_value_network(DEFAULT_VALUE_NETWORK_PATH, require_serving_ready=True)
        self._value_network_mtime = mtime

    def _active_root_temperature(self, state: GameState) -> float:
        if self.temperature_turns is not None and state.turn_number > self.temperature_turns:
            return 0.0
        return self.root_temperature

    def _active_root_dirichlet_epsilon(self, state: GameState) -> float:
        if self.temperature_turns is not None and state.turn_number > self.temperature_turns:
            return 0.0
        return self.root_dirichlet_epsilon

    def _record_policy(self, distribution: dict[str, float] | None) -> None:
        if not distribution:
            self.last_policy_distribution = None
            self.last_policy_target = None
            return
        self.last_policy_distribution = distribution
        self.last_policy_target = max(distribution.items(), key=lambda item: item[1])[0]


@dataclass
class _Node:
    state: GameState
    player_id: int
    action: Action | None = None
    prior: float = 1.0
    parent: _Node | None = None
    untried_actions: list[tuple[Action, float]] = field(default_factory=list)
    children: list[_Node] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0

    def add_child(self, action: Action, prior: float, state: GameState, candidates: list[tuple[Action, float]]) -> _Node:
        child = _Node(
            state=state.clone(),
            player_id=self.player_id,
            action=action,
            prior=prior,
            parent=self,
            untried_actions=list(candidates),
        )
        self.children.append(child)
        return child

    def select_child(self, exploration: float, rng: random.Random) -> _Node:
        # PUCT-style selection: the policy prior steers exploration instead of
        # only ordering candidate moves.
        sqrt_parent = math.sqrt(max(1, self.visits))
        best_score = -float("inf")
        best_children: list[_Node] = []
        for child in self.children:
            exploit = child.value / child.visits if child.visits else 0.5
            explore = exploration * child.prior * sqrt_parent / (1 + child.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        return rng.choice(best_children)


def _with_dirichlet_noise(
    candidates: list[tuple[Action, float]],
    epsilon: float,
    alpha: float,
    rng: random.Random,
) -> list[tuple[Action, float]]:
    if epsilon <= 0.0 or alpha <= 0.0 or len(candidates) <= 1:
        return candidates
    samples = [rng.gammavariate(alpha, 1.0) for _action, _prior in candidates]
    total = sum(samples)
    if total <= 0.0:
        return candidates
    noisy = [
        (action, (1.0 - epsilon) * max(0.0, prior) + epsilon * (sample / total))
        for (action, prior), sample in zip(candidates, samples, strict=True)
    ]
    return _normalize_action_weights(noisy)


def _normalize_action_weights(candidates: list[tuple[Action, float]]) -> list[tuple[Action, float]]:
    if not candidates:
        return []
    weights = [max(0.0, weight) for _action, weight in candidates]
    total = sum(weights)
    if total <= 0.0:
        uniform = 1.0 / len(candidates)
        return [(action, uniform) for action, _weight in candidates]
    return [(action, weight / total) for (action, _old_weight), weight in zip(candidates, weights, strict=True)]


def _root_label_distribution(children: list[_Node]) -> dict[str, float] | None:
    weighted_actions = [
        (child.action, float(child.visits))
        for child in children
        if child.action is not None and child.visits > 0
    ]
    if len(weighted_actions) <= 1:
        return None
    return _label_distribution(weighted_actions)


def _candidate_label_distribution(candidates: list[tuple[Action, float]]) -> dict[str, float] | None:
    if not candidates:
        return None
    return _label_distribution(candidates)


def _label_distribution(weighted_actions: list[tuple[Action, float]]) -> dict[str, float] | None:
    labels: dict[str, float] = {}
    total = 0.0
    for action, weight in weighted_actions:
        weight = max(0.0, weight)
        if weight <= 0.0:
            continue
        labels[action_policy_label(action)] = labels.get(action_policy_label(action), 0.0) + weight
        total += weight
    if total <= 0.0:
        actions = [action for action, _weight in weighted_actions]
        if not actions:
            return None
        uniform = 1.0 / len(actions)
        for action in actions:
            labels[action_policy_label(action)] = labels.get(action_policy_label(action), 0.0) + uniform
        return labels
    return {label: weight / total for label, weight in labels.items()}


def _sample_root_action(
    children: list[_Node],
    candidates: list[tuple[Action, float]],
    temperature: float,
    rng: random.Random,
) -> Action:
    visited_children = [child for child in children if child.action is not None and child.visits > 0]
    if len(visited_children) > 1:
        return _weighted_action_choice(
            [
                (child.action, float(child.visits) ** (1.0 / max(temperature, 1e-6)))
                for child in visited_children
                if child.action is not None
            ],
            rng,
        )
    return _weighted_action_choice(
        [
            (action, max(0.0, prior) ** (1.0 / max(temperature, 1e-6)))
            for action, prior in candidates
        ],
        rng,
    )


def _weighted_action_choice(weighted_actions: list[tuple[Action, float]], rng: random.Random) -> Action:
    if not weighted_actions:
        raise ValueError("cannot sample from an empty action list")
    total = sum(max(0.0, weight) for _action, weight in weighted_actions)
    if total <= 0.0:
        return rng.choice([action for action, _weight in weighted_actions])
    threshold = rng.random() * total
    cumulative = 0.0
    for action, weight in weighted_actions:
        cumulative += max(0.0, weight)
        if cumulative >= threshold:
            return action
    return weighted_actions[-1][0]


def _policy_probs(state: GameState, value_network: ValueNetwork | None) -> dict[str, float] | None:
    """Run the NN policy head once for a state; callers reuse it for every action."""
    if value_network is None:
        return None
    features = extract_state_features(state, state.current_player)
    return value_network.predict_policy(features)


def _action_prior(action: Action, policy_probs: dict[str, float] | None) -> float:
    if policy_probs is None:
        return 0.0
    exact = policy_probs.get(action_policy_label(action))
    if exact is not None:
        return exact
    return policy_probs.get(action.action_type.name, 0.0)


def _effective_budget(
    state: GameState,
    player_id: int,
    legal_actions: list[Action],
    *,
    iterations: int,
    rollout_depth: int,
    branch_limit: int,
) -> _SearchBudget:
    base_iterations = max(1, int(iterations))
    base_depth = max(1, int(rollout_depth))
    base_branch = max(1, int(branch_limit))

    if state.phase == Phase.SETUP_SETTLEMENT:
        return _SearchBudget(
            iterations=_raised_budget(base_iterations, multiplier=4.0, floor=12, cap=24),
            rollout_depth=_raised_budget(base_depth, multiplier=1.25, floor=7, cap=10),
            branch_limit=min(
                len(legal_actions),
                _raised_budget(base_branch, multiplier=2.0, floor=12, cap=16),
            ),
        )
    if state.phase == Phase.SETUP_ROAD:
        return _SearchBudget(
            iterations=_raised_budget(base_iterations, multiplier=3.0, floor=10, cap=18),
            rollout_depth=_raised_budget(base_depth, multiplier=1.15, floor=7, cap=9),
            branch_limit=min(
                len(legal_actions),
                _raised_budget(base_branch, multiplier=1.5, floor=6, cap=8),
            ),
        )
    if state.phase == Phase.MOVE_ROBBER:
        return _SearchBudget(
            iterations=_raised_budget(base_iterations, multiplier=3.0, floor=10, cap=18),
            rollout_depth=_raised_budget(base_depth, multiplier=1.15, floor=7, cap=9),
            branch_limit=min(
                len(legal_actions),
                _raised_budget(base_branch, multiplier=1.75, floor=10, cap=12),
            ),
        )
    if state.phase == Phase.MAIN and _is_near_win_or_block_turn(state, player_id, legal_actions):
        return _SearchBudget(
            iterations=_raised_budget(base_iterations, multiplier=3.0, floor=10, cap=18),
            rollout_depth=_raised_budget(base_depth, multiplier=1.25, floor=8, cap=10),
            branch_limit=min(
                len(legal_actions),
                _raised_budget(base_branch, multiplier=1.75, floor=10, cap=14),
            ),
        )

    return _SearchBudget(
        iterations=base_iterations,
        rollout_depth=base_depth,
        branch_limit=min(len(legal_actions), base_branch),
    )


def _raised_budget(base: int, *, multiplier: float, floor: int, cap: int) -> int:
    raised = max(base, floor, math.ceil(base * multiplier))
    if base <= cap:
        return min(raised, cap)
    return base


def _is_near_win_or_block_turn(state: GameState, player_id: int, legal_actions: list[Action]) -> bool:
    opponent_id = state.opponent_id(player_id)
    pressure_line = max(0, state.config.target_vp - 3)
    if total_vp(state, player_id) >= pressure_line:
        return True
    if visible_vp(state, opponent_id) >= pressure_line:
        return True
    vp_action_types = {ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT}
    return total_vp(state, player_id) >= state.config.target_vp - 4 and any(
        action.action_type in vp_action_types for action in legal_actions
    )


def _candidates_with_priors(
    state: GameState,
    player_id: int,
    legal_actions: list[Action],
    branch_limit: int,
    weights: EvaluationWeights,
    value_network: ValueNetwork | None,
    rng: random.Random,
) -> list[tuple[Action, float]]:
    policy_probs = _policy_probs(state, value_network) if len(legal_actions) > 1 else None
    actions = _candidate_actions(state, player_id, legal_actions, branch_limit, weights, value_network, rng, policy_probs)
    if policy_probs is None:
        return _heuristic_priors(state, player_id, actions)
    raw = [_action_prior(action, policy_probs) for action in actions]
    total = sum(raw)
    if total <= 0.0:
        return _heuristic_priors(state, player_id, actions)
    return [(action, prior / total) for action, prior in zip(actions, raw, strict=True)]


def _heuristic_priors(state: GameState, player_id: int, actions: list[Action]) -> list[tuple[Action, float]]:
    if not actions:
        return []
    scores = [_phase_action_score(state, player_id, action) for action in actions]
    if all(score == 0.0 for score in scores):
        uniform = 1.0 / len(actions)
        return [(action, uniform) for action in actions]
    lowest = min(scores)
    adjusted = [max(0.05, score - lowest + 0.05) for score in scores]
    total = sum(adjusted)
    return [(action, prior / total) for action, prior in zip(actions, adjusted, strict=True)]


def _phase_action_score(state: GameState, player_id: int, action: Action) -> float:
    if state.phase == Phase.SETUP_SETTLEMENT and action.action_type == ActionType.PLACE_SETTLEMENT:
        return _setup_settlement_score(state, player_id, int(action.payload["node_id"]))
    if state.phase == Phase.SETUP_ROAD and action.action_type == ActionType.PLACE_ROAD:
        return _setup_road_score(state, player_id, int(action.payload["edge_id"]))
    if state.phase == Phase.MOVE_ROBBER and action.action_type == ActionType.MOVE_ROBBER:
        return _robber_score(state, player_id, int(action.payload["hex_id"]))
    return 0.0


def _rollout(
    state: GameState,
    player_id: int,
    depth: int,
    branch_limit: int,
    weights: EvaluationWeights,
    value_network: ValueNetwork | None,
    rng: random.Random,
) -> float:
    current = state
    for _ in range(depth):
        if current.phase == Phase.GAME_OVER:
            break
        legal_actions = get_legal_actions(current)
        if not legal_actions:
            break
        if current.phase == Phase.DISCARD:
            action = _least_valuable_discard(legal_actions)
        elif current.phase in {Phase.ROLL, Phase.STEAL}:
            action = legal_actions[0]
        elif len(legal_actions) == 1:
            action = legal_actions[0]
        else:
            mover = current.current_player
            policy_probs = _policy_probs(current, value_network)
            candidates = _candidate_actions(current, mover, legal_actions, branch_limit, weights, value_network, rng, policy_probs)
            action_scores = _action_values(current, mover, candidates, weights, value_network, rng, policy_probs)
            action = max(zip(candidates, action_scores, strict=True), key=lambda item: item[1])[0]
        current = apply_action(current, action, rng)
    return _reward(current, player_id, weights, value_network)


def _reward(
    state: GameState,
    player_id: int,
    weights: EvaluationWeights,
    value_network: ValueNetwork | None,
) -> float:
    opponent_id = state.opponent_id(player_id)
    if state.winner == player_id:
        return 1.0
    if state.winner == opponent_id:
        return 0.0
    delta = _evaluate_state(state, player_id, weights) - _evaluate_state(state, opponent_id, weights)
    heuristic_reward = 1.0 / (1.0 + math.exp(-delta / 25.0))
    if value_network is None:
        return heuristic_reward
    return 0.7 * value_network.predict_state(state, player_id) + 0.3 * heuristic_reward


def _candidate_actions(
    state: GameState,
    player_id: int,
    legal_actions: list[Action],
    branch_limit: int,
    weights: EvaluationWeights,
    value_network: ValueNetwork | None,
    rng: random.Random,
    policy_probs: dict[str, float] | None = None,
) -> list[Action]:
    if state.phase == Phase.SETUP_SETTLEMENT:
        return sorted(
            legal_actions,
            key=lambda action: _setup_settlement_score(state, player_id, int(action.payload["node_id"])),
            reverse=True,
        )[:branch_limit]
    if state.phase == Phase.SETUP_ROAD:
        return sorted(
            legal_actions,
            key=lambda action: _setup_road_score(state, player_id, int(action.payload["edge_id"])),
            reverse=True,
        )[:branch_limit]
    if state.phase == Phase.MOVE_ROBBER:
        return sorted(
            legal_actions,
            key=lambda action: _robber_score(state, player_id, int(action.payload["hex_id"])),
            reverse=True,
        )[:branch_limit]

    if len(legal_actions) <= branch_limit:
        return legal_actions

    priority_types = (
        ActionType.BUILD_CITY,
        ActionType.BUILD_SETTLEMENT,
        ActionType.PLAY_KNIGHT,
        ActionType.PLAY_MONOPOLY,
        ActionType.PLAY_YEAR_OF_PLENTY,
        ActionType.PLAY_ROAD_BUILDING,
        ActionType.BUY_DEV_CARD,
        ActionType.BUILD_ROAD,
        ActionType.MARITIME_TRADE,
        ActionType.END_TURN,
    )
    ordered: list[Action] = []
    for action_type in priority_types:
        typed = [action for action in legal_actions if action.action_type == action_type]
        if not typed:
            continue
        typed_scores = _action_values(state, player_id, typed, weights, value_network, rng, policy_probs)
        typed = [action for action, _score in sorted(zip(typed, typed_scores, strict=True), key=lambda item: item[1], reverse=True)]
        ordered.extend(typed)
        if len(ordered) >= branch_limit:
            return ordered[:branch_limit]
    return legal_actions[:branch_limit]


def _action_values(
    state: GameState,
    player_id: int,
    actions: list[Action],
    weights: EvaluationWeights,
    value_network: ValueNetwork | None,
    rng: random.Random,
    policy_probs: dict[str, float] | None = None,
) -> list[float]:
    if value_network is None or len(actions) <= 1:
        return [_action_value(state, player_id, action, weights, value_network, rng, policy_probs) for action in actions]

    rng_state = rng.getstate()
    scores: list[float] = []
    next_states: list[GameState] = []
    next_indexes: list[int] = []
    for index, action in enumerate(actions):
        try:
            rng.setstate(rng_state)
            next_state = apply_action(state, action, rng)
        except Exception:
            scores.append(-float("inf"))
            continue
        finally:
            rng.setstate(rng_state)
        scores.append(
            _evaluate_state(next_state, player_id, weights)
            + _action_tactical_value(state, next_state, player_id, action)
            + 3.0 * _action_prior(action, policy_probs)
        )
        next_states.append(next_state)
        next_indexes.append(index)

    for index, prediction in zip(next_indexes, value_network.predict_states(next_states, player_id), strict=True):
        scores[index] += 8.0 * (prediction - 0.5)
    return scores


def _action_value(
    state: GameState,
    player_id: int,
    action: Action,
    weights: EvaluationWeights,
    value_network: ValueNetwork | None,
    rng: random.Random,
    policy_probs: dict[str, float] | None = None,
) -> float:
    rng_state = rng.getstate()
    try:
        next_state = apply_action(state, action, rng)
    except Exception:
        return -float("inf")
    finally:
        rng.setstate(rng_state)
    score = _evaluate_state(next_state, player_id, weights)
    score += _action_tactical_value(state, next_state, player_id, action)
    if value_network is not None:
        # The NN value head scores the resulting state, while the policy head
        # gives a small prior to exact actions self-play has favored before.
        # policy_probs is computed once per state by the caller.
        score += 8.0 * (value_network.predict_state(next_state, player_id) - 0.5)
        score += 3.0 * _action_prior(action, policy_probs)
    return score


def _actor_action_tactical_value(state: GameState, next_state: GameState, player_id: int, action: Action) -> float:
    actor_id = action.player_id
    action_type = action.action_type
    if actor_id < 0 or actor_id >= len(state.players):
        return 0.0

    if action_type == ActionType.PLACE_SETTLEMENT and state.phase == Phase.SETUP_SETTLEMENT:
        raw_score = _setup_settlement_score(state, actor_id, int(action.payload["node_id"]))
    elif action_type == ActionType.PLACE_ROAD and state.phase == Phase.SETUP_ROAD:
        raw_score = _setup_road_score(state, actor_id, int(action.payload["edge_id"]))
    elif action_type in {ActionType.PLACE_SETTLEMENT, ActionType.BUILD_SETTLEMENT}:
        raw_score = _settlement_tactical_value(state, actor_id, int(action.payload["node_id"]))
    elif action_type == ActionType.BUILD_CITY:
        raw_score = _city_tactical_value(state, actor_id, int(action.payload["node_id"]))
    elif action_type in {ActionType.PLACE_ROAD, ActionType.BUILD_ROAD, ActionType.PLAY_ROAD_BUILDING}:
        raw_score = _road_tactical_value(state, next_state, actor_id, action)
    elif action_type == ActionType.MOVE_ROBBER:
        raw_score = _robber_score(state, actor_id, int(action.payload["hex_id"]))
    elif action_type == ActionType.PLAY_KNIGHT:
        target = action.payload.get("target_player")
        raw_score = _robber_score(state, actor_id, int(action.payload["robber_hex_id"]), target)
    elif action_type == ActionType.STEAL_RESOURCE:
        raw_score = _expected_steal_value(state, actor_id, int(action.payload["target_player"]))
    elif action_type == ActionType.PLAY_MONOPOLY:
        raw_score = _monopoly_tactical_value(state, next_state, actor_id, action)
    elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        raw_score = _year_of_plenty_tactical_value(state, next_state, actor_id, action)
    elif action_type == ActionType.BUY_DEV_CARD:
        raw_score = _buy_dev_card_tactical_value(state, actor_id)
    elif action_type == ActionType.MARITIME_TRADE:
        raw_score = _trade_tactical_value(state, next_state, actor_id, action)
    else:
        raw_score = 0.0

    return raw_score if actor_id == player_id else -raw_score


def _settlement_tactical_value(state: GameState, player_id: int, node_id: int) -> float:
    node_pips = _node_pips_by_resource(state, int(node_id))
    total_pips = sum(node_pips.values())
    if total_pips <= 0.0 and state.board.nodes[int(node_id)].port is None:
        return 0.0

    production = _production_pips_by_resource(state, player_id)
    new_resources = sum(1 for resource in node_pips if production[resource] <= 0.0)
    diversity = len(node_pips)
    need_fit = sum(pips * _resource_need_value(state, player_id, resource) for resource, pips in node_pips.items())
    port_value = _port_tactical_value(state, player_id, int(node_id), node_pips)
    return 0.45 * total_pips + 0.75 * diversity + 1.1 * new_resources + 0.18 * need_fit + port_value


def _city_tactical_value(state: GameState, player_id: int, node_id: int) -> float:
    node_pips = _node_pips_by_resource(state, int(node_id))
    pip_gain = sum(node_pips.values())
    need_fit = sum(pips * _resource_need_value(state, player_id, resource) for resource, pips in node_pips.items())
    return 0.75 * pip_gain + 0.16 * need_fit


def _road_tactical_value(state: GameState, next_state: GameState, player_id: int, action: Action) -> float:
    edge_ids = _action_edge_ids(action)
    before_length = calculate_longest_road(state.board, state, player_id)
    after_length = calculate_longest_road(next_state.board, next_state, player_id)
    length_gain = max(0, after_length - before_length)

    award_gain = 0.0
    if state.longest_road_owner != player_id and next_state.longest_road_owner == player_id:
        award_gain += 4.0
        if state.longest_road_owner == state.opponent_id(player_id):
            award_gain += 2.0

    before_spots = set(_settlement_spots(state, player_id))
    after_spots = set(_settlement_spots(next_state, player_id))
    new_spots = after_spots - before_spots
    best_new_spot = max((_settlement_tactical_value(next_state, player_id, node_id) for node_id in new_spots), default=0.0)
    frontier_target = _best_road_frontier_target(next_state, player_id, edge_ids)
    return 1.15 * length_gain + award_gain + 0.4 * len(new_spots) + 0.35 * best_new_spot + 0.15 * frontier_target


def _monopoly_tactical_value(state: GameState, next_state: GameState, player_id: int, action: Action) -> float:
    resource = _resource_from_payload(action.payload["resource"])
    opponent = state.players[state.opponent_id(player_id)]
    expected_take = opponent.resources[resource]
    if expected_take <= 0:
        return 0.0
    resource_value = _resource_need_value(state, player_id, resource)
    return expected_take * (1.1 + resource_value) + 0.35 * _immediate_build_potential(next_state, player_id)


def _year_of_plenty_tactical_value(state: GameState, next_state: GameState, player_id: int, action: Action) -> float:
    resources = [_resource_from_payload(resource) for resource in action.payload.get("resources", [])[:2]]
    resource_fit = sum(_resource_need_value(state, player_id, resource) for resource in resources)
    return 0.8 * resource_fit + 0.85 * _immediate_build_potential(next_state, player_id)


def _buy_dev_card_tactical_value(state: GameState, player_id: int) -> float:
    player = state.players[player_id]
    score = 1.0
    if visible_vp(state, player_id) >= state.config.target_vp - 2:
        score += 0.8
    if state.largest_army_owner != player_id and player.played_knights >= 2:
        score += 0.7
    if not player.played_dev_card_this_turn:
        score += 0.25
    return score


def _trade_tactical_value(state: GameState, next_state: GameState, player_id: int, action: Action) -> float:
    give = _resource_from_payload(action.payload["give"])
    receive = _resource_from_payload(action.payload["receive"])
    give_count = int(action.payload.get("give_count", maritime_trade_ratio(state, player_id, give)))
    resource_delta = _resource_need_value(state, player_id, receive) - 0.35 * give_count * _resource_need_value(state, player_id, give)
    return resource_delta + 0.7 * _immediate_build_potential(next_state, player_id)


def _immediate_build_potential(state: GameState, player_id: int) -> float:
    player = state.players[player_id]
    score = 0.0

    if player.cities_remaining > 0 and player.settlements and player.has_resources(CITY_COST):
        best_city = max((_city_tactical_value(state, player_id, node_id) for node_id in player.settlements), default=0.0)
        score += 6.0 + 0.35 * best_city

    if player.settlements_remaining > 0 and player.has_resources(SETTLEMENT_COST):
        spots = _settlement_spots(state, player_id)
        if spots:
            best_settlement = max((_settlement_tactical_value(state, player_id, node_id) for node_id in spots), default=0.0)
            score += 4.0 + 0.25 * best_settlement

    if player.roads_remaining > 0 and player.has_resources(ROAD_COST):
        edges = _buildable_edges(state, player_id)
        if edges:
            score += 1.5 + 0.15 * _best_open_node_for_edges(state, player_id, edges)

    if state.dev_card_deck and player.has_resources(DEV_CARD_COST):
        score += 1.0

    return score


def _resource_need_value(state: GameState, player_id: int, resource: Resource) -> float:
    player = state.players[player_id]
    value = _RESOURCE_BASE_VALUE[resource]
    if _production_pips_by_resource(state, player_id)[resource] <= 0.0:
        value += 0.45

    for cost in (ROAD_COST, SETTLEMENT_COST, CITY_COST, DEV_CARD_COST):
        if resource not in cost or player.resources[resource] >= cost[resource]:
            continue
        missing_total = sum(max(0, amount - player.resources[cost_resource]) for cost_resource, amount in cost.items())
        if missing_total <= 2:
            value += 0.35
    return value


def _production_pips_by_resource(state: GameState, player_id: int) -> dict[Resource, float]:
    production = {resource: 0.0 for resource in Resource}
    player = state.players[player_id]
    for node_id in player.settlements | player.cities:
        multiplier = 2.0 if node_id in player.cities else 1.0
        for resource, pips in _node_pips_by_resource(state, node_id).items():
            production[resource] += multiplier * pips
    return production


def _node_pips_by_resource(state: GameState, node_id: int) -> dict[Resource, float]:
    pips: dict[Resource, float] = {}
    for hex_id in state.board.get_hexes_for_node(int(node_id)):
        hex_tile = state.board.hexes[hex_id]
        if hex_tile.number_token is None or hex_tile.hex_type.name not in Resource.__members__:
            continue
        resource = Resource[hex_tile.hex_type.name]
        pips[resource] = pips.get(resource, 0.0) + _PIP_WEIGHTS.get(hex_tile.number_token, 0)
    return pips


def _port_tactical_value(state: GameState, player_id: int, node_id: int, node_pips: dict[Resource, float]) -> float:
    port = state.board.nodes[int(node_id)].port
    if port is None:
        return 0.0

    production = _production_pips_by_resource(state, player_id)
    if port.kind == "generic":
        return 1.1 + 0.03 * (sum(production.values()) + sum(node_pips.values()))

    if port.resource is None:
        return 0.0
    resource_pips = production[port.resource] + node_pips.get(port.resource, 0.0)
    return 1.0 + 0.22 * resource_pips


def _settlement_spots(state: GameState, player_id: int) -> list[int]:
    return [
        node_id
        for node_id in state.board.nodes
        if can_place_settlement(state, node_id, setup=False, player_id=player_id)
    ]


def _buildable_edges(state: GameState, player_id: int) -> list[int]:
    return [
        edge_id
        for edge_id in state.board.edges
        if can_build_road(state, player_id, edge_id)
    ]


def _action_edge_ids(action: Action) -> list[int]:
    if action.action_type == ActionType.PLAY_ROAD_BUILDING:
        return [int(edge_id) for edge_id in action.payload.get("edge_ids", [])[:2]]
    edge_id = action.payload.get("edge_id")
    return [int(edge_id)] if edge_id is not None else []


def _best_road_frontier_target(state: GameState, player_id: int, edge_ids: list[int]) -> float:
    best = _best_open_node_for_edges(state, player_id, edge_ids)
    for edge_id in edge_ids:
        edge = state.board.edges[edge_id]
        for node_id in (edge.node_a, edge.node_b):
            for next_edge_id in state.board.get_edges_for_node(node_id):
                if not can_build_road(state, player_id, next_edge_id):
                    continue
                best = max(best, 0.8 * _best_open_node_for_edges(state, player_id, [next_edge_id]))
    return best


def _best_open_node_for_edges(state: GameState, player_id: int, edge_ids: list[int]) -> float:
    occupied_nodes = state.occupied_nodes()
    best = 0.0
    for edge_id in edge_ids:
        edge = state.board.edges[edge_id]
        for node_id in (edge.node_a, edge.node_b):
            if node_id in occupied_nodes:
                continue
            if any(adjacent in occupied_nodes for adjacent in state.board.get_adjacent_nodes(node_id)):
                continue
            best = max(best, _settlement_tactical_value(state, player_id, node_id))
    return best


def _resource_from_payload(value: Resource | str) -> Resource:
    if isinstance(value, Resource):
        return value
    resource_name = str(value)
    return Resource[resource_name] if resource_name in Resource.__members__ else Resource(resource_name)


def _evaluate_state(state: GameState, player_id: int, weights: EvaluationWeights = DEFAULT_WEIGHTS) -> float:
    player = state.players[player_id]
    opponent_id = state.opponent_id(player_id)
    opponent = state.players[opponent_id]
    score = weights.own_vp * total_vp(state, player_id)
    score += weights.opponent_vp * total_vp(state, opponent_id)
    score += weights.own_resources * player.total_resources()
    score += weights.opponent_resources * opponent.total_resources()
    score += weights.resource_diversity * sum(1 for amount in player.resources.values() if amount > 0)
    score += weights.production * _production_score(state, player_id)
    score += weights.port * _port_score(state, player_id)
    score += weights.expansion * _expansion_count(state, player_id)
    score += weights.road_length * calculate_longest_road(state.board, state, player_id)
    score += weights.own_knights * player.played_knights
    score += weights.opponent_knights * opponent.played_knights
    score += weights.dev_cards * sum(player.dev_cards.values())
    score += weights.new_dev_cards * sum(player.new_dev_cards.values())
    score += weights.visible_vp_deficit * max(0, visible_vp(state, opponent_id) - visible_vp(state, player_id))
    return score


def _action_tactical_value(state: GameState, next_state: GameState, player_id: int, action: Action) -> float:
    player = state.players[player_id]
    next_player = next_state.players[player_id]
    score = _actor_action_tactical_value(state, next_state, player_id, action)

    score += 20.0 * (total_vp(next_state, player_id) - total_vp(state, player_id))
    score += 0.25 * (next_player.total_resources() - player.total_resources())
    return score


def _setup_settlement_score(state: GameState, player_id: int, node_id: int) -> float:
    node_id = int(node_id)
    node_pips = _node_pips_by_resource(state, node_id)
    resource_counts = _node_resource_counts(state, node_id)
    total_pips = sum(node_pips.values())
    diversity = len(node_pips)
    score = 1.25 * total_pips + 1.6 * diversity

    score += sum(0.35 * _RESOURCE_BASE_VALUE[resource] * pips for resource, pips in node_pips.items())
    score += _setup_pair_bonus(node_pips)

    production = _production_pips_by_resource(state, player_id)
    if any(production.values()):
        missing_resources = [resource for resource, pips in node_pips.items() if pips > 0.0 and production[resource] <= 0.0]
        weak_resources = [resource for resource, pips in node_pips.items() if pips > 0.0 and production[resource] < 3.0]
        score += 1.7 * len(missing_resources)
        score += 0.25 * sum(node_pips[resource] for resource in missing_resources)
        score += 0.15 * sum(min(3.0, node_pips[resource]) for resource in weak_resources)
        score -= 0.08 * sum(max(0.0, production[resource] - 7.0) for resource in node_pips)

    if state.setup_step in _SETUP_SECOND_SETTLEMENT_STEPS:
        score += _starting_resource_score(resource_counts)

    return score + _setup_port_score(state, player_id, node_id, node_pips)


def _setup_road_score(state: GameState, player_id: int, edge_id: int) -> float:
    edge_id = int(edge_id)
    pending_node = state.pending_setup_node
    if pending_node is None or edge_id not in state.board.get_edges_for_node(pending_node):
        return _road_tactical_value(state, state, player_id, Action(ActionType.BUILD_ROAD, player_id, {"edge_id": edge_id}))

    next_node = state.board.get_opposite_node(edge_id, pending_node)
    occupied_edges = state.occupied_edges()
    continuation_edges = [
        next_edge_id
        for next_edge_id in state.board.get_edges_for_node(next_node)
        if next_edge_id != edge_id and next_edge_id not in occupied_edges
    ]
    score = 0.35 * len(continuation_edges)
    future_scores: list[float] = []
    for next_edge_id in continuation_edges:
        future_node = state.board.get_opposite_node(next_edge_id, next_node)
        if can_place_settlement(state, future_node, setup=True, player_id=player_id):
            future_scores.append(_setup_settlement_score(state, player_id, future_node))

    future_scores.sort(reverse=True)
    if future_scores:
        score += 0.38 * future_scores[0] + 0.12 * sum(future_scores[1:3]) + 0.55 * len(future_scores)
    else:
        score -= 1.5

    opponent_id = state.opponent_id(player_id)
    occupied_nodes = state.occupied_nodes()
    score -= 0.5 * sum(
        1
        for adjacent_node in state.board.get_adjacent_nodes(next_node)
        if occupied_nodes.get(adjacent_node) == opponent_id
    )
    return score


def _node_resource_counts(state: GameState, node_id: int) -> dict[Resource, int]:
    counts = {resource: 0 for resource in Resource}
    for hex_id in state.board.get_hexes_for_node(int(node_id)):
        hex_tile = state.board.hexes[hex_id]
        if hex_tile.hex_type.name in Resource.__members__:
            counts[Resource[hex_tile.hex_type.name]] += 1
    return counts


def _setup_pair_bonus(node_pips: dict[Resource, float]) -> float:
    score = 0.0
    score += 1.7 if node_pips.get(Resource.LUMBER, 0.0) and node_pips.get(Resource.BRICK, 0.0) else 0.0
    score += 1.4 if node_pips.get(Resource.GRAIN, 0.0) and node_pips.get(Resource.ORE, 0.0) else 0.0
    score += 0.8 if node_pips.get(Resource.GRAIN, 0.0) and node_pips.get(Resource.WOOL, 0.0) else 0.0
    score += 0.12 * min(node_pips.get(Resource.LUMBER, 0.0), node_pips.get(Resource.BRICK, 0.0))
    score += 0.10 * min(node_pips.get(Resource.GRAIN, 0.0), node_pips.get(Resource.ORE, 0.0))
    return score


def _starting_resource_score(resource_counts: dict[Resource, int]) -> float:
    score = sum(_RESOURCE_BASE_VALUE[resource] * count for resource, count in resource_counts.items())
    if resource_counts[Resource.LUMBER] and resource_counts[Resource.BRICK]:
        score += 2.2
    if resource_counts[Resource.GRAIN] and resource_counts[Resource.WOOL]:
        score += 1.0
    if resource_counts[Resource.GRAIN] and resource_counts[Resource.ORE]:
        score += 1.3
    return score


def _setup_port_score(
    state: GameState,
    player_id: int,
    node_id: int,
    node_pips: dict[Resource, float],
) -> float:
    port = state.board.nodes[int(node_id)].port
    if port is None:
        return 0.0

    production = _production_pips_by_resource(state, player_id)
    if port.kind == "generic":
        return 0.65 + 0.03 * (sum(production.values()) + sum(node_pips.values()))

    if port.resource is None:
        return 0.0
    resource_pips = production[port.resource] + node_pips.get(port.resource, 0.0)
    if resource_pips <= 0.0:
        return -0.35
    return 0.35 + min(2.0, 0.23 * resource_pips)


def _production_score(state: GameState, player_id: int) -> float:
    score = 0.0
    player = state.players[player_id]
    for node_id in player.settlements | player.cities:
        multiplier = 2 if node_id in player.cities else 1
        score += multiplier * _node_score(state, node_id)
        for hex_id in state.board.get_hexes_for_node(node_id):
            hex_tile = state.board.hexes[hex_id]
            if hex_tile.hex_type.name in {"ORE", "GRAIN"} and hex_tile.number_token is not None:
                score += multiplier * _PIP_WEIGHTS.get(hex_tile.number_token, 0) * 0.25
    return score


def _node_score(state: GameState, node_id: int) -> float:
    score = 0.0
    resources: set[str] = set()
    for hex_id in state.board.get_hexes_for_node(node_id):
        hex_tile = state.board.hexes[hex_id]
        if hex_tile.number_token is None:
            continue
        score += _PIP_WEIGHTS.get(hex_tile.number_token, 0)
        resources.add(hex_tile.hex_type.name)
    port = state.board.nodes[node_id].port
    if port is not None:
        score += 1.5 if port.kind == "generic" else 2.0
    return score + 0.5 * len(resources)


def _port_score(state: GameState, player_id: int) -> float:
    score = 0.0
    for resource in Resource:
        ratio = maritime_trade_ratio(state, player_id, resource)
        if ratio == 2:
            score += 2.0
        elif ratio == 3:
            score += 0.8
    return score


def _expansion_count(state: GameState, player_id: int) -> int:
    return len(_settlement_spots(state, player_id))


def _robber_score(state: GameState, player_id: int, hex_id: int, target_player: int | str | None = None) -> float:
    opponent_id = state.opponent_id(player_id)
    hex_tile = state.board.hexes[int(hex_id)]
    pips = _PIP_WEIGHTS.get(hex_tile.number_token or 0, 0)
    opponent_blocked = 0.0
    own_blocked = 0.0
    for node_id in state.board.get_nodes_for_hex(hex_id):
        if node_id in state.players[opponent_id].settlements:
            opponent_blocked += pips
        if node_id in state.players[opponent_id].cities:
            opponent_blocked += 2.0 * pips
        if node_id in state.players[player_id].settlements:
            own_blocked += pips
        if node_id in state.players[player_id].cities:
            own_blocked += 2.0 * pips
    return opponent_blocked - 0.85 * own_blocked + _robber_steal_potential(state, player_id, int(hex_id), target_player)


def _robber_steal_potential(state: GameState, player_id: int, hex_id: int, target_player: int | str | None = None) -> float:
    if target_player is not None:
        return _expected_steal_value(state, player_id, int(target_player))

    occupied_nodes = state.occupied_nodes()
    targets = {
        owner
        for node_id in state.board.get_nodes_for_hex(hex_id)
        if (owner := occupied_nodes.get(node_id)) is not None and owner != player_id
    }
    return max((_expected_steal_value(state, player_id, target) for target in targets), default=0.0)


def _expected_steal_value(state: GameState, player_id: int, target_player: int) -> float:
    if target_player == player_id:
        return 0.0
    target = state.players[target_player]
    hand_size = target.total_resources()
    if hand_size <= 0:
        return 0.0
    expected_resource = sum(
        amount * _resource_need_value(state, player_id, resource) / hand_size
        for resource, amount in target.resources.items()
    )
    return 1.0 + expected_resource + min(hand_size, 8) * 0.05


def _least_valuable_discard(legal_actions: list[Action]) -> Action:
    values = {"LUMBER": 0, "BRICK": 1, "WOOL": 2, "GRAIN": 3, "ORE": 4}
    return min(
        legal_actions,
        key=lambda action: sum(values[resource] * count for resource, count in action.payload["resources"].items()),
    )
