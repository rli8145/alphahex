from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from catan_bots.base import Bot
from catan_bots.value_network import DEFAULT_VALUE_NETWORK_PATH, ValueNetwork, extract_state_features, load_value_network
from catan_engine.actions import Action, ActionType, Phase
from catan_engine.resources import Resource
from catan_engine.rules import apply_action, can_place_settlement, get_legal_actions, maritime_trade_ratio
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
    ) -> None:
        self.iterations = iterations
        self.rollout_depth = rollout_depth
        self.exploration = exploration
        self.branch_limit = branch_limit
        self.weights = weights or load_trained_weights()
        self.value_network = value_network if value_network is not None else load_value_network() if use_value_network else None
        self._reload_value_network = value_network is None and use_value_network
        self._value_network_mtime = _checkpoint_mtime(DEFAULT_VALUE_NETWORK_PATH)

    def choose_action(self, observation: dict, legal_actions: list[Action], rng: random.Random) -> Action:
        self._refresh_value_network()
        if len(legal_actions) == 1:
            return legal_actions[0]

        state = observation.get("_state")
        if state is None and "state" in observation:
            state = GameState.from_dict(observation["state"])
        if state is None:
            return rng.choice(legal_actions)

        player_id = observation["player_id"]
        state = state.clone()
        state.action_log = []
        if state.phase == Phase.DISCARD:
            return _least_valuable_discard(legal_actions)
        if state.phase in {Phase.ROLL, Phase.STEAL}:
            return legal_actions[0]

        root_actions = _candidate_actions(state, player_id, legal_actions, self.branch_limit, self.weights, self.value_network, rng)
        if len(root_actions) == 1:
            return root_actions[0]

        root = _Node(state=state.clone(), player_id=player_id, untried_actions=list(root_actions))
        for _ in range(self.iterations):
            node = root
            search_state = state.clone()

            while not node.untried_actions and node.children:
                node = node.select_child(self.exploration, rng)
                search_state = node.state.clone()

            if node.untried_actions and search_state.phase != Phase.GAME_OVER:
                action = node.untried_actions.pop(rng.randrange(len(node.untried_actions)))
                search_state = apply_action(search_state, action, rng)
                child_actions = _candidate_actions(
                    search_state,
                    player_id,
                    get_legal_actions(search_state),
                    self.branch_limit,
                    self.weights,
                    self.value_network,
                    rng,
                )
                node = node.add_child(action, search_state, child_actions)

            reward = _rollout(search_state, player_id, self.rollout_depth, self.branch_limit, self.weights, self.value_network, rng)
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        return max(root.children, key=lambda child: (child.visits, child.value / max(1, child.visits))).action

    def _refresh_value_network(self) -> None:
        if not self._reload_value_network:
            return
        mtime = _checkpoint_mtime(DEFAULT_VALUE_NETWORK_PATH)
        if mtime is None or mtime == self._value_network_mtime:
            return
        loaded = load_value_network(DEFAULT_VALUE_NETWORK_PATH)
        if loaded is not None:
            self.value_network = loaded
            self._value_network_mtime = mtime


@dataclass
class _Node:
    state: GameState
    player_id: int
    action: Action | None = None
    parent: _Node | None = None
    untried_actions: list[Action] = field(default_factory=list)
    children: list[_Node] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0

    def add_child(self, action: Action, state: GameState, actions: list[Action]) -> _Node:
        child = _Node(
            state=state.clone(),
            player_id=self.player_id,
            action=action,
            parent=self,
            untried_actions=list(actions),
        )
        self.children.append(child)
        return child

    def select_child(self, exploration: float, rng: random.Random) -> _Node:
        log_parent = math.log(max(1, self.visits))
        best_score = -float("inf")
        best_children: list[_Node] = []
        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.value / child.visits
                explore = exploration * math.sqrt(log_parent / child.visits)
                score = exploit + explore
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        return rng.choice(best_children)


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
        else:
            candidates = _candidate_actions(current, current.current_player, legal_actions, branch_limit, weights, value_network, rng)
            action = max(candidates, key=lambda item: _action_value(current, current.current_player, item, weights, value_network, rng))
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
) -> list[Action]:
    if len(legal_actions) <= branch_limit:
        return legal_actions

    if state.phase == Phase.SETUP_SETTLEMENT:
        return sorted(legal_actions, key=lambda action: _node_score(state, action.payload["node_id"]), reverse=True)[:branch_limit]
    if state.phase == Phase.MOVE_ROBBER:
        return sorted(legal_actions, key=lambda action: _robber_score(state, player_id, action.payload["hex_id"]), reverse=True)[:branch_limit]

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
        typed.sort(key=lambda action: _action_value(state, player_id, action, weights, value_network, rng), reverse=True)
        ordered.extend(typed)
        if len(ordered) >= branch_limit:
            return ordered[:branch_limit]
    return legal_actions[:branch_limit]


def _action_value(
    state: GameState,
    player_id: int,
    action: Action,
    weights: EvaluationWeights,
    value_network: ValueNetwork | None,
    rng: random.Random,
) -> float:
    rng_state = rng.getstate()
    try:
        next_state = apply_action(state, action, rng)
    except Exception:
        return -float("inf")
    finally:
        rng.setstate(rng_state)
    score = _evaluate_state(next_state, player_id, weights)
    if value_network is not None:
        # The NN value head scores the resulting state, while the policy head
        # gives a small prior to exact actions self-play has favored before.
        score += 8.0 * (value_network.predict_state(next_state, player_id) - 0.5)
        score += 3.0 * value_network.action_prior(extract_state_features(state, player_id), action)
    return score


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


def _production_score(state: GameState, player_id: int) -> float:
    weights = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    score = 0.0
    player = state.players[player_id]
    for node_id in player.settlements | player.cities:
        multiplier = 2 if node_id in player.cities else 1
        score += multiplier * _node_score(state, node_id)
        for hex_id in state.board.get_hexes_for_node(node_id):
            hex_tile = state.board.hexes[hex_id]
            if hex_tile.hex_type.name in {"ORE", "GRAIN"} and hex_tile.number_token is not None:
                score += multiplier * weights.get(hex_tile.number_token, 0) * 0.25
    return score


def _node_score(state: GameState, node_id: int) -> float:
    weights = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    score = 0.0
    resources: set[str] = set()
    for hex_id in state.board.get_hexes_for_node(node_id):
        hex_tile = state.board.hexes[hex_id]
        if hex_tile.number_token is None:
            continue
        score += weights.get(hex_tile.number_token, 0)
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
    return sum(
        1
        for node_id in state.board.nodes
        if can_place_settlement(state, node_id, setup=False, player_id=player_id)
    )


def _robber_score(state: GameState, player_id: int, hex_id: int) -> float:
    opponent_id = state.opponent_id(player_id)
    score = 0.0
    for node_id in state.board.get_nodes_for_hex(hex_id):
        if node_id in state.players[opponent_id].settlements:
            score += 2.0
        if node_id in state.players[opponent_id].cities:
            score += 4.0
        if node_id in state.players[player_id].settlements:
            score -= 2.0
        if node_id in state.players[player_id].cities:
            score -= 4.0
    return score


def _least_valuable_discard(legal_actions: list[Action]) -> Action:
    values = {"LUMBER": 0, "BRICK": 1, "WOOL": 2, "GRAIN": 3, "ORE": 4}
    return min(
        legal_actions,
        key=lambda action: sum(values[resource] * count for resource, count in action.payload["resources"].items()),
    )
