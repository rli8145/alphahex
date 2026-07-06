from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from catan_engine.actions import Action, ActionType, Phase
from catan_engine.resources import ALL_RESOURCES, HEX_TO_RESOURCE, Resource, parse_resource
from catan_engine.rules import CITY_COST, SETTLEMENT_COST, can_build_road, can_place_settlement, maritime_trade_ratio
from catan_engine.scoring import calculate_longest_road, total_vp, visible_vp
from catan_engine.state import GameState

DEFAULT_VALUE_NETWORK_PATH = Path(__file__).with_name("mcts_value_network.json")
PHASES = tuple(Phase)
ACTION_TYPES = tuple(ActionType)
STANDARD_HEX_COUNT = 19
STANDARD_NODE_COUNT = 54
STANDARD_EDGE_COUNT = 72


def _build_policy_names() -> list[str]:
    labels = [action_type.name for action_type in ACTION_TYPES]
    for action_type in (ActionType.PLACE_SETTLEMENT, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
        labels.extend(f"{action_type.name}:node:{node_id}" for node_id in range(STANDARD_NODE_COUNT))
    for action_type in (ActionType.PLACE_ROAD, ActionType.BUILD_ROAD, ActionType.PLAY_ROAD_BUILDING):
        labels.extend(f"{action_type.name}:edge:{edge_id}" for edge_id in range(STANDARD_EDGE_COUNT))
    labels.extend(f"{ActionType.MOVE_ROBBER.name}:hex:{hex_id}" for hex_id in range(STANDARD_HEX_COUNT))
    for hex_id in range(STANDARD_HEX_COUNT):
        labels.append(f"{ActionType.PLAY_KNIGHT.name}:hex:{hex_id}:target:none")
        labels.extend(f"{ActionType.PLAY_KNIGHT.name}:hex:{hex_id}:target:{target}" for target in range(2))
    labels.extend(f"{ActionType.STEAL_RESOURCE.name}:target:{target}" for target in range(2))
    labels.extend(f"{ActionType.PLAY_MONOPOLY.name}:resource:{resource.name}" for resource in ALL_RESOURCES)
    for first in ALL_RESOURCES:
        for second in ALL_RESOURCES:
            labels.append(f"{ActionType.PLAY_YEAR_OF_PLENTY.name}:resources:{first.name},{second.name}")
    for give in ALL_RESOURCES:
        for receive in ALL_RESOURCES:
            if receive != give:
                labels.append(f"{ActionType.MARITIME_TRADE.name}:give:{give.name}:receive:{receive.name}")
    return labels


def action_policy_label(action: Action) -> str:
    action_type = action.action_type
    payload = action.payload
    if action_type in {ActionType.PLACE_SETTLEMENT, ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY}:
        node_id = _payload_int(payload, "node_id")
        if node_id is not None:
            return f"{action_type.name}:node:{node_id}"
    if action_type in {ActionType.PLACE_ROAD, ActionType.BUILD_ROAD}:
        edge_id = _payload_int(payload, "edge_id")
        if edge_id is not None:
            return f"{action_type.name}:edge:{edge_id}"
    if action_type == ActionType.PLAY_ROAD_BUILDING:
        edge_ids = payload.get("edge_ids", [])
        if isinstance(edge_ids, list) and edge_ids:
            try:
                return f"{action_type.name}:edge:{int(edge_ids[0])}"
            except (TypeError, ValueError):
                pass
    if action_type == ActionType.MOVE_ROBBER:
        hex_id = _payload_int(payload, "hex_id")
        if hex_id is not None:
            return f"{action_type.name}:hex:{hex_id}"
    if action_type == ActionType.PLAY_KNIGHT:
        hex_id = _payload_int(payload, "robber_hex_id", "hex_id")
        target = _payload_int(payload, "target_player")
        if hex_id is not None:
            target_label = str(target) if target is not None else "none"
            return f"{action_type.name}:hex:{hex_id}:target:{target_label}"
    if action_type == ActionType.STEAL_RESOURCE:
        target = _payload_int(payload, "target_player")
        if target is not None:
            return f"{action_type.name}:target:{target}"
    if action_type == ActionType.PLAY_MONOPOLY:
        resource = _payload_resource(payload, "resource")
        if resource is not None:
            return f"{action_type.name}:resource:{resource}"
    if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        resources = payload.get("resources", [])
        if isinstance(resources, list) and len(resources) >= 2:
            first = _resource_name(resources[0])
            second = _resource_name(resources[1])
            if first is not None and second is not None:
                return f"{action_type.name}:resources:{first},{second}"
    if action_type == ActionType.MARITIME_TRADE:
        give = _payload_resource(payload, "give")
        receive = _payload_resource(payload, "receive")
        if give is not None and receive is not None:
            return f"{action_type.name}:give:{give}:receive:{receive}"
    return action_type.name


def _payload_int(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _payload_resource(payload: dict[str, Any], key: str) -> str | None:
    return _resource_name(payload.get(key))


def _resource_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return parse_resource(value).name
    except (TypeError, ValueError):
        return None


POLICY_NAMES = _build_policy_names()
POLICY_INDEX = {name: index for index, name in enumerate(POLICY_NAMES)}
FEATURE_NAMES = [
    "is_current_player",
    "turn_number",
    "dice_roll",
    "own_total_vp",
    "opp_total_vp",
    "own_visible_vp",
    "opp_visible_vp",
    "own_resources_total",
    "opp_resources_total",
    "own_resource_diversity",
    "opp_resource_diversity",
    "own_settlements",
    "opp_settlements",
    "own_cities",
    "opp_cities",
    "own_roads",
    "opp_roads",
    "own_roads_remaining",
    "own_settlements_remaining",
    "own_cities_remaining",
    "own_longest_road_len",
    "opp_longest_road_len",
    "own_played_knights",
    "opp_played_knights",
    "own_dev_cards",
    "opp_dev_cards",
    "own_new_dev_cards",
    "opp_new_dev_cards",
    "owns_longest_road",
    "opp_owns_longest_road",
    "owns_largest_army",
    "opp_owns_largest_army",
    "own_production",
    "opp_production",
    "own_port_score",
    "opp_port_score",
    "own_expansion",
    "opp_expansion",
    "own_robber_exposure",
    "opp_robber_exposure",
]
FEATURE_NAMES.extend(f"own_prod_{resource.name.lower()}" for resource in Resource)
FEATURE_NAMES.extend(f"opp_prod_{resource.name.lower()}" for resource in Resource)
FEATURE_NAMES.extend(
    [
        "own_settlement_potential",
        "opp_settlement_potential",
        "own_city_potential",
        "opp_city_potential",
        "own_road_frontier",
        "opp_road_frontier",
        "own_longest_road_threat",
        "opp_longest_road_threat",
        "own_blocks_opp_expansion",
        "opp_blocks_own_expansion",
        "own_played_dev_this_turn",
        "opp_played_dev_this_turn",
        "dev_deck_remaining",
        "own_bank_pressure",
        "opp_bank_pressure",
        "own_port_fit",
        "opp_port_fit",
        "own_can_win_now",
        "opp_can_win_now",
        "own_best_settlement_spot",
        "opp_best_settlement_spot",
        "own_best_city_spot",
        "opp_best_city_spot",
        "own_best_road_target",
        "opp_best_road_target",
        "own_best_robber_gain",
        "opp_best_robber_gain",
    ]
)
FEATURE_NAMES.extend(f"phase_{phase.name}" for phase in PHASES)
FEATURE_NAMES.extend(f"own_{resource.name.lower()}" for resource in Resource)
FEATURE_NAMES.extend(f"opp_{resource.name.lower()}" for resource in Resource)

TrainingExample = tuple[list[float], float] | tuple[list[float], float, str | None]


@dataclass
class ValueNetwork:
    input_size: int
    hidden_size: int
    hidden_weights: np.ndarray  # (hidden, input)
    hidden_bias: np.ndarray  # (hidden,)
    output_weights: np.ndarray  # (hidden,)
    output_bias: float
    policy_weights: np.ndarray  # (policy, hidden)
    policy_bias: np.ndarray  # (policy,)

    @classmethod
    def create(cls, input_size: int, hidden_size: int, rng: random.Random) -> "ValueNetwork":
        # Weights are drawn from random.Random so seeded runs stay reproducible.
        scale = 1.0 / math.sqrt(max(1, input_size))
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            hidden_weights=np.array(
                [[rng.uniform(-scale, scale) for _ in range(input_size)] for _ in range(hidden_size)]
            ),
            hidden_bias=np.zeros(hidden_size),
            output_weights=np.array([rng.uniform(-scale, scale) for _ in range(hidden_size)]),
            output_bias=0.0,
            policy_weights=np.array(
                [[rng.uniform(-scale, scale) for _ in range(hidden_size)] for _ in POLICY_NAMES]
            ),
            policy_bias=np.zeros(len(POLICY_NAMES)),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValueNetwork":
        hidden_size = int(data["hidden_size"])
        return cls(
            input_size=int(data["input_size"]),
            hidden_size=hidden_size,
            hidden_weights=np.asarray(data["hidden_weights"], dtype=float),
            hidden_bias=np.asarray(data["hidden_bias"], dtype=float),
            output_weights=np.asarray(data["output_weights"], dtype=float),
            output_bias=float(data["output_bias"]),
            policy_weights=_normalize_policy_weights(data.get("policy_weights"), hidden_size),
            policy_bias=_normalize_policy_bias(data.get("policy_bias")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "feature_names": FEATURE_NAMES,
            "hidden_weights": self.hidden_weights.tolist(),
            "hidden_bias": self.hidden_bias.tolist(),
            "output_weights": self.output_weights.tolist(),
            "output_bias": self.output_bias,
            "policy_names": POLICY_NAMES,
            "policy_weights": self.policy_weights.tolist(),
            "policy_bias": self.policy_bias.tolist(),
        }

    def predict(self, features: list[float]) -> float:
        hidden = self._hidden(features)
        logit = float(self.output_weights @ hidden) + self.output_bias
        return _sigmoid(logit)

    def predict_state(self, state: GameState, player_id: int) -> float:
        return self.predict(extract_state_features(state, player_id))

    def predict_policy(self, features: list[float]) -> dict[str, float]:
        hidden = self._hidden(features)
        logits = self.policy_weights @ hidden + self.policy_bias
        probabilities = _softmax(logits)
        return dict(zip(POLICY_NAMES, probabilities.tolist(), strict=True))

    def train(
        self,
        examples: list[TrainingExample],
        *,
        epochs: int,
        learning_rate: float,
        l2: float,
        rng: random.Random,
        policy_loss_weight: float = 0.15,
    ) -> dict[str, float]:
        if not examples:
            return {"loss_before": 0.0, "loss_after": 0.0}
        loss_before = self.loss(examples, policy_loss_weight=policy_loss_weight)
        value_before, policy_before = self.loss_parts(examples)
        order = list(range(len(examples)))
        for _epoch in range(epochs):
            # Shuffle each epoch so adjacent positions from the same self-play game
            # do not push the weights in the exact same order every pass.
            rng.shuffle(order)
            for index in order:
                features, target, policy_target = _unpack_example(examples[index])
                self._train_one(
                    features,
                    target,
                    policy_target,
                    learning_rate=learning_rate,
                    l2=l2,
                    policy_loss_weight=policy_loss_weight,
                )
        value_after, policy_after = self.loss_parts(examples)
        return {
            "loss_before": round(loss_before, 6),
            "loss_after": round(self.loss(examples, policy_loss_weight=policy_loss_weight), 6),
            "value_loss_before": round(value_before, 6),
            "value_loss_after": round(value_after, 6),
            "policy_loss_before": round(policy_before, 6),
            "policy_loss_after": round(policy_after, 6),
        }

    def loss(self, examples: list[TrainingExample], *, policy_loss_weight: float = 0.15) -> float:
        value_loss, policy_loss = self.loss_parts(examples)
        return value_loss + policy_loss_weight * policy_loss

    def loss_parts(self, examples: list[TrainingExample]) -> tuple[float, float]:
        if not examples:
            return 0.0, 0.0
        value_total = 0.0
        policy_total = 0.0
        policy_count = 0
        for example in examples:
            features, target, policy_target = _unpack_example(example)
            hidden = self._hidden(features)
            output = _sigmoid(float(self.output_weights @ hidden) + self.output_bias)
            value_total += (output - target) ** 2
            if policy_target in POLICY_INDEX:
                policy_count += 1
                probabilities = _softmax(self.policy_weights @ hidden + self.policy_bias)
                policy_total += -math.log(max(1e-9, float(probabilities[POLICY_INDEX[policy_target]])))
        return value_total / len(examples), policy_total / policy_count if policy_count else 0.0

    def _hidden(self, features: list[float]) -> np.ndarray:
        if len(features) != self.input_size:
            raise ValueError(f"expected {self.input_size} features, got {len(features)}")
        return np.tanh(self.hidden_weights @ np.asarray(features, dtype=float) + self.hidden_bias)

    def _train_one(
        self,
        features: list[float],
        target: float,
        policy_target: str | None,
        *,
        learning_rate: float,
        l2: float,
        policy_loss_weight: float,
    ) -> None:
        inputs = np.asarray(features, dtype=float)
        hidden = np.tanh(self.hidden_weights @ inputs + self.hidden_bias)
        output = _sigmoid(float(self.output_weights @ hidden) + self.output_bias)

        # Backprop derivative for MSE(sigmoid(logit), target) with respect to
        # the value logit. This is the line where the value-head derivative is
        # computed before SGD applies it to the weights below.
        d_logit = (output - target) * output * (1.0 - output)
        one_minus_h2 = 1.0 - hidden * hidden
        d_hidden = d_logit * self.output_weights * one_minus_h2

        new_output_weights = self.output_weights - learning_rate * (d_logit * hidden + l2 * self.output_weights)

        if policy_target in POLICY_INDEX:
            # Policy-head gradient is probability - one_hot(label), scaled by
            # policy_loss_weight, applied to the shared hidden layer as well.
            probabilities = _softmax(self.policy_weights @ hidden + self.policy_bias)
            d_policy_logits = policy_loss_weight * probabilities
            d_policy_logits[POLICY_INDEX[policy_target]] -= policy_loss_weight
            d_hidden += (self.policy_weights.T @ d_policy_logits) * one_minus_h2
            self.policy_weights -= learning_rate * (np.outer(d_policy_logits, hidden) + l2 * self.policy_weights)
            self.policy_bias -= learning_rate * d_policy_logits

        self.output_weights = new_output_weights
        self.output_bias -= learning_rate * d_logit
        self.hidden_weights -= learning_rate * (np.outer(d_hidden, inputs) + l2 * self.hidden_weights)
        self.hidden_bias -= learning_rate * d_hidden


def load_value_network(path: str | Path | None = None, *, require_serving_ready: bool = False) -> ValueNetwork | None:
    target = Path(path) if path is not None else DEFAULT_VALUE_NETWORK_PATH
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if require_serving_ready and not _checkpoint_data_serving_ready(data):
            return None
        network = ValueNetwork.from_dict(data["network"])
        if network.input_size != len(FEATURE_NAMES):
            return None
        return network
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def checkpoint_serving_ready(path: str | Path | None = None) -> bool:
    target = Path(path) if path is not None else DEFAULT_VALUE_NETWORK_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    return _checkpoint_data_serving_ready(data)


def _checkpoint_data_serving_ready(data: dict[str, Any]) -> bool:
    training = data.get("training", {})
    return isinstance(training, dict) and bool(training.get("serving_ready"))


def save_value_network(
    network: ValueNetwork,
    path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    target = Path(path) if path is not None else DEFAULT_VALUE_NETWORK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "mcts_value_network",
        "network": network.to_dict(),
    }
    if metadata:
        payload["training"] = metadata
    temp_target = target.with_name(f".{target.name}.tmp")
    temp_target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_target.replace(target)
    return target


def extract_state_features(state: GameState, player_id: int) -> list[float]:
    opponent_id = state.opponent_id(player_id)
    player = state.players[player_id]
    opponent = state.players[opponent_id]
    target_vp = max(1, state.config.target_vp)

    # Expensive board sweeps are computed once here and reused by every
    # feature that needs them (potential counts, threat features, port fit).
    own_spots = _settlement_spots(state, player_id)
    opp_spots = _settlement_spots(state, opponent_id)
    own_edges = _buildable_edges(state, player_id)
    opp_edges = _buildable_edges(state, opponent_id)
    own_road_len = calculate_longest_road(state.board, state, player_id)
    opp_road_len = calculate_longest_road(state.board, state, opponent_id)
    own_prod = {resource: _production_by_resource(state, player_id, resource) for resource in Resource}
    opp_prod = {resource: _production_by_resource(state, opponent_id, resource) for resource in Resource}

    features = [
        1.0 if state.current_player == player_id else 0.0,
        _scale(state.turn_number, 300.0),
        _scale(state.dice_roll or 0, 12.0),
        _scale(total_vp(state, player_id), target_vp),
        _scale(total_vp(state, opponent_id), target_vp),
        _scale(visible_vp(state, player_id), target_vp),
        _scale(visible_vp(state, opponent_id), target_vp),
        _scale(player.total_resources(), 25.0),
        _scale(opponent.total_resources(), 25.0),
        _scale(sum(1 for amount in player.resources.values() if amount > 0), 5.0),
        _scale(sum(1 for amount in opponent.resources.values() if amount > 0), 5.0),
        _scale(len(player.settlements), 5.0),
        _scale(len(opponent.settlements), 5.0),
        _scale(len(player.cities), 4.0),
        _scale(len(opponent.cities), 4.0),
        _scale(len(player.roads), 15.0),
        _scale(len(opponent.roads), 15.0),
        _scale(player.roads_remaining, 15.0),
        _scale(player.settlements_remaining, 5.0),
        _scale(player.cities_remaining, 4.0),
        _scale(own_road_len, 15.0),
        _scale(opp_road_len, 15.0),
        _scale(player.played_knights, 10.0),
        _scale(opponent.played_knights, 10.0),
        _scale(sum(player.dev_cards.values()), 8.0),
        _scale(sum(opponent.dev_cards.values()), 8.0),
        _scale(sum(player.new_dev_cards.values()), 5.0),
        _scale(sum(opponent.new_dev_cards.values()), 5.0),
        1.0 if state.longest_road_owner == player_id else 0.0,
        1.0 if state.longest_road_owner == opponent_id else 0.0,
        1.0 if state.largest_army_owner == player_id else 0.0,
        1.0 if state.largest_army_owner == opponent_id else 0.0,
        _scale(_production_score(state, player_id), 50.0),
        _scale(_production_score(state, opponent_id), 50.0),
        _scale(_port_score(state, player_id), 10.0),
        _scale(_port_score(state, opponent_id), 10.0),
        _scale(len(own_spots), 30.0),
        _scale(len(opp_spots), 30.0),
        _scale(_robber_exposure(state, player_id), 4.0),
        _scale(_robber_exposure(state, opponent_id), 4.0),
    ]
    features.extend(_scale(own_prod[resource], 18.0) for resource in Resource)
    features.extend(_scale(opp_prod[resource], 18.0) for resource in Resource)
    features.extend(
        [
            _scale(len(own_spots), 12.0),
            _scale(len(opp_spots), 12.0),
            _scale(_city_potential(state, player_id), 5.0),
            _scale(_city_potential(state, opponent_id), 5.0),
            _scale(len(own_edges) if player.roads_remaining > 0 else 0, 15.0),
            _scale(len(opp_edges) if opponent.roads_remaining > 0 else 0, 15.0),
            _scale(max(0, own_road_len - 3), 12.0),
            _scale(max(0, opp_road_len - 3), 12.0),
            _scale(_blocking_pressure(state, player_id, opponent_id), 12.0),
            _scale(_blocking_pressure(state, opponent_id, player_id), 12.0),
            1.0 if player.played_dev_card_this_turn else 0.0,
            1.0 if opponent.played_dev_card_this_turn else 0.0,
            _scale(len(state.dev_card_deck), 25.0),
            _scale(max(0, player.total_resources() - state.config.discard_limit), 15.0),
            _scale(max(0, opponent.total_resources() - state.config.discard_limit), 15.0),
            _scale(_port_fit(state, player_id, own_prod), 10.0),
            _scale(_port_fit(state, opponent_id, opp_prod), 10.0),
            1.0 if _can_gain_vp_now(state, player_id, len(own_spots)) else 0.0,
            1.0 if _can_gain_vp_now(state, opponent_id, len(opp_spots)) else 0.0,
            _scale(_best_node_score(state, own_spots), 12.0),
            _scale(_best_node_score(state, opp_spots), 12.0),
            _scale(_best_node_score(state, player.settlements), 12.0),
            _scale(_best_node_score(state, opponent.settlements), 12.0),
            _scale(_best_road_target(state, own_edges), 12.0),
            _scale(_best_road_target(state, opp_edges), 12.0),
            _scale(_best_robber_gain(state, player_id), 15.0),
            _scale(_best_robber_gain(state, opponent_id), 15.0),
        ]
    )
    features.extend(1.0 if state.phase == phase else 0.0 for phase in PHASES)
    features.extend(_scale(player.resources[resource], 10.0) for resource in Resource)
    features.extend(_scale(opponent.resources[resource], 10.0) for resource in Resource)
    return features


def _production_score(state: GameState, player_id: int) -> float:
    player = state.players[player_id]
    return sum(
        (2.0 if node_id in player.cities else 1.0) * _node_score(state, node_id)
        for node_id in player.settlements | player.cities
    )


def _node_score(state: GameState, node_id: int) -> float:
    token_weights = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    score = 0.0
    resources: set[str] = set()
    for hex_id in state.board.get_hexes_for_node(node_id):
        hex_tile = state.board.hexes[hex_id]
        if hex_tile.number_token is None:
            continue
        score += token_weights.get(hex_tile.number_token, 0)
        resources.add(hex_tile.hex_type.name)
    port = state.board.nodes[node_id].port
    if port is not None:
        score += 1.5 if port.kind == "generic" else 2.0
    return score + 0.5 * len(resources)


def _production_by_resource(state: GameState, player_id: int, resource: Resource) -> float:
    player = state.players[player_id]
    token_weights = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    score = 0.0
    for node_id in player.settlements | player.cities:
        multiplier = 2.0 if node_id in player.cities else 1.0
        for hex_id in state.board.get_hexes_for_node(node_id):
            hex_tile = state.board.hexes[hex_id]
            if hex_id == state.board.robber_hex_id or hex_tile.number_token is None:
                continue
            if HEX_TO_RESOURCE.get(hex_tile.hex_type) == resource:
                score += multiplier * token_weights.get(hex_tile.number_token, 0)
    return score


def _settlement_spots(state: GameState, player_id: int) -> list[int]:
    return [
        node_id
        for node_id in state.board.nodes
        if can_place_settlement(state, node_id, setup=False, player_id=player_id)
    ]


def _city_potential(state: GameState, player_id: int) -> float:
    player = state.players[player_id]
    readiness = 1.0 if player.has_resources(CITY_COST) else 0.35
    return readiness * min(player.cities_remaining, len(player.settlements))


def _buildable_edges(state: GameState, player_id: int) -> list[int]:
    return [
        edge_id
        for edge_id in state.board.edges
        if can_build_road(state, player_id, edge_id)
    ]


def _best_node_score(state: GameState, node_ids) -> float:
    return max((_node_score(state, node_id) for node_id in node_ids), default=0.0)


def _best_road_target(state: GameState, edge_ids: list[int]) -> float:
    occupied = state.occupied_nodes()
    best = 0.0
    for edge_id in edge_ids:
        edge = state.board.edges[edge_id]
        for node_id in (edge.node_a, edge.node_b):
            if node_id in occupied:
                continue
            if any(adjacent in occupied for adjacent in state.board.get_adjacent_nodes(node_id)):
                continue
            best = max(best, _node_score(state, node_id))
    return best


def _best_robber_gain(state: GameState, player_id: int) -> float:
    token_weights = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    opponent = state.players[state.opponent_id(player_id)]
    player = state.players[player_id]
    best = 0.0
    for hex_id, hex_tile in state.board.hexes.items():
        if hex_id == state.board.robber_hex_id or hex_tile.number_token is None:
            continue
        buildings = 0
        for node_id in state.board.get_nodes_for_hex(hex_id):
            if node_id in opponent.cities:
                buildings += 2
            elif node_id in opponent.settlements:
                buildings += 1
            if node_id in player.cities:
                buildings -= 2
            elif node_id in player.settlements:
                buildings -= 1
        best = max(best, token_weights.get(hex_tile.number_token, 0) * buildings)
    return best


def _blocking_pressure(state: GameState, blocker_id: int, seeker_id: int) -> int:
    occupied_nodes = state.occupied_nodes()
    seeker = state.players[seeker_id]
    blocked = 0
    for blocker_node in state.players[blocker_id].settlements | state.players[blocker_id].cities:
        for adjacent_node in state.board.get_adjacent_nodes(blocker_node):
            if adjacent_node in occupied_nodes:
                continue
            has_seeker_road = any(edge_id in seeker.roads for edge_id in state.board.get_edges_for_node(adjacent_node))
            if has_seeker_road:
                blocked += 1
    return blocked


def _port_fit(state: GameState, player_id: int, production: dict[Resource, float]) -> float:
    score = 0.0
    for resource in Resource:
        ratio = maritime_trade_ratio(state, player_id, resource)
        if ratio >= 4:
            continue
        score += (4 - ratio) * (0.6 + production[resource] / 10.0)
    return score


def _can_gain_vp_now(state: GameState, player_id: int, settlement_spot_count: int) -> bool:
    player = state.players[player_id]
    if total_vp(state, player_id) < state.config.target_vp - 1:
        return False
    can_city = player.cities_remaining > 0 and player.settlements and player.has_resources(CITY_COST)
    can_settle = (
        player.settlements_remaining > 0
        and player.has_resources(SETTLEMENT_COST)
        and settlement_spot_count > 0
    )
    return can_city or can_settle


def _port_score(state: GameState, player_id: int) -> float:
    score = 0.0
    for resource in Resource:
        ratio = maritime_trade_ratio(state, player_id, resource)
        if ratio == 2:
            score += 2.0
        elif ratio == 3:
            score += 0.8
    return score


def _robber_exposure(state: GameState, player_id: int) -> int:
    player = state.players[player_id]
    return sum(
        2 if node_id in player.cities else 1
        for node_id in state.board.get_nodes_for_hex(state.board.robber_hex_id)
        if node_id in player.settlements or node_id in player.cities
    )


def _scale(value: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return max(-1.0, min(1.0, float(value) / denominator))


def _unpack_example(example: TrainingExample) -> tuple[list[float], float, str | None]:
    if len(example) == 2:
        features, target = example
        return features, float(target), None
    features, target, policy_target = example
    return features, float(target), policy_target


def _normalize_policy_weights(values: Any, hidden_size: int) -> np.ndarray:
    try:
        weights = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        weights = None
    if weights is None or weights.shape != (len(POLICY_NAMES), hidden_size):
        return np.zeros((len(POLICY_NAMES), hidden_size))
    return weights


def _normalize_policy_bias(values: Any) -> np.ndarray:
    try:
        bias = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        bias = None
    if bias is None or bias.shape != (len(POLICY_NAMES),):
        return np.zeros(len(POLICY_NAMES))
    return bias


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    exps = np.exp(logits - np.max(logits))
    total = exps.sum()
    if total <= 0:
        return np.full(logits.shape, 1.0 / logits.size)
    return exps / total
