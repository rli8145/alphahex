from __future__ import annotations

import random

from catan_bots.base import Bot
from catan_bots.greedy_bot import _simulate_without_rng_drift
from catan_engine.actions import Action, ActionType, Phase
from catan_engine.observation import Observation
from catan_engine.resources import Resource
from catan_engine.rules import can_place_settlement, get_legal_actions, maritime_trade_ratio
from catan_engine.scoring import road_length, total_points
from catan_engine.state import GameState


class HeuristicBot(Bot):
    name = "heuristic"

    def choose_action(
        self,
        observation: Observation,
        legal_actions: list[Action],
        rng: random.Random,
    ) -> Action:
        player_id = observation.player_id
        state = observation.state
        if state.phase in {Phase.ROLL, Phase.STEAL}:
            return legal_actions[0]
        if state.phase == Phase.DISCARD:
            return _lowest_value_discard(legal_actions)
        if state.phase in {Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD, Phase.MOVE_ROBBER}:
            return self._best_action(player_id, state, legal_actions, rng)

        end_turn = next((action for action in legal_actions if action.type == ActionType.END_TURN), None)
        candidates = self._main_candidates(state, legal_actions, rng)
        baseline = evaluate_state(player_id, state)
        best_score, best_action = self._best_scored_action(player_id, state, candidates, rng)
        if end_turn is not None and best_score <= baseline + 0.05:
            return end_turn
        return best_action

    def _best_action(
        self,
        player_id: int,
        state: GameState,
        legal_actions: list[Action],
        rng: random.Random,
    ) -> Action:
        return self._best_scored_action(player_id, state, legal_actions, rng)[1]

    def _best_scored_action(
        self,
        player_id: int,
        state: GameState,
        legal_actions: list[Action],
        rng: random.Random,
    ) -> tuple[float, Action]:
        scored: list[tuple[float, Action]] = []
        for action in legal_actions:
            next_state = _simulate_without_rng_drift(state, action, rng)
            scored.append((evaluate_state(player_id, next_state), action))
        best_score = max(score for score, _action in scored)
        best_actions = [action for score, action in scored if score == best_score]
        return best_score, rng.choice(best_actions)

    def _main_candidates(
        self,
        state: GameState,
        legal_actions: list[Action],
        rng: random.Random,
    ) -> list[Action]:
        for action_type in (
            ActionType.BUILD_CITY,
            ActionType.BUILD_SETTLEMENT,
            ActionType.PLAY_KNIGHT,
            ActionType.PLAY_MONOPOLY,
            ActionType.PLAY_YEAR_OF_PLENTY,
            ActionType.PLAY_ROAD_BUILDING,
            ActionType.BUY_DEV_CARD,
            ActionType.BUILD_ROAD,
        ):
            typed = [action for action in legal_actions if action.type == action_type]
            if typed:
                return typed
        trades = _trades_that_enable_build(state, legal_actions, rng)
        if trades:
            return trades
        return legal_actions


def evaluate_state(player_id: int, state: GameState) -> float:
    player = state.player(player_id)
    opponent_id = state.other_player(player_id)
    opponent = state.player(opponent_id)

    vp_score = 12.0 * total_points(player_id, state)
    opponent_pressure = -9.0 * total_points(opponent_id, state)
    production = _production_value(player_id, state)
    ore_wheat = _ore_wheat_quality(player_id, state)
    diversity = 1.5 * sum(1 for amount in player.resources if amount > 0)
    port_access = _port_access_value(player_id, state)
    expansion = 0.7 * _expansion_spots(player_id, state)
    longest = 0.7 * road_length(player_id, state)
    army = 1.3 * player.played_knights - 1.0 * opponent.played_knights
    material = 0.25 * player.total_resources() - 0.15 * opponent.total_resources()
    return vp_score + opponent_pressure + production + ore_wheat + diversity + port_access + expansion + longest + army + material


def _production_value(player_id: int, state: GameState) -> float:
    value = 0.0
    player = state.player(player_id)
    number_weights = {
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        8: 5,
        9: 4,
        10: 3,
        11: 2,
        12: 1,
    }
    for node_id in player.settlements | player.cities:
        multiplier = 2 if node_id in player.cities else 1
        for hex_id in state.board.nodes[node_id].adjacent_hex_ids:
            hex_tile = state.board.hexes[hex_id]
            if hex_tile.resource is None or hex_tile.number is None:
                continue
            value += multiplier * number_weights.get(hex_tile.number, 0)
    return value


def _ore_wheat_quality(player_id: int, state: GameState) -> float:
    player = state.player(player_id)
    score = 0.0
    for node_id in player.settlements | player.cities:
        for hex_id in state.board.nodes[node_id].adjacent_hex_ids:
            hex_tile = state.board.hexes[hex_id]
            if hex_tile.resource == Resource.ORE:
                score += 2.0
            elif hex_tile.resource == Resource.GRAIN:
                score += 1.5
    score += 0.8 * player.resource_count(Resource.ORE)
    score += 0.6 * player.resource_count(Resource.GRAIN)
    return score


def _port_access_value(player_id: int, state: GameState) -> float:
    value = 0.0
    for resource in Resource:
        ratio = maritime_trade_ratio(state, player_id, resource)
        if ratio == 2:
            value += 2.0
        elif ratio == 3:
            value += 0.8
    return value


def _expansion_spots(player_id: int, state: GameState) -> int:
    return sum(
        1
        for node_id in state.board.nodes
        if can_place_settlement(state, node_id, setup=False, player_id=player_id)
    )


def _lowest_value_discard(legal_actions: list[Action]) -> Action:
    resource_order = {
        Resource.LUMBER.value: 0,
        Resource.BRICK.value: 1,
        Resource.WOOL.value: 2,
        Resource.GRAIN.value: 3,
        Resource.ORE.value: 4,
    }

    def score(action: Action) -> int:
        return sum(resource_order[resource] * amount for resource, amount in action.payload["resources"].items())

    return min(legal_actions, key=score)


def _trades_that_enable_build(
    state: GameState,
    legal_actions: list[Action],
    rng: random.Random,
) -> list[Action]:
    build_types = {
        ActionType.BUILD_CITY,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_ROAD,
        ActionType.BUY_DEV_CARD,
    }
    trades = [action for action in legal_actions if action.type == ActionType.MARITIME_TRADE]
    useful: list[Action] = []
    for action in trades:
        next_state = _simulate_without_rng_drift(state, action, rng)
        if any(legal.type in build_types for legal in get_legal_actions(next_state)):
            useful.append(action)
    return useful
