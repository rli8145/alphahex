from __future__ import annotations

import random

from catan_bots.base import Bot
from catan_engine.actions import Action, ActionType, Phase
from catan_engine.observation import Observation
from catan_engine.rules import apply_action, get_legal_actions
from catan_engine.scoring import total_points


class GreedyBot(Bot):
    name = "greedy"

    def choose_action(
        self,
        observation: Observation,
        legal_actions: list[Action],
        rng: random.Random,
    ) -> Action:
        player_id = observation.player_id
        state = observation.state

        winning = self._first_immediate_win(state, player_id, legal_actions, rng)
        if winning is not None:
            return winning

        blocking = self._first_obvious_block(state, player_id, legal_actions)
        if blocking is not None:
            return blocking

        priority = [
            ActionType.BUILD_CITY,
            ActionType.BUILD_SETTLEMENT,
            ActionType.BUY_DEV_CARD,
            ActionType.BUILD_ROAD,
            ActionType.MARITIME_TRADE,
            ActionType.END_TURN,
        ]
        for action_type in priority:
            typed = [action for action in legal_actions if action.type == action_type]
            if not typed:
                continue
            if action_type == ActionType.MARITIME_TRADE:
                enabling = self._trade_enables_build(state, typed, rng)
                if enabling is not None:
                    return enabling
                continue
            return typed[0]
        return rng.choice(legal_actions)

    def _first_immediate_win(
        self,
        state,
        player_id: int,
        legal_actions: list[Action],
        rng: random.Random,
    ) -> Action | None:
        for action in legal_actions:
            next_state = _simulate_without_rng_drift(state, action, rng)
            if next_state.phase == Phase.GAME_OVER and next_state.winner == player_id:
                return action
            if total_points(player_id, next_state) >= next_state.config.target_vp:
                return action
        return None

    def _first_obvious_block(
        self,
        state,
        player_id: int,
        legal_actions: list[Action],
    ) -> Action | None:
        opponent_id = state.other_player(player_id)
        if total_points(opponent_id, state) < state.config.target_vp - 1:
            return None
        for action in legal_actions:
            if action.type in {ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT}:
                return action
        road_actions = [action for action in legal_actions if action.type == ActionType.BUILD_ROAD]
        return road_actions[0] if road_actions else None

    def _trade_enables_build(
        self,
        state,
        trade_actions: list[Action],
        rng: random.Random,
    ) -> Action | None:
        build_types = {
            ActionType.BUILD_CITY,
            ActionType.BUILD_SETTLEMENT,
            ActionType.BUILD_ROAD,
            ActionType.BUY_DEV_CARD,
        }
        for action in trade_actions:
            next_state = _simulate_without_rng_drift(state, action, rng)
            if any(legal.type in build_types for legal in get_legal_actions(next_state)):
                return action
        return None


def _simulate_without_rng_drift(state, action: Action, rng: random.Random):
    rng_state = rng.getstate()
    try:
        return apply_action(state, action, rng)
    finally:
        rng.setstate(rng_state)
