from __future__ import annotations

import random
from dataclasses import dataclass

from catan_engine.state import GameState


@dataclass
class NormalDice:
    def roll(self, player_id: int, state: GameState, rng: random.Random) -> int:
        return rng.randint(1, 6) + rng.randint(1, 6)


@dataclass
class BalancedDice:
    recent_window: int = 12
    max_same_player_sevens: int = 2

    def roll(self, player_id: int, state: GameState, rng: random.Random) -> int:
        normal = rng.randint(1, 6) + rng.randint(1, 6)
        if normal != 7:
            if self._should_nudge_to_seven(player_id, state, rng):
                return 7
            return normal
        if self._should_replace_seven(player_id, state, rng):
            return self._non_seven_roll(rng)
        return 7

    def _recent_sevens(self, state: GameState) -> tuple[int, ...]:
        return state.dice_seven_history[-self.recent_window :]

    def _should_replace_seven(self, player_id: int, state: GameState, rng: random.Random) -> bool:
        recent = self._recent_sevens(state)
        same_player = sum(1 for owner in recent[-4:] if owner == player_id)
        ownership_gap = recent.count(player_id) - recent.count(1 - player_id)
        if same_player >= self.max_same_player_sevens:
            return rng.random() < 0.75
        if ownership_gap >= 2:
            return rng.random() < 0.45
        return False

    def _should_nudge_to_seven(self, player_id: int, state: GameState, rng: random.Random) -> bool:
        recent = self._recent_sevens(state)
        ownership_gap = recent.count(1 - player_id) - recent.count(player_id)
        if ownership_gap >= 3:
            return rng.random() < 0.03
        return False

    def _non_seven_roll(self, rng: random.Random) -> int:
        roll = 7
        while roll == 7:
            roll = rng.randint(1, 6) + rng.randint(1, 6)
        return roll


def dice_for_config(state: GameState) -> NormalDice | BalancedDice:
    return BalancedDice() if state.config.balanced_dice else NormalDice()
