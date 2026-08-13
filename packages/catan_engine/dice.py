from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class NormalDice:
    def roll(self, player_id: int, rng: random.Random) -> int:
        return rng.randint(1, 6) + rng.randint(1, 6)


@dataclass
class BalancedDice:
    """MVP stub with a light anti-streak nudge for repeated 7 ownership."""

    def roll(self, player_id: int, rng: random.Random, seven_history: list[int] | None = None) -> int:
        value = rng.randint(1, 6) + rng.randint(1, 6)
        history = seven_history or []
        if value == 7 and history[-3:].count(player_id) >= 2 and rng.random() < 0.6:
            while value == 7:
                value = rng.randint(1, 6) + rng.randint(1, 6)
        return value


def roll_dice(player_id: int, rng: random.Random, *, balanced: bool, seven_history: list[int]) -> int:
    if balanced:
        return BalancedDice().roll(player_id, rng, seven_history)
    return NormalDice().roll(player_id, rng)
