from __future__ import annotations

import random

from catan_bots.base import Bot
from catan_engine.actions import Action
from catan_engine.observation import Observation


class RandomBot(Bot):
    name = "random"

    def choose_action(
        self,
        observation: Observation,
        legal_actions: list[Action],
        rng: random.Random,
    ) -> Action:
        return rng.choice(legal_actions)
