from __future__ import annotations

import random
from abc import ABC, abstractmethod

from catan_engine.actions import Action
from catan_engine.observation import Observation


class Bot(ABC):
    name = "bot"

    @abstractmethod
    def choose_action(
        self,
        observation: Observation,
        legal_actions: list[Action],
        rng: random.Random,
    ) -> Action:
        raise NotImplementedError


def create_bot(name: str) -> Bot:
    normalized = name.strip().lower()
    if normalized == "random":
        from catan_bots.random_bot import RandomBot

        return RandomBot()
    if normalized == "greedy":
        from catan_bots.greedy_bot import GreedyBot

        return GreedyBot()
    if normalized == "heuristic":
        from catan_bots.heuristic_bot import HeuristicBot

        return HeuristicBot()
    raise ValueError(f"unknown bot '{name}'")
