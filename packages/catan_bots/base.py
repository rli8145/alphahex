from __future__ import annotations

import random
from abc import ABC, abstractmethod

from catan_engine.actions import Action


class Bot(ABC):
    name: str = "bot"

    @abstractmethod
    def choose_action(self, observation: dict, legal_actions: list[Action], rng: random.Random) -> Action:
        raise NotImplementedError


def create_bot(name: str) -> Bot:
    normalized = name.strip().lower()
    if normalized in {"mcts", "optimal"}:
        from catan_bots.mcts_bot import MCTSBot

        return MCTSBot()
    raise ValueError(f"unknown bot: {name}")


def available_bots() -> list[str]:
    return ["mcts"]
