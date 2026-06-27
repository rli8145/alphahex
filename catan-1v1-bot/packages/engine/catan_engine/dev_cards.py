from __future__ import annotations

import random
from enum import Enum


class DevCard(str, Enum):
    KNIGHT = "knight"
    VICTORY_POINT = "victory_point"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"


def default_dev_deck() -> tuple[DevCard, ...]:
    return (
        (DevCard.KNIGHT,) * 14
        + (DevCard.VICTORY_POINT,) * 5
        + (DevCard.ROAD_BUILDING,) * 2
        + (DevCard.YEAR_OF_PLENTY,) * 2
        + (DevCard.MONOPOLY,) * 2
    )


def shuffled_dev_deck(rng: random.Random) -> tuple[DevCard, ...]:
    deck = list(default_dev_deck())
    rng.shuffle(deck)
    return tuple(deck)


def parse_dev_card(value: DevCard | str) -> DevCard:
    if isinstance(value, DevCard):
        return value
    return DevCard(value)
