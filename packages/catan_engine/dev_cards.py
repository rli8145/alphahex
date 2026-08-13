from __future__ import annotations

import random
from enum import Enum


class DevCard(str, Enum):
    KNIGHT = "KNIGHT"
    VICTORY_POINT = "VICTORY_POINT"
    ROAD_BUILDING = "ROAD_BUILDING"
    YEAR_OF_PLENTY = "YEAR_OF_PLENTY"
    MONOPOLY = "MONOPOLY"


def default_dev_card_deck() -> list[DevCard]:
    return (
        [DevCard.KNIGHT] * 14
        + [DevCard.VICTORY_POINT] * 5
        + [DevCard.ROAD_BUILDING] * 2
        + [DevCard.YEAR_OF_PLENTY] * 2
        + [DevCard.MONOPOLY] * 2
    )


def shuffled_dev_card_deck(rng: random.Random) -> list[DevCard]:
    deck = default_dev_card_deck()
    rng.shuffle(deck)
    return deck


def parse_dev_card(value: DevCard | str) -> DevCard:
    if isinstance(value, DevCard):
        return value
    return DevCard[str(value)] if str(value) in DevCard.__members__ else DevCard(value)


def empty_dev_card_dict() -> dict[DevCard, int]:
    return {card: 0 for card in DevCard}


def normalize_dev_cards(values: dict[DevCard | str, int] | None = None) -> dict[DevCard, int]:
    cards = empty_dev_card_dict()
    if not values:
        return cards
    for card, amount in values.items():
        cards[parse_dev_card(card)] = int(amount)
    return cards


def dev_cards_to_json(values: dict[DevCard, int]) -> dict[str, int]:
    normalized = normalize_dev_cards(values)
    return {card.name: normalized[card] for card in DevCard}
