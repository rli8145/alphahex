from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    SETUP_SETTLEMENT = "setup_settlement"
    SETUP_ROAD = "setup_road"
    ROLL = "roll"
    DISCARD = "discard"
    MOVE_ROBBER = "move_robber"
    STEAL = "steal"
    MAIN = "main"
    GAME_OVER = "game_over"


class ActionType(str, Enum):
    PLACE_SETTLEMENT = "place_settlement"
    PLACE_ROAD = "place_road"
    ROLL_DICE = "roll_dice"
    DISCARD = "discard"
    MOVE_ROBBER = "move_robber"
    STEAL_RESOURCE = "steal_resource"
    BUILD_ROAD = "build_road"
    BUILD_SETTLEMENT = "build_settlement"
    BUILD_CITY = "build_city"
    BUY_DEV_CARD = "buy_dev_card"
    PLAY_KNIGHT = "play_knight"
    PLAY_ROAD_BUILDING = "play_road_building"
    PLAY_YEAR_OF_PLENTY = "play_year_of_plenty"
    PLAY_MONOPOLY = "play_monopoly"
    MARITIME_TRADE = "maritime_trade"
    END_TURN = "end_turn"


class IllegalActionError(ValueError):
    pass


@dataclass(frozen=True)
class Action:
    type: ActionType
    player_id: int
    payload: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "player_id": self.player_id,
            "payload": _encode_value(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        return cls(
            type=ActionType(data["type"]),
            player_id=int(data["player_id"]),
            payload=dict(data.get("payload", {})),
        )


def _encode_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(_encode_value(key)): _encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    return value


def normalized_payload(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return tuple(sorted((str(normalized_payload(key)), normalized_payload(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(normalized_payload(item) for item in value)
    return value


def action_key(action: Action) -> tuple[str, int, Any]:
    return (action.type.value, action.player_id, normalized_payload(action.payload))
