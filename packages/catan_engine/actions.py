from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    SETUP_SETTLEMENT = "SETUP_SETTLEMENT"
    SETUP_ROAD = "SETUP_ROAD"
    ROLL = "ROLL"
    DISCARD = "DISCARD"
    MOVE_ROBBER = "MOVE_ROBBER"
    STEAL = "STEAL"
    MAIN = "MAIN"
    GAME_OVER = "GAME_OVER"


class ActionType(str, Enum):
    PLACE_SETTLEMENT = "PLACE_SETTLEMENT"
    PLACE_ROAD = "PLACE_ROAD"
    ROLL_DICE = "ROLL_DICE"
    DISCARD = "DISCARD"
    MOVE_ROBBER = "MOVE_ROBBER"
    STEAL_RESOURCE = "STEAL_RESOURCE"
    BUILD_ROAD = "BUILD_ROAD"
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"
    BUILD_CITY = "BUILD_CITY"
    BUY_DEV_CARD = "BUY_DEV_CARD"
    PLAY_KNIGHT = "PLAY_KNIGHT"
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"
    PLAY_MONOPOLY = "PLAY_MONOPOLY"
    MARITIME_TRADE = "MARITIME_TRADE"
    END_TURN = "END_TURN"


class IllegalActionError(ValueError):
    pass


@dataclass
class Action:
    action_type: ActionType
    player_id: int
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> ActionType:
        return self.action_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.name,
            "player_id": self.player_id,
            "payload": _to_jsonable(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        action_value = data.get("action_type", data.get("type"))
        return cls(
            action_type=ActionType[str(action_value)] if str(action_value) in ActionType.__members__ else ActionType(action_value),
            player_id=int(data["player_id"]),
            payload=dict(data.get("payload", {})),
        )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(_to_jsonable(key)): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return value


def action_key(action: Action) -> tuple[str, int, Any]:
    return (action.action_type.name, action.player_id, _normalized(action.payload))


def _normalized(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return tuple(sorted((str(_normalized(key)), _normalized(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_normalized(item) for item in value)
    return value
