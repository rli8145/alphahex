"""Standalone 1v1 Catan-style engine."""

from catan_engine.actions import Action, ActionType, Phase
from catan_engine.resources import Resource
from catan_engine.dev_cards import DevCard
from catan_engine.rules import apply_action, get_legal_actions
from catan_engine.state import GameConfig, GameState, PlayerState, new_game

__all__ = [
    "Action",
    "ActionType",
    "DevCard",
    "GameConfig",
    "GameState",
    "Phase",
    "PlayerState",
    "Resource",
    "apply_action",
    "get_legal_actions",
    "new_game",
]
