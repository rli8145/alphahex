from __future__ import annotations

from catan_engine.actions import Action, IllegalActionError
from catan_engine.rules import is_legal_action
from catan_engine.state import GameState


def validate_action(state: GameState, action: Action) -> None:
    if not is_legal_action(state, action):
        raise IllegalActionError(f"illegal action: {action}")
