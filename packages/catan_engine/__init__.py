from catan_engine.actions import Action, ActionType, IllegalActionError, Phase
from catan_engine.board import Board, Edge, Hex, Node, Port, create_standard_board
from catan_engine.dev_cards import DevCard
from catan_engine.observation import create_observation
from catan_engine.replay import save_replay
from catan_engine.resources import HexType, Resource
from catan_engine.rules import apply_action, get_legal_actions, is_legal_action
from catan_engine.scoring import calculate_longest_road, total_vp, visible_vp
from catan_engine.state import GameConfig, GameState, PlayerState, initialize_game


def run_game(*args, **kwargs):
    from catan_engine.simulator import run_game as _run_game

    return _run_game(*args, **kwargs)


def run_many_games(*args, **kwargs):
    from catan_engine.simulator import run_many_games as _run_many_games

    return _run_many_games(*args, **kwargs)

__all__ = [
    "Action",
    "ActionType",
    "Board",
    "DevCard",
    "Edge",
    "GameConfig",
    "GameState",
    "Hex",
    "HexType",
    "IllegalActionError",
    "Node",
    "Phase",
    "PlayerState",
    "Port",
    "Resource",
    "apply_action",
    "calculate_longest_road",
    "create_observation",
    "create_standard_board",
    "get_legal_actions",
    "initialize_game",
    "is_legal_action",
    "run_game",
    "run_many_games",
    "save_replay",
    "total_vp",
    "visible_vp",
]
