from __future__ import annotations

import random

from fastapi import APIRouter, Depends, Query

from catan_api.auth import current_user
from catan_api.db import fetch_recent_games, record_finished_game
from catan_api.schemas import BotMatchRequest, BotStepRequest, GameActionRequest, NewGameRequest
from catan_bots import create_bot
from catan_engine.actions import Action, Phase
from catan_engine.observation import create_observation
from catan_engine.rules import apply_action, get_legal_actions
from catan_engine.simulator import run_many_games
from catan_engine.state import GameState, initialize_game

router = APIRouter()

# A single reusable bot instance; MCTSBot is stateless across calls.
_BOT = create_bot("mcts")


@router.post("/games/new")
def new_game(request: NewGameRequest) -> dict:
    state = initialize_game(seed=request.seed)
    return _game_payload(state, player_id=0)


@router.post("/games/action")
def game_action(request: GameActionRequest, user_id: str | None = Depends(current_user)) -> dict:
    state = GameState.from_dict(request.state)
    action = Action.from_dict(request.action)
    rng = random.Random(state.rng_seed + len(state.action_log))
    new_state = apply_action(state, action, rng)
    _record_if_finished(state, new_state, user_id)
    payload = _game_payload(new_state, player_id=new_state.current_player)
    payload["observations"] = [create_observation(new_state, 0), create_observation(new_state, 1)]
    payload["winner"] = new_state.winner
    return payload


@router.post("/games/bot-step")
def bot_step(request: BotStepRequest, user_id: str | None = Depends(current_user)) -> dict:
    """Let the bot choose and apply a single action for whoever is to move."""
    state = GameState.from_dict(request.state)
    if state.phase == Phase.GAME_OVER or state.winner is not None:
        payload = _game_payload(state, player_id=state.current_player)
        payload["action"] = None
        return payload

    player_id = state.current_player
    legal_actions = get_legal_actions(state)
    observation = create_observation(state, player_id)
    observation["_state"] = state
    rng = random.Random(state.rng_seed + len(state.action_log))
    action = _BOT.choose_action(observation, legal_actions, rng)
    new_state = apply_action(state, action, rng)
    _record_if_finished(state, new_state, user_id)
    payload = _game_payload(new_state, player_id=new_state.current_player)
    payload["action"] = new_state.action_log[-1] if new_state.action_log else action.to_dict()
    return payload


@router.post("/games/bot-match")
def bot_match(request: BotMatchRequest) -> dict:
    return run_many_games(create_bot(request.bot_a), create_bot(request.bot_b), request.games, seed=request.seed)


@router.get("/games/history")
def game_history(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """Recently completed games (all players), if Postgres is configured (see catan_api.db); [] otherwise."""
    return {"games": fetch_recent_games(limit)}


@router.get("/games/history/me")
def my_game_history(
    limit: int = Query(default=20, ge=1, le=100), user_id: str | None = Depends(current_user)
) -> dict:
    """Recently completed games for the signed-in caller; [] if not signed in or Postgres isn't configured."""
    if user_id is None:
        return {"games": []}
    return {"games": fetch_recent_games(limit, user_id=user_id)}


def _record_if_finished(state: GameState, new_state: GameState, user_id: str | None) -> None:
    """Persist a game exactly once, on the move that transitions it into game-over."""
    if state.winner is None and new_state.winner is not None:
        record_finished_game(
            seed=new_state.rng_seed,
            winner=new_state.winner,
            turn_count=new_state.turn_number,
            move_count=len(new_state.action_log),
            user_id=user_id,
        )


def _game_payload(state: GameState, player_id: int) -> dict:
    return {
        "state": state.to_dict(),
        "observation": create_observation(state, player_id),
        "legal_actions": [action.to_dict() for action in get_legal_actions(state)],
        "winner": state.winner,
    }
