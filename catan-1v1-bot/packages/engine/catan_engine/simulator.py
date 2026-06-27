from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from catan_engine.actions import IllegalActionError, Phase
from catan_engine.observation import make_observation
from catan_engine.rules import apply_action, get_legal_actions, is_legal_action
from catan_engine.scoring import total_points
from catan_engine.state import GameConfig, GameState, new_game


@dataclass(frozen=True)
class GameResult:
    winner: int | None
    turns: int
    illegal_actions: int
    final_vp: tuple[int, int]
    replay_path: Path


def play_game(
    bot_a,
    bot_b,
    *,
    seed: int,
    replay_dir: str | Path = "replays",
    game_index: int = 0,
    config: GameConfig | None = None,
) -> GameResult:
    rng = random.Random(seed)
    state = new_game(seed=seed, config=config)
    bots = (bot_a, bot_b)
    illegal_actions = 0
    action_count = 0
    max_actions = state.config.max_turns * 200
    events: list[dict[str, Any]] = []

    while state.phase != Phase.GAME_OVER:
        if state.turn > state.config.max_turns or action_count > max_actions:
            state = _finish_by_score(state)
            break

        legal_actions = get_legal_actions(state)
        if not legal_actions:
            state = _finish_by_score(state)
            break

        player_id = state.current_player
        observation = make_observation(state, player_id)
        action = bots[player_id].choose_action(observation, legal_actions, rng)
        if not is_legal_action(state, action):
            illegal_actions += 1
            action = rng.choice(legal_actions)

        before = {
            "phase": state.phase.value,
            "current_player": state.current_player,
            "turn": state.turn,
        }
        try:
            state = apply_action(state, action, rng)
        except IllegalActionError:
            illegal_actions += 1
            action = rng.choice(legal_actions)
            state = apply_action(state, action, rng)
        action_count += 1
        events.append({"before": before, "action": action.to_dict(), "after": state.to_json()})

    replay_path = _write_replay(
        replay_dir=Path(replay_dir),
        bot_names=(bot_a.name, bot_b.name),
        seed=seed,
        game_index=game_index,
        state=state,
        events=events,
        illegal_actions=illegal_actions,
    )
    return GameResult(
        winner=state.winner,
        turns=state.turn,
        illegal_actions=illegal_actions,
        final_vp=(total_points(0, state), total_points(1, state)),
        replay_path=replay_path,
    )


def run_series(
    bot_a_name: str,
    bot_b_name: str,
    *,
    games: int,
    seed: int,
    replay_dir: str | Path = "replays",
    config: GameConfig | None = None,
) -> dict[str, Any]:
    from catan_bots import create_bot

    bot_a_label = f"{bot_a_name}_p0" if bot_a_name == bot_b_name else bot_a_name
    bot_b_label = f"{bot_b_name}_p1" if bot_a_name == bot_b_name else bot_b_name
    wins = {bot_a_label: 0, bot_b_label: 0, "draw": 0}
    total_turns = 0
    total_illegal = 0
    total_vp = [0, 0]
    replay_paths: list[str] = []

    for game_index in range(games):
        bot_a = create_bot(bot_a_name)
        bot_b = create_bot(bot_b_name)
        result = play_game(
            bot_a,
            bot_b,
            seed=seed + game_index,
            replay_dir=replay_dir,
            game_index=game_index,
            config=config,
        )
        if result.winner == 0:
            wins[bot_a_label] += 1
        elif result.winner == 1:
            wins[bot_b_label] += 1
        else:
            wins["draw"] += 1
        total_turns += result.turns
        total_illegal += result.illegal_actions
        total_vp[0] += result.final_vp[0]
        total_vp[1] += result.final_vp[1]
        replay_paths.append(str(result.replay_path))

    return {
        "bot_a": bot_a_name,
        "bot_b": bot_b_name,
        "games": games,
        "wins": wins,
        "average_turns": total_turns / games if games else 0.0,
        "illegal_action_count": total_illegal,
        "average_vp": {
            bot_a_label: total_vp[0] / games if games else 0.0,
            bot_b_label: total_vp[1] / games if games else 0.0,
        },
        "replay_file_path": replay_paths[-1] if replay_paths else None,
        "replay_files": replay_paths,
    }


def _finish_by_score(state: GameState) -> GameState:
    scores = (total_points(0, state), total_points(1, state))
    if scores[0] > scores[1]:
        winner = 0
    elif scores[1] > scores[0]:
        winner = 1
    else:
        winner = None
    return replace(state, phase=Phase.GAME_OVER, winner=winner)


def _write_replay(
    *,
    replay_dir: Path,
    bot_names: tuple[str, str],
    seed: int,
    game_index: int,
    state: GameState,
    events: list[dict[str, Any]],
    illegal_actions: int,
) -> Path:
    replay_dir.mkdir(parents=True, exist_ok=True)
    path = replay_dir / f"{bot_names[0]}_vs_{bot_names[1]}_seed_{seed}_game_{game_index}.json"
    payload = {
        "bot_a": bot_names[0],
        "bot_b": bot_names[1],
        "seed": seed,
        "game_index": game_index,
        "winner": state.winner,
        "turns": state.turn,
        "illegal_actions": illegal_actions,
        "final_vp": [total_points(0, state), total_points(1, state)],
        "final_state": state.to_json(),
        "events": events,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run 1v1 Catan-style bot matches.")
    parser.add_argument("--bot-a", default="random", help="Bot for player 0: random, greedy, heuristic")
    parser.add_argument("--bot-b", default="random", help="Bot for player 1: random, greedy, heuristic")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-dir", default="replays")
    args = parser.parse_args(argv)

    summary = run_series(
        args.bot_a,
        args.bot_b,
        games=args.games,
        seed=args.seed,
        replay_dir=args.replay_dir,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "replay_files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
