from __future__ import annotations

import argparse
import json
import random
from typing import Any

from catan_engine.actions import IllegalActionError, Phase
from catan_engine.replay import save_replay
from catan_engine.rules import apply_action, get_legal_actions, is_legal_action
from catan_engine.scoring import total_vp
from catan_engine.state import GameConfig, initialize_game


def run_game(
    bot_a,
    bot_b,
    config: GameConfig | None = None,
    seed: int = 0,
    max_turns: int = 1000,
    save_replay_file: bool = True,
) -> dict:
    rng = random.Random(seed)
    state = initialize_game(config=config, seed=seed)
    bots = [bot_a, bot_b]
    illegal_action_count = 0
    crash_count = 0
    max_actions = max_turns * 200
    action_count = 0

    while state.phase != Phase.GAME_OVER and state.turn_number <= max_turns and action_count <= max_actions:
        legal_actions = get_legal_actions(state)
        if not legal_actions:
            break
        player_id = state.current_player
        try:
            observation = {"player_id": player_id, "phase": state.phase.name, "_state": state}
            action = bots[player_id].choose_action(observation, legal_actions, rng)
            if not is_legal_action(state, action):
                illegal_action_count += 1
                action = rng.choice(legal_actions)
            state = apply_action(state, action, rng)
        except Exception as exc:
            crash_count += 1
            fallback = rng.choice(legal_actions)
            try:
                state = apply_action(state, fallback, rng)
            except IllegalActionError:
                raise RuntimeError("fallback legal action failed") from exc
        action_count += 1

    final_vp = [total_vp(state, 0), total_vp(state, 1)]
    if state.phase != Phase.GAME_OVER:
        if final_vp[0] > final_vp[1]:
            state.winner = 0
        elif final_vp[1] > final_vp[0]:
            state.winner = 1
        else:
            state.winner = None
        state.phase = Phase.GAME_OVER

    result = {
        "seed": seed,
        "config": state.config.to_dict(),
        "winner": state.winner,
        "final_score": final_vp,
        "turn_count": state.turn_number,
        "action_log": state.action_log,
        "final_state_summary": _final_state_summary(state),
        "illegal_action_count": illegal_action_count,
        "crash_count": crash_count,
    }
    result["replay_path"] = save_replay(result) if save_replay_file else None
    return result


def run_many_games(bot_a, bot_b, n: int, seed: int = 0, save_replays: bool = True) -> dict:
    wins = {0: 0, 1: 0, "draw": 0}
    total_turns = 0
    total_vp = [0, 0]
    illegal_action_count = 0
    crash_count = 0
    replay_paths: list[str] = []
    results: list[dict[str, Any]] = []

    for index in range(n):
        result = run_game(bot_a, bot_b, seed=seed + index, save_replay_file=save_replays)
        winner = result["winner"]
        if winner in (0, 1):
            wins[winner] += 1
        else:
            wins["draw"] += 1
        total_turns += result["turn_count"]
        total_vp[0] += result["final_score"][0]
        total_vp[1] += result["final_score"][1]
        illegal_action_count += result["illegal_action_count"]
        crash_count += result["crash_count"]
        if result["replay_path"] is not None:
            replay_paths.append(result["replay_path"])
        results.append(result)

    return {
        "games_played": n,
        "wins_by_player": {"0": wins[0], "1": wins[1], "draw": wins["draw"]},
        "average_turns": total_turns / n if n else 0.0,
        "average_final_vp": [total_vp[0] / n if n else 0.0, total_vp[1] / n if n else 0.0],
        "illegal_action_count": illegal_action_count,
        "crash_count": crash_count,
        "replay_path": replay_paths[-1] if replay_paths else None,
        "replay_paths": replay_paths,
        "results": results,
    }


def _final_state_summary(state) -> dict:
    return {
        "phase": state.phase.name,
        "winner": state.winner,
        "turn_number": state.turn_number,
        "final_vp": [total_vp(state, 0), total_vp(state, 1)],
        "longest_road_owner": state.longest_road_owner,
        "largest_army_owner": state.largest_army_owner,
        "players": [
            {
                "settlements": sorted(player.settlements),
                "cities": sorted(player.cities),
                "roads": sorted(player.roads),
                "played_knights": player.played_knights,
                "resource_count": player.total_resources(),
            }
            for player in state.players
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run standalone 1v1 Catan bot simulations.")
    parser.add_argument("--bot-a", default="mcts")
    parser.add_argument("--bot-b", default="mcts")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-replay", action="store_true", help="run simulations without writing replay JSON files")
    args = parser.parse_args(argv)

    from catan_bots import create_bot

    summary = run_many_games(
        create_bot(args.bot_a),
        create_bot(args.bot_b),
        args.games,
        seed=args.seed,
        save_replays=not args.no_replay,
    )
    printable = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
