from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from catan_engine.simulator import run_series


def evaluate_ladder(
    bot_names: list[str],
    *,
    games: int,
    seed: int,
    replay_dir: str | Path = "replays/ladder",
) -> dict[str, Any]:
    standings = {
        bot: {"wins": 0, "losses": 0, "draws": 0, "vp_total": 0.0, "games": 0}
        for bot in bot_names
    }
    matchups: list[dict[str, Any]] = []
    next_seed = seed

    for bot_a in bot_names:
        for bot_b in bot_names:
            if bot_a == bot_b:
                continue
            summary = run_series(bot_a, bot_b, games=games, seed=next_seed, replay_dir=replay_dir)
            next_seed += games
            matchups.append({key: value for key, value in summary.items() if key != "replay_files"})

            wins = summary["wins"]
            standings[bot_a]["wins"] += wins.get(bot_a, 0)
            standings[bot_a]["losses"] += wins.get(bot_b, 0)
            standings[bot_a]["draws"] += wins.get("draw", 0)
            standings[bot_a]["vp_total"] += summary["average_vp"][bot_a] * games
            standings[bot_a]["games"] += games

            standings[bot_b]["wins"] += wins.get(bot_b, 0)
            standings[bot_b]["losses"] += wins.get(bot_a, 0)
            standings[bot_b]["draws"] += wins.get("draw", 0)
            standings[bot_b]["vp_total"] += summary["average_vp"][bot_b] * games
            standings[bot_b]["games"] += games

    for bot, row in standings.items():
        row["average_vp"] = row["vp_total"] / row["games"] if row["games"] else 0.0
        del row["vp_total"]

    return {"bots": bot_names, "games_per_ordered_matchup": games, "standings": standings, "matchups": matchups}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Catan bots in an ordered round-robin ladder.")
    parser.add_argument("--bots", default="random,greedy,heuristic")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-dir", default="replays/ladder")
    args = parser.parse_args(argv)

    bot_names = [name.strip() for name in args.bots.split(",") if name.strip()]
    summary = evaluate_ladder(bot_names, games=args.games, seed=args.seed, replay_dir=args.replay_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
