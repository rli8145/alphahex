from __future__ import annotations

import argparse
import json

from catan_engine.simulator import run_series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate baseline self-play games for future training.")
    parser.add_argument("--bot", default="heuristic", help="Bot name to self-play.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay-dir", default="replays/self_play")
    args = parser.parse_args(argv)
    summary = run_series(args.bot, args.bot, games=args.games, seed=args.seed, replay_dir=args.replay_dir)
    print(json.dumps({key: value for key, value in summary.items() if key != "replay_files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
