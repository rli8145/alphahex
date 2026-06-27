# catan-1v1-bot

Standalone 1v1 Catan-style bot research project. This project does not automate,
scrape, or interact with Colonist.io or live multiplayer games. It implements an
independent simulator and bot framework from public board-game rules.

## Status

Implemented:

- Fixed standard-style 19-hex board with explicit hexes, nodes, edges, and ports
- 1v1 rules config: 15 VP target, 9-card discard limit, friendly robber, maritime trades
- Setup, roll, discard, robber, steal, main, and game-over phases
- Settlements, cities, roads, resources, robber, development cards
- Longest Road, Largest Army, and hidden VP cards
- Modular normal dice and MVP balanced dice
- RandomBot, GreedyBot, and HeuristicBot
- JSON replay output
- Bot-vs-bot simulator CLI and evaluation ladder CLI

MCTS/self-play training is intentionally left as a future extension.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Run Tests

```bash
pytest
```

## Run Bot Matches

```bash
python -m catan_engine.simulator --bot-a random --bot-b random --games 100 --seed 0
python -m training.evaluate_ladder --bots random,greedy,heuristic --games 500 --seed 0
```

The simulator writes completed game replays as JSON files under `replays/` by
default.
