# Agent Guidance

## Scope And Safety

This is a standalone 1v1 Catan-style simulator and bot research project.

Do not:

- Automate Colonist.io or any other live game site.
- Scrape Colonist.io.
- Reverse engineer network traffic.
- Interact with live multiplayer games.

Build and test only the independent simulator, bot framework, training stubs, and replay tooling in this repository.

## Project Layout

- `packages/engine/catan_engine/`: rules engine, state, board, dice, scoring, simulator CLI.
- `packages/bots/catan_bots/`: bot API and baseline bots.
- `packages/training/`: ladder and self-play entry points.
- `packages/engine/tests/`: pytest coverage for core rules.

## Development Commands

From `catan-1v1-bot/`:

```bash
python -m pip install -e ".[dev]"
pytest
python -m catan_engine.simulator --bot-a random --bot-b random --games 100 --seed 0
python -m training.evaluate_ladder --bots random,greedy,heuristic --games 500 --seed 0
```

When running directly from source without an editable install on Windows PowerShell:

```powershell
$env:PYTHONPATH='packages/engine;packages/bots;packages'
```

## Engineering Notes

- Keep game legality in `catan_engine.rules`, not in bots.
- Bots should choose only from `get_legal_actions(state)`.
- `apply_action(state, action, rng)` must validate legality before state transition.
- Preserve cloneable state semantics: avoid shared mutable state inside `GameState` and `PlayerState`.
- Keep dice policy modular in `dice.py`; the balanced dice implementation is intentionally an MVP approximation.
- Store completed simulator games as JSON replays, but do not commit generated replay directories unless explicitly requested.
- Do not add MCTS until RandomBot, GreedyBot, and HeuristicBot are reliable.
