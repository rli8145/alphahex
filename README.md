# AlphaHex — 1v1 Catan

A 1v1 Catan MCTS agent.

**Play:** https://alphahex.vercel.app/

## Layout

- `packages/engine/` — rules engine, board generation, simulator
- `packages/bots/` — MCTS agent (heuristic eval + neural value/policy network) and trainers
- `packages/api/` — FastAPI backend
- `web/` — React/Vite client
- `api/` + `vercel.json` — Vercel config for the live site (don't delete; deploys break without them)

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --reload
cd web && npm install && npm run dev   # http://localhost:5173, proxies /api -> :8000
```

## Train the agent

```bash
# Neural value/policy network (self-play, parallel workers, gated acceptance)
python -m catan_bots.train_mcts_nn --profile offline --seed 0            # one cycle
python -m catan_bots.train_mcts_nn --profile offline --continuous --seed 0
python -m catan_bots.train_mcts_nn --profile quick --smoke --seed 0      # regression smoke

# Heuristic MCTS evaluation weights (evolutionary self-play)
python -m catan_bots.train_mcts --seed 0
```

Live checkpoints: `packages/bots/catan_bots/mcts_value_network.json` (NN, hot-reloaded
by the server) and `mcts_weights.json` (heuristic). Training artifacts — replay buffer,
checkpoint history, leaderboard, JSON logs, per-cycle `train_metrics.csv` — live under
`data/training/` (gitignored). Useful flags: `--workers` (parallel self-play/eval,
default cpu−1), `--lr-decay`, `--dataset-max-games`, `--eval-report`, `--fresh`.

## Simulate agent vs agent

```bash
python -m catan_engine.simulator --bot-a mcts --bot-b mcts --games 10 --seed 0 --no-replay
```

## How the agent works

The agent is Monte Carlo tree search guided by two evaluators: a hand-tuned
heuristic (`mcts_bot.py`, weights evolved by `train_mcts.py`) and a small neural
network (`value_network.py`). Search uses PUCT-style selection - the network's
policy prior steers which branches get explored — with heuristic move ordering,
branch limiting, and shallow rollouts; leaf positions are scored as a 70/30 blend
of the network's win probability and the heuristic. The agent only ever chooses
from `get_legal_actions(state)`.

The network itself is deliberately tiny: ~95 hand-crafted state features
(production per resource, expansion potential, longest-road threat, robber
exposure, tactical targets like "best available settlement spot", resource
counts, phase one-hots) feeding one 64-unit tanh hidden layer with two heads —
a sigmoid **value head** trained with MSE to predict win probability, and a
softmax **policy head** over ~700 exact-action labels (e.g. `BUILD_ROAD:edge:17`)
trained with cross-entropy at 0.15 weight. Plain numpy SGD with L2; checkpoints
are JSON.

Training (`train_mcts_nn.py`) is a self-play loop: play games in parallel
workers, label every recorded position with the final outcome (unfinished games
get VP-margin pseudo-labels), train a candidate on the new games plus samples
from a replay buffer, then gate acceptance — the candidate must beat the current
checkpoint over up to 20 evaluation games (early-stopped once decided) to
replace it. Accepted checkpoints join a history pool that self-play occasionally
samples opponents from, which guards against overfitting to self. The web
server hot-reloads the live checkpoint whenever it changes on disk.