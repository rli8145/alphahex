# AlphaHex - 1v1 Catan

A 1v1 Catan MCTS agent.

**Play:** https://alphahex.vercel.app/

## Layout

- `packages/engine/` - rules engine, board generation, simulator
- `packages/bots/` - MCTS agent (heuristic eval + neural value/policy network) and trainers
- `packages/api/` - FastAPI backend
- `web/` - React/Vite client
- `api/` + `vercel.json` - Vercel config for the live site (don't delete; deploys break without them)

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
by the server) and `mcts_weights.json` (heuristic). Training artifacts - replay buffer,
checkpoint history, leaderboard, JSON logs, per-cycle `train_metrics.csv` - live under
`data/training/` (gitignored). Useful flags: `--workers` (parallel self-play/eval,
default cpu−1), `--lr-decay`, `--dataset-max-games`, `--eval-report`, `--fresh`.

## Simulate agent vs agent

```bash
python -m catan_engine.simulator --bot-a mcts --bot-b mcts --games 10 --seed 0 --no-replay
```

## How the model works

We use Monte Carlo tree search guided by two evaluators: a hand-tuned heuristic (`mcts_bot.py`, weights evolved by `train_mcts.py`) and a small neural network (`value_network.py`). Search uses Polynomial Upper Confidence Trees (PUCT), as in AlphaZero: the network's policy suggests which moves are worth exploring, a heuristic orders and prunes the rest, and short rollouts score leaf positions — 70% network win-probability, 30% heuristic. Every move comes from the engine's legal-action list (`get_legal_actions(state)`).

```text
ENGINE / SEARCH LOOP  (rules-owned, every move legal-checked)
--------------------------------------------------------------
state --> get_legal_actions --> MCTS / PUCT --> apply_action
  ^              |                    |                 |
  |              |                    |                 v
  |              |                    +-- chosen legal action
  |              |
  |              v
  |        action candidates
  |        from catan_engine.rules only
  |
  +-------- next GameState clone for search / rollout

MODEL EVALUATION  (mixed heuristic + neural guidance)
-----------------------------------------------------
GameState --> 95-feature encoder --> PyTorch value-policy network
                                      |
                                      +-- value head: win probability
                                      +-- policy head: 522 exact-action labels

heuristic evaluator --> production, VP, roads, ports, tactics
         |
         +-- leaf score + move ordering fallback

MCTS leaf score = 70% NN value + 30% heuristic
PUCT selection = Q + c * policy_prior * sqrt(N) / (1 + n)

TRAINING LOOP  (parallel self-play, checkpoint gated)
-----------------------------------------------------
self-play games --> JSONL replay buffer --> SGD training
       |                    |                    |
       |                    |                    +-- value loss: MSE(win target)
       |                    |                    +-- policy loss: exact-action CE
       |                    |
       v                    v
 candidate checkpoint --> eval vs incumbent --> eval vs heuristic baseline
                                      |                    |
                                      +--------+-----------+
                                               v
                                 serving_ready JSON checkpoint
                                               |
                                               v
                                    website bot hot reloads
```

The NN is lightweight with only 95 state features (production per resource, expansion potential, longest-road threat, robber exposure, tactical targets like "best available settlement spot", resource counts, phase one-hots) feeding one tanh hidden layer with two heads - a sigmoid **value head** trained with MSE to predict win probability, and a softmax **policy head** over 522 exact-action labels (e.g. `BUILD_ROAD:edge:17`) trained with cross-entropy loss at 0.15 weight. Training uses PyTorch/AdamW in batches, with CUDA when available; checkpoints stay JSON for easy serving and review.

Training (`train_mcts_nn.py`) is a self-play loop: play games in parallel, label every recorded position with the final outcome (unfinished games get VP-margin pseudo-labels), train a candidate on the new games plus samples from a replay buffer, then gate acceptance - the candidate must beat the current checkpoint and also beat the heuristic-only MCTS baseline over up to 20 evaluation games (early-stopped once decided) before it is marked serving-ready. Accepted checkpoints join a history pool that self-play occasionally samples opponents from, which guards against overfitting to self. The web server hot-reloads only serving-ready live checkpoints; otherwise it falls back to heuristic MCTS.
