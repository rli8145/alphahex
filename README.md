# AlphaHex

<img src="banner-ink.svg" alt="AlphaHex - 1v1 Catan" width="100%" />

A standalone 1v1 version of *Settlers of Catan*, played against an AlphaZero-style MCTS bot.

**Play it:** https://alphahex.vercel.app/

## Layout

- `packages/catan_engine/` - rules engine, board generation, simulator
- `packages/catan_bots/` - MCTS agent (heuristic eval + neural value/policy network) and trainers
- `packages/catan_api/` - FastAPI backend
- `web/` - React/Vite client
- `api/` + `vercel.json` - Vercel config for the live site (don't delete; deploys break without them)

## Run it locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --reload
cd web && npm install && npm run dev   # http://localhost:5173, proxies /api -> :8000
```

## Database

Postgres is optional - `catan_api/db.py` records each completed game (seed, winner,
turn/move counts, signed-in user if any) for `GET /games/history`; every route
still works with no database configured, it just skips persistence silently.

Local: `docker compose up -d db`, then `export DATABASE_URL=postgresql://catan:catan@localhost:5432/catan` before starting uvicorn.

Production: use [Supabase](https://supabase.com) - Postgres plus the Auth provider
below in one project. Add it to the `catan` Vercel project via the Storage tab's
marketplace integration (auto-injects `DATABASE_URL`), or set it by hand from
Supabase's Project Settings -> Database. Schema applies itself lazily on first use.

## Accounts

Sign-in is optional and additive - `web/src/supabase.js` / `AuthPanel` are no-ops
without the env vars below, so anonymous play is unaffected. Uses [Supabase
Auth](https://supabase.com/docs/guides/auth) (GitHub + Google OAuth) on the same
Supabase project as the database above.

Setup, once per project:

1. Auth -> Providers: enable GitHub/Google with their OAuth app credentials;
   callback URL is `https://<project-ref>.supabase.co/auth/v1/callback`.
2. Auth -> URL Configuration: allow `http://localhost:5173` and
   `https://alphahex.vercel.app` as redirect URLs.
3. Env vars (Vercel + local shell): `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`
   for the frontend (Project Settings -> API, safe to expose - scoped by RLS, not
   secrecy), plus `SUPABASE_URL` (same URL, no `VITE_` prefix) for the backend.

The backend verifies tokens against Supabase's public JWKS endpoint
(`<SUPABASE_URL>/auth/v1/.well-known/jwks.json`) rather than a shared secret -
Supabase moved every project to asymmetric JWT signing in late 2025, so
there's no secret to copy. No `SUPABASE_URL` set -> `catan_api/auth.py`
always resolves an anonymous caller, same as before this existed.

## Train the agent

```bash
python -m pip install -e ".[training]"   # adds PyTorch on top of the base install

# Neural value/policy network (self-play, parallel workers, gated acceptance)
python -m catan_bots.train_mcts_nn --profile offline --seed 0            # one cycle
python -m catan_bots.train_mcts_nn --profile offline --continuous --seed 0
python -m catan_bots.train_mcts_nn --profile quick --smoke --seed 0      # regression smoke

# Heuristic MCTS evaluation weights (evolutionary self-play)
python -m catan_bots.train_mcts --seed 0
# Useful flags: --generations, --candidates, --games-per-candidate, --iterations,
# --rollout-depth, --branch-limit, --fresh, --continuous
```

Or in Docker, which pins the training environment and skips the local venv/PyTorch
setup (training only - the live app deploys separately, see below):

```bash
docker compose build train
docker compose run --rm train --profile offline --continuous --seed 0
docker compose run --rm train-heuristic --generations 20 --continuous
```

Live checkpoints: `packages/catan_bots/mcts_value_network.json` (NN, hot-reloaded by the server) and `mcts_weights.json` (heuristic). Training artifacts - replay buffer, checkpoint history, leaderboard, JSON logs, per-cycle `train_metrics.csv` - live under `data/training/` (gitignored; bind-mounted in Docker so they persist the same way). Useful flags: `--workers` (parallel self-play/eval, default cpu−1), `--lr-decay`, `--dataset-max-games`, `--eval-report`, `--fresh`.

## Deploy

Production deploys run through `.github/workflows/vercel-production.yml` on every
push to `main`. The workflow targets the linked Vercel project IDs and needs one
GitHub secret:

```text
VERCEL_TOKEN
```

Runtime installs use `requirements.txt` (just `-e .`, no PyTorch), keeping
Vercel's Python function under the serverless bundle limit. Local training
additionally installs the `training` extra (see above), which pulls in PyTorch.

## Simulate agent vs agent

```bash
python -m catan_engine.simulator --bot-a mcts --bot-b mcts --games 10 --seed 0 --no-replay
```

## How the model works

We use Monte Carlo Tree Search (MCTS) guided by two evaluators: a hand-tuned heuristic (`mcts_bot.py`, weights updated by `train_mcts.py`) and a small neural network (`value_network.py`). Search uses Polynomial Upper Confidence Trees (PUCT), as in AlphaZero: the network's policy suggests which moves are worth exploring, a heuristic orders and prunes the rest, and short rollouts score leaf positions - 70% network win-probability, 30% heuristic. Every move comes from the engine's legal-action list (`get_legal_actions(state)`).

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

The NN is lightweight with only 95 state features (production per resource, expansion potential, longest-road threat, robber exposure, tactical targets like "best available settlement spot", resource counts, phase one-hots, ...) feeding one tanh activation layer with two heads - a sigmoid **value head** trained with MSE to predict win probability, and a softmax **policy head** over 522 exact-action labels (e.g. `BUILD_ROAD:edge:17`) trained with cross-entropy loss at 0.15 weight. 

Training uses PyTorch/AdamW in batches, with CUDA when available; checkpoints are JSON for easy serving and review.

Training (`train_mcts_nn.py`) is a self-play loop: play games in parallel, label every recorded position with the final outcome (unfinished games get VP-margin pseudo-labels), train a candidate on the new games plus samples from a replay buffer, then gate acceptance - the candidate must beat the current checkpoint and also beat the heuristic-only MCTS baseline over up to 20 evaluation games (early-stopped once decided) before it is marked serving-ready. Accepted checkpoints join a history pool that self-play occasionally samples opponents from, which guards against overfitting to self. The web server hot-reloads only serving-ready live checkpoints; otherwise it falls back to heuristic MCTS.

## Background & further reading

The MCTS + value/policy network design here borrows directly from AlphaZero. Some relevant reading:

- [Monte-Carlo Tree Search in Settlers of Catan](http://www.spronck.net/pubs/ACG12Szita.pdf) - Szita, Chaslot & Spronck (2010). One of the earliest applications of MCTS to Catan itself. Good explanation on why vanilla MCTS falls short of capturing the game's randomness and trading mechanics (that said, trading is only really applicable to multiplayer games, not 1v1).
- [Re-L Catan: Evaluation of Deep Reinforcement Learning for Resource Management Under Competitive and Uncertain Environments](http://cs230.stanford.edu/projects_fall_2021/reports/103176936.pdf) - Kim & Li, Stanford CS230 (2021). A Deep Q-Learning approach to Catan, an interesting contrast to the PUCT/value-policy approach we used.
- [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270) - Silver et al., *Nature* (2017). The AlphaGo Zero paper - origin of the "value + policy network guiding MCTS via self-play" idea we use.
- [A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play](https://arxiv.org/abs/1712.01815) - Silver et al. (2017). The AlphaZero paper - origin of the PUCT selection formula used in `mcts_bot.py`.
- [A Survey of Monte Carlo Tree Search Methods](https://www.semanticscholar.org/paper/A-Survey-of-Monte-Carlo-Tree-Search-Methods-Browne-Powley/c37f1baac3c8ba30250084f067167ac3837cf6fd) - Browne et al., *IEEE TCIAIG* (2012). Broader background.
