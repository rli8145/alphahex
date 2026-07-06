# TODO

This file tracks concrete work for the 1v1 Catan simulator, web client, and MCTS-NN bot.

## Next Steps (July 2026)

Ordered by expected impact on bot strength per hour of work.

- [x] Cache per-state NN work in `_action_value` (`packages/bots/catan_bots/mcts_bot.py`): the policy distribution is now computed once per state (`_policy_probs`) and shared across candidate actions (2026-07-06).
- [x] Deduplicate `_settlement_potential`/`_expansion_count`, longest-road, and per-resource production sweeps inside `extract_state_features` — each is now computed once per call (2026-07-06).
- [x] Raise `eval_games` to 20 with an early stop once the win lead exceeds the remaining games (2026-07-06).
- [x] Use the policy prior inside tree selection (PUCT-style: `Q + c * prior * sqrt(N) / (1 + n)`), with priors normalized over the candidate set and highest-prior actions expanded first (2026-07-06).
- [x] Add parallel self-play and evaluation workers (`--workers`, default auto = cpu count - 1) (2026-07-06).
- [x] Play 2 games per history opponent in `_evaluate_history` (2026-07-06).
- [x] Gate the live website checkpoint on also beating the heuristic-only baseline: serving now only loads checkpoints marked `serving_ready`, and training marks a candidate serving-ready only after it passes both the incumbent gate and the heuristic-only baseline gate (2026-07-06).
- [x] Retune training parameters (2026-07-06): offline profile `iterations` 6→12 (benchmark showed stronger play ends games in fewer turns, so wall time per game is nearly unchanged), `games` 6→12 (parallel workers amortize it), `eval_games` 6→20 (with early stop), `buffer_samples` 2000→3000; `--accept-vp-margin` default -0.25→0.0 so tied-win candidates need a non-negative VP margin; `train_mcts.py` defaults `candidates` 2→4, `games-per-candidate` 2→6, `iterations`/`rollout-depth` 2→4, `branch-limit` 4→6 (at the old depth most games hit the 200-turn cap unfinished; heuristic games cost only ~5s each at the new depth).

## Gameplay UI

- [x] Show resource-gain events clearly in the action log.
- [x] Make the robber-blocked hex number easier to read.
- [x] Enforce friendly robber placement protection, not just friendly robber stealing/blocking protection.
- [x] Log exact stolen resource when the robber or a Knight steals.
- [x] During normal play, show hover/click targets for legal road, settlement, and city builds.
- [x] Show each player's total card count.
- [x] Warn when a player has 10 or more resource cards and may need to discard on a 7.
- [x] Replace remaining emoji UI with custom token-style icons.
- [x] Add visible delays/status messages between bot actions so players can follow the bot turn.
- [x] Add board animations for dice payouts (matching hexes flash after a roll) and robber movement (the robber slides between hexes).
- [x] Add a compact game summary panel for VP, production, and longest-road pressure.

## Bot Strength Roadmap

- [x] Keep the current heuristic-weight MCTS as the baseline model.
- [x] Add checkpoint evaluation before accepting neural-network updates.
- [x] Train a policy head in addition to the value head.
- [x] Save a self-play replay dataset under `data/training/`.
- [x] Train from a replay buffer instead of only the latest self-play batch.
- [x] Evaluate candidate checkpoints against older checkpoint versions.
- [x] Sample self-play opponents from checkpoint history to reduce overfitting.
- [x] Use larger MCTS search budgets for training and evaluation.
- [x] Keep a capped search budget for the website bot so play remains responsive.
- [x] Add richer exact-action policy targets, not only action-type policy targets.
- [x] Add parallel self-play workers once single-process training is stable.

## Feature Engineering

- [x] Add per-resource production features.
- [x] Add settlement and city build-potential features.
- [x] Add road connectivity and longest-road threat features.
- [x] Add opponent expansion-blocking features.
- [x] Add development-card timing features.
- [x] Add resource scarcity and bank-pressure-style hand-risk features.
- [x] Add port usefulness features based on actual production.
- [x] Add robber-impact features for both players.
- [x] Add immediate tactical win/block features.
- [x] Add action-specific tactical features for exact road, city, settlement, and robber targets (best settlement spot, best city spot, best road target, best robber gain for both players).

## Training Quality

- [x] Track win rate, VP margin, illegal actions, crashes, and average turns for every candidate checkpoint.
- [x] Keep a small leaderboard of recent checkpoints.
- [x] Add a finite evaluation command for `candidate vs current`.
- [x] Add a stronger offline training profile separate from web-play defaults.
- [x] Add a quick regression smoke command for random valid boards plus MCTS-NN.
- [x] Enforce stricter valid boards with no adjacent matching number tokens.
- [x] Add live JSON progress logs for self-play games, SGD training, evaluation, and checkpoint saves.
- [x] Randomize first setup pick between player and bot.
- [x] Randomize port order/rotation during board generation.
- [x] Move continuous training logs under `data/training/logs/` instead of the temp folder.
- [x] Improve random board selection with a simple fairness score for resource clustering and production balance.
- [x] Stop tracking generated training artifacts; keep `data/training/.gitkeep`, ignore `selfplay.jsonl`, `leaderboard.json`, checkpoint history, and local model JSON files.
- [x] Let the first exact-policy training cycles finish and record the result (2026-07-06, new tactical-feature pipeline, offline profile): cycle 1 bootstrapped a fresh network (auto-accepted, no incumbent) but lost 0-11 to the heuristic-only baseline before eval early-stopped; cycle 2's candidate was **rejected** 8-9 (3 draws, +0.05 VP margin) against the cycle-1 incumbent over the full 20 eval games. Value loss fell 0.24→0.049 across the two cycles; zero illegal actions or crashes.
- [x] Add a readable eval report command for win rate, VP margin, average turns, illegal actions, crashes, and checkpoint comparison.
- [x] Add chart-friendly CSV export for long training runs (`data/training/logs/train_metrics.csv`, one row per cycle).
- [x] Add periodic pruning/compaction for very large `data/training/selfplay.jsonl` files (`--dataset-max-games`, default 4000 newest games).
- [x] Add learning-rate decay across continuous training cycles (`--lr-decay`, default 0.995 per cycle).
- [x] Port `ValueNetwork` forward/backward passes to PyTorch (2026-07-06): training now uses batched AdamW on CUDA when available, while checkpoints remain plain JSON and the MCTS/runtime API stays unchanged.

## Training Math

- The value head predicts win chance from state features. Its derivative is computed in `packages/bots/catan_bots/value_network.py` in `_train_one()` at `d_logit = (output - target) * output * (1.0 - output)`.
- The policy head predicts the chosen exact action label, such as `BUILD_ROAD:edge:17` or `MOVE_ROBBER:hex:8`. Its gradient is `probability - label`, scaled by `policy_loss_weight`, and is applied in the same `_train_one()` method.
- `rng.shuffle(order)` randomizes training-example order each epoch so sequential positions from the same self-play game do not update the weights in the same repeated pattern.
