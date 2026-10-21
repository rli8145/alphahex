# Checkpoint archive

This directory preserves all 14 checkpoint files found locally on 2026-10-20:
one heuristic weight file and 13 neural-network snapshots. They are small JSON
files (about 3 MB combined), stored directly in Git without Git LFS. Original
files remain in their training locations; these copies preserve their bytes and
training metadata. `manifest.json` records source paths, byte counts, SHA-256
hashes, architecture, and compatibility.

| Path | Contents | Compatibility |
| --- | --- | --- |
| `heuristic/mcts_weights.json` | Evolutionarily tuned MCTS evaluation weights; training seed 21000000 | Loadable with the current `load_trained_weights(path)` function |
| `neural/mcts_value_network.json` | Last local live-file snapshot; 87 inputs, 32 hidden units, 522 policy labels | Legacy encoder; last recorded training cycle was rejected |
| `neural/history/value_network_178283*.json` | Two accepted historical snapshots; 87 inputs, 32 hidden units, 522 policy labels | Legacy exact-action policy architecture |
| Other files in `neural/history/` | Ten accepted historical snapshots; 87 inputs, 32 hidden units, 16 policy labels | Legacy action-type policy architecture |

The `accepted` field describes the recorded training cycle, not a guarantee of
compatibility or playing strength. The live-file snapshot can contain incumbent
weights after a rejected candidate; its metadata alone does not identify the
saved weights as that rejected candidate.

## Using the weights

The current app reads `packages/catan_bots/mcts_weights.json` and
`packages/catan_bots/mcts_value_network.json`. This archive is separate from
those live paths and is excluded by `.vercelignore` and `.dockerignore`.

The heuristic file can be passed to `load_trained_weights(path)` or copied to
the live heuristic path to opt into the saved tuning.

All archived neural networks have 87 input features. The current encoder has 95
features and 522 policy labels, and its serving loader requires
`training.serving_ready`. None of these snapshots has that flag, so none is a
current serving checkpoint. Do not copy them to the live NN path or change their
readiness metadata to activate them.

For historical reproduction, the 87-input/16-policy architecture is present at
commit `5363fc1`; the 87-input/522-policy architecture is present at `44a79cb`.
Both use the former `packages/bots/catan_bots/` layout. Use the corresponding
engine and encoder when evaluating those models. Train a fresh model with the
current trainer and let its incumbent and heuristic-baseline gates establish
serving readiness for the current app.

At the archive audit on 2026-10-20,
[`/api/bots/version`](https://alphahex.vercel.app/api/bots/version) returned
`active_model: heuristic`, `version: heuristic-baseline`,
`serving_ready: false`, and `checkpoint: null`. Inference does not require a
separate hosted model service: `value_network.py` can evaluate JSON weights
without PyTorch; PyTorch is needed for training.
