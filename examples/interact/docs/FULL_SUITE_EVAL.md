# Full-suite checkpoint evaluation

Use this workflow to compare checkpoints over every frozen scenario with one
rollout per scenario and checkpoint. It reports train and development-
validation subsets separately. It does not replace the repeated online
validation points used during training.

## Build immutable evaluation datasets

```bash
python examples/interact/evaluation/prepare_full_suite.py \
  --screensim-split /durable/data/screensim-gemini-18train-12val \
  --cooking-split /durable/data/cooking-gemini \
  --output /durable/eval/full-suite/data
```

The command requires 18/12 ScreenSim scenarios and 120/30 CookSim scenarios,
rejects duplicate native environment specifications, and records source and
output hashes in `manifest.json`.

## Declare checkpoints

Create a JSON file outside the repository. A Hugging Face update-0 model uses
`checkpoint_index: -1`; a trained update `N` uses checkpoint index `N - 1`.
`load_env` resolves a path from the launch environment, while `load` stores a
literal checkpoint directory.

```json
{
  "version": 1,
  "checkpoints": [
    {
      "label": "update-0",
      "update": 0,
      "checkpoint_index": -1,
      "load_env": "Q35_MODEL"
    },
    {
      "label": "update-12",
      "update": 12,
      "checkpoint_index": 11,
      "load": "/durable/runs/example/checkpoints"
    }
  ]
}
```

The resolver rejects unsafe labels, missing environment variables, and
inconsistent update/checkpoint indices.

## Launch

Set the standard model, W&B, provider, and engine variables from the training
reproduction guide, then set:

```bash
export FULL_SUITE_ROOT=/durable/eval/full-suite
export FULL_SUITE_CHECKPOINTS=/durable/eval/checkpoints.json
sbatch --array=0-1 examples/interact/evaluation/screensim.sbatch
```

For CookSim, also set `COOKING_SPLIT` and `COOKING_WORKER_PYTHON`, then run:

```bash
sbatch --array=0-1 examples/interact/evaluation/cooking.sbatch
```

Each completed array element writes `audit/result.json`. Evaluation retries
isolated transient episode failures, requires exactly one completed episode for
every catalog entry, and fails rather than reporting an incomplete aggregate.

## Read results

- `eval/success`: episode-weighted success over the complete suite.
- `eval/train/success`: success over frozen training scenarios.
- `eval/validation/success`: success over development-validation scenarios.
- `eval/*/task_macro_success`: equal-weighted average across native tasks.

One rollout per scenario is useful for broad coverage and trajectory review,
but stochastic policy and Gemini-user variation still applies. Report counts
with percentages and inspect paired trajectories before assigning a causal
interpretation to a checkpoint difference.

## Build local replays

ScreenSim:

```bash
python examples/interact/tools/visualization/build_full_suite_replay.py \
  --eval-root /durable/eval/full-suite/screensim \
  --before update-0 --after update-12 \
  --output interact-runs/screensim-full-suite-replay
```

CookSim:

```bash
python examples/interact/tools/visualization/build_cooking_full_suite_replay.py \
  --eval-root /durable/eval/full-suite/cooking \
  --output interact-runs/cooking-full-suite-replay
```

Replay builders only create local artifacts. Public packaging is a separate,
explicit step through `build_cooking_public_replay.py` or
`prepare_replay_site.py`; neither tool uploads or deploys content.

## Published evidence

- [ScreenSim update 0 vs. 12](https://zixianma.github.io/interact-rl-replays/screensim-gemini-u0-vs-u12/): complete 30-scenario matched replay.
- [CookSim update 0 vs. 12](https://zixianma.github.io/interact-rl-replays/cooking-u0-vs-u12/): update 0 is complete; the displayed update-12 export contains 149 of 150 scenarios and is marked partial.
