# CookSim + Gemini-user RL reproduction

This is the short launch recipe. Read the
[full CookSim handoff](COOKING_GEMINI_HANDOFF.md) for reward semantics,
topology fallback, monitoring, and interpretation.

## 1. Build the immutable data

Check out `cook-bench-engine` at
`f6ea8d2f677ca2f92759e13affd23208cd2d33b9`, then run:

```bash
source examples/interact/models/qwen35/qwen35_env.sh
python examples/interact/cooking_gemini/prepare_cooking_rl.py \
  --extended-budget --output /durable/data/cooking-parent
python examples/interact/cooking_gemini/prepare_split.py \
  --engine-pool ../cook-bench-engine/bench/v5/cases_composite_v4.json \
  --parent-split /durable/data/cooking-parent \
  --persona classic_novice --wall-seconds 3600 \
  --output /durable/data/cooking-gemini
```

This creates 120 parent-disjoint training scenarios and 10 development-
validation scenarios. The user is native `GeminiHuman`, model
`gemini-3.7-flash`, persona `classic_novice`.

## 2. Validate

```bash
pytest -q \
  tests/interact/test_cooking_gemini_reward.py \
  tests/interact/test_cooking_speculative_rollout.py \
  tests/interact/test_cooking_episode_cache.py \
  tests/interact/test_wandb_credentials.py \
  tests/interact/test_wandb_resume.py \
  tests/test_dp_schedule.py
bash -n examples/interact/cooking_gemini/train.sbatch \
  examples/interact/cooking_gemini/run.sh
```

## 3. Submit

```bash
export Q35_RUNTIME=/path/to/qwen35-runtime
export Q35_MODEL=/path/to/Qwen3.5-4B
export COOKING_WORKER_PYTHON=/path/to/cookbench-venv/bin/python
export COOKING_GEMINI_SPLIT=/durable/data/cooking-gemini
export COOKING_GEMINI_RUN_DIR=/durable/runs/cooking-gemini-001
export COOKING_WANDB_TEAM=your-wandb-entity
export COOKING_WANDB_PROJECT=interact-slime-rl
export WANDB_ENV_FILE=/private/wandb.env
export PROVIDER_ENV_FILE=/private/gemini.env
export COOKING_NUM_ROLLOUT=12
sbatch examples/interact/cooking_gemini/train.sbatch
```

One update retains the first four valid attempts from five launched attempts for
each of six groups: 24 training episodes. Validation uses 10 scenarios x two
attempts at updates 0, 3, 6, 9, and 12.

## 4. Resume

After the old writer exits:

```bash
export COOKING_GEMINI_LOAD=/durable/runs/cooking-gemini-001/checkpoints
export COOKING_GEMINI_RUN_DIR=/durable/runs/cooking-gemini-001-resume
export INTERACT_WANDB_RUN_PATH=entity/project/run-id
sbatch examples/interact/cooking_gemini/train.sbatch
```

The launcher verifies split hashes, checkpoint state, optimizer/W&B history, and
run identity before resuming the same cloud curve.
