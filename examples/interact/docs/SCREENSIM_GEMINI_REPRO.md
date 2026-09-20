# ScreenSim + Gemini-user RL reproduction

This is the supported Qwen3.5-4B experiment: 18 training scenarios, 12 held-out
development-validation scenarios, engine-native `PlanHuman` using
`gemini-3.7-flash`, and validation at updates 0, 3, 6, 9, and 12.

## 1. Prerequisites

- One four-H200 node (32 CPUs, 480 GiB RAM) and a compatible Slime runtime.
- A local Qwen3.5-4B checkpoint.
- `screensim-engine` checked out at
  `869d8e9e5b9e27c4a71bcc4cc31c75216bfad418`.
- A private W&B env file containing `WANDB_API_KEY`.
- A private provider env file containing `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

The pinned engine's `LLMClient` must read the provider key from the environment,
or the same key must be installed at its native `~/.gemini_key` location. Never
commit or print either credential.

## 2. Build the frozen Gemini split

Start from the existing deterministic 18/12 ScreenSim RL manifest. Output must
be a new directory.

```bash
source examples/interact/models/qwen35/qwen35_env.sh
python examples/interact/screensim/prepare_gemini_split.py \
  --source /durable/data/screensim-rl-v2/manifest.json \
  --engine-root ../screensim-engine \
  --output-dir /durable/data/screensim-gemini-18train-12val
```

The builder recertifies every composite against the checked-out engine and
records the source hash, exact engine revision, human profile, and split counts.

## 3. Validate code and data handling

```bash
pytest -q \
  tests/interact/test_screensim.py \
  tests/interact/test_screensim_gemini_split.py \
  tests/interact/test_wandb_resume.py \
  tests/test_dp_schedule.py
bash -n examples/interact/screensim/gemini_train.sbatch \
  examples/interact/models/qwen35/run_qwen35_train.sh
```

## 4. Submit a fresh 12-update run

```bash
export Q35_RUNTIME=/path/to/qwen35-runtime
export Q35_MODEL=/path/to/Qwen3.5-4B
export SCREENSIM_ENGINE_ROOT=/path/to/screensim-engine
export SCREENSIM_GEMINI_SPLIT=/durable/data/screensim-gemini-18train-12val
export SCREENSIM_GEMINI_RUN_DIR=/durable/runs/screensim-gemini-001
export SCREENSIM_WANDB_TEAM=your-wandb-entity
export SCREENSIM_WANDB_PROJECT=interact-slime-rl
export WANDB_ENV_FILE=/private/wandb.env
export PROVIDER_ENV_FILE=/private/gemini.env
export SCREENSIM_NUM_ROLLOUT=12
sbatch examples/interact/screensim/gemini_train.sbatch
```

Each update samples six scenario groups with eight policy attempts (48 episodes,
global batch 48). Validation samples all 12 held-out scenarios four times (48
episodes). The assistant uses Qwen3.5-4B; only the simulated user calls Gemini.
Large visual-tensor debug dumps are disabled by default.

## 5. Resume an interrupted allocation

Wait for the old process to exit, then use its checkpoint directory and a new,
empty output directory:

```bash
export SCREENSIM_GEMINI_LOAD=/durable/runs/screensim-gemini-001/checkpoints
export SCREENSIM_GEMINI_RUN_DIR=/durable/runs/screensim-gemini-001-resume
export INTERACT_WANDB_RUN_PATH=entity/project/run-id
export SCREENSIM_NUM_ROLLOUT=12
sbatch examples/interact/screensim/gemini_train.sbatch
```

The W&B identity is also stored in `checkpoints/wandb_run.json`. Resume verifies
optimizer history and replays a missing boundary validation before collecting
the next update. Never run two writers against one W&B run.

## Reading results

Use `train/success` for rollout behavior and `eval/success` plus
`eval/task_macro_success` for held-out comparison. W&B `eval/step` is the number
of completed updates: 0, 3, 6, 9, and 12. Because Gemini and policy sampling are
stochastic, compare all 48 validation episodes and confidence intervals rather
than interpreting one trajectory.

The only experiment predating this guide was job `299664`: one update, two
failed training episodes, and no validation. It is a pipeline smoke test, not a
baseline learning curve.
