# CookSim Gemini-user RL handoff

This is the supported CookSim training path. It keeps CookBench's native
supervisor prompt, rendered observations, Gemini user, simulator, and outcome
report, while Slime trains a local Qwen3.5-4B assistant with GRPO.

## Validated configuration

- Slime base: official THUDM/slime through `4c193f1`, plus this repository's
  unified-environment changes.
- CookBench engine: sibling checkout at revision
  `f6ea8d2f677ca2f92759e13affd23208cd2d33b9`.
- Assistant: Qwen3.5-4B, language weights trainable and vision tower frozen.
- User: engine-native `GeminiHuman`, model pinned to `gemini-3.7-flash`, persona
  `classic_novice`. These are paid, stochastic API calls.
- Data: 120 parent-disjoint single-error training scenarios and 10 held-out
  parent variants for development validation.
- Update: 6 groups x 4 accepted attempts = 24 episodes; five attempts are
  launched per group and the first four valid completions are retained.
- Validation: 10 groups x 2 attempts = 20 episodes at baseline and every three
  completed updates.
- Reward:

  ```text
  success + 0.30 * prevented
          - 0.02 * min(false_flags, 15)
          - 0.05 * min(assistant_turns / 200, 1)
  ```

  Every success remains above every failure. Native task success, prevention,
  false flags, turns, invalid-response fraction, and outcomes are logged
  separately from the scalar reward.
- Hardware: one 4-H200 node with 32 CPUs and 480 GiB RAM. Rollout uses four TP1
  SGLang servers when renderer/policy colocation passes; otherwise it safely
  falls back to one renderer plus three policy servers. Learning uses all four
  GPUs as TP2 x DP2.

The first run of this exact profile improved 20-episode validation success from
13/20 (65%) at update 0 to 16/20 (80%) at update 3; reward increased from 0.606
to 0.772. This is promising development evidence, not a significance claim or a
final result. The run was still active when this handoff was written.

## Prerequisites

1. Clone this repository and `cook-bench-engine` as sibling directories.
2. Check out the CookBench revision above and install its native dependencies,
   browser, and Vulkan renderer requirements.
3. Provide a Qwen3.5-4B checkpoint and a compatible Slime/Megatron/SGLang
   runtime. `Q35_RUNTIME` may override the site-local default in
   `qwen35_env.sh`.
4. Create two private env files:
   - `WANDB_ENV_FILE` containing `WANDB_API_KEY`;
   - `PROVIDER_ENV_FILE` containing `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

The credential wrapper reads only those allowlisted keys. Never commit either
file, print its contents, or pass a secret as a command-line argument.

## Rebuild the immutable split

Use Python 3.12 from the configured runtime. Output directories must be new.

```bash
source examples/interact/qwen35_env.sh
python examples/interact/prepare_cooking_rl.py \
  --extended-budget --output /durable/data/cooking-parent-split
python examples/interact/cooking_gemini/prepare_split.py \
  --engine-pool ../cook-bench-engine/bench/v5/cases_composite_v4.json \
  --parent-split /durable/data/cooking-parent-split \
  --persona classic_novice --wall-seconds 3600 \
  --output /durable/data/cooking-gemini-novice-split
```

Keep each generated `manifest.json`. Training verifies all dataset and case-pool
hashes before allocating Ray workers.

## Validate before submission

```bash
source examples/interact/qwen35_env.sh
pytest -q tests/interact/test_cooking_gemini_reward.py \
  tests/interact/test_cooking_speculative_rollout.py \
  tests/interact/test_cooking_episode_cache.py \
  tests/interact/test_wandb_credentials.py \
  tests/interact/test_wandb_resume.py \
  tests/test_dp_schedule.py
bash -n examples/interact/cooking_gemini/train.sbatch \
  examples/interact/cooking_gemini/run.sh
```

## Start a fresh run

The launcher refuses missing paths and performs renderer-only and shared
renderer/policy stress tests before training. Create the Slurm output directory
or submit from a writable directory.

```bash
export Q35_RUNTIME=/path/to/qwen35-runtime
export Q35_MODEL=/path/to/Qwen3.5-4B
export COOKING_WORKER_PYTHON=/path/to/cookbench-venv/bin/python
export COOKING_GEMINI_SPLIT=/durable/data/cooking-gemini-novice-split
export COOKING_GEMINI_RUN_DIR=/durable/runs/cooking-gemini-001
export COOKING_WANDB_TEAM=your-wandb-entity
export COOKING_WANDB_PROJECT=interact-slime-rl
export WANDB_ENV_FILE=/private/wandb.env
export PROVIDER_ENV_FILE=/private/gemini.env
export COOKING_NUM_ROLLOUT=9
sbatch examples/interact/cooking_gemini/train.sbatch
```

The run directory owns checkpoints, episode artifacts, preflight evidence,
speculative-rollout receipts, and local W&B state. Do not point two live jobs at
the same run directory.

## Resume safely

Wait until the first writer has exited. Use its `checkpoints` directory and the
exact W&B identity from `checkpoints/wandb_run.json`:

```bash
export COOKING_GEMINI_LOAD=/durable/runs/cooking-gemini-001/checkpoints
export INTERACT_WANDB_RUN_PATH=entity/project/run-id
export COOKING_GEMINI_RUN_DIR=/durable/runs/cooking-gemini-001-resume
export COOKING_NUM_ROLLOUT=9
sbatch examples/interact/cooking_gemini/train.sbatch
```

Resume fails before training if checkpoint metadata, optimizer/scheduler state,
W&B history, data hashes, model/sampling configuration, or run identity disagree.
Every completed update is checkpointed, so an allocation timeout loses at most
the in-progress rollout/update.

## Monitoring and interpretation

```bash
squeue -j JOB_ID -o '%.18i %.2t %.10M %.10l %R'
tail -n 100 cooking-gemini-JOB_ID.out
cat "$COOKING_GEMINI_RUN_DIR/checkpoints/latest_checkpointed_iteration.txt"
ls "$COOKING_GEMINI_RUN_DIR/audit"/speculative-rollout-*.json
```

Checkpoint directory `iter_0000002` means rollout IDs 0--2 are committed: three
completed updates. Training episode metrics use one-based `train/step`; optimizer
metrics use zero-based `train/step`. Validation uses the number of completed
updates (`eval/step` 0, 3, 6, 9).

Treat validation as noisy because the user and policy are stochastic and there
are only 20 attempts. Compare success, reward, prevention, timeout, wrong-serve,
turns, false flags, and invalid-response fraction together. The speculative
first-four-of-five collector reduces straggler time but can favor shorter
episodes; its per-attempt receipts make that bias auditable.

## Key files

- `cooking_gemini/train.sbatch`: fresh/resume Slurm entry point.
- `cooking_gemini/run.sh`: frozen optimizer and rollout arguments.
- `cooking_gemini/train.py`: data/topology/resume invariants.
- `cooking_gemini/prepare_split.py`: deterministic split derivation.
- `cooking_gemini/speculative_rollout.py`: first-four-of-five collector.
- `cooking_gemini/metrics.py`: episode-weighted train/eval metrics.
- `cooking_gemini/render_policy_preflight.py`: shared-GPU topology gate.
- `cooking_gemini/validate_reward.py`: retrospective reward audit.
- `interact_env/adapters/cooking_rewards.py`: versioned reward.

Cluster paths, API keys, generated datasets, episode artifacts, checkpoints, and
W&B local files are intentionally not committed.
