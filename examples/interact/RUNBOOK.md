# Unified assistant RL runbook

This checkout extends official THUDM Slime with one assistant-supervision API.
ScreenSim and CookBench are implemented; VH Streaming still requires a Unity
lease adapter. For the production CookSim path, follow
[COOKING_GEMINI_HANDOFF.md](COOKING_GEMINI_HANDOFF.md).

## Invariants

- Preserve each engine's native prompt, images, clocks, action schema, and grader.
- Group GRPO samples by complete `EpisodeSpec`; never mix tasks in a reward group.
- Train only on eligible terminal episodes. Infrastructure aborts are not reward zero.
- Keep generated splits immutable and verify their hashes at startup.
- Checkpoint every completed update before advancing policy weights.
- Resume the exact W&B run associated with the checkpoint; never merge engines.
- Keep one online W&B writer per job. Ray workers forward metrics through the
  logger actor and must not attach to the cloud run independently.
- Keep credentials, datasets, checkpoints, episodes, browser output, and
  `interact-runs/` out of Git.

## Local validation

```bash
source examples/interact/qwen35_env.sh
pytest -q tests/interact
```

The shared interface is in `interact_env/protocol.py`; adapters are registered in
`interact_env/registry.py`; Slime hooks are under `interact_env/slime_bridge/`.

## Current CookSim experiment

- Human: native stochastic `gemini-3.7-flash`, `classic_novice` persona.
- Train/validation: 120/10 parent-disjoint single-error scenarios.
- Update: 6 groups x 4 accepted attempts = 24 episodes.
- Validation: 10 groups x 2 attempts = 20 episodes at 0, 3, 6, and 9 updates.
- Learner: Qwen3.5-4B, frozen vision tower, TP2 x DP2, LR `5e-7`.
- Rollout: four TP1 servers when shared rendering passes, otherwise three.
- Reward: success + prevention credit - false-flag cost - bounded turn cost.

Required launch variables and split-generation commands are in the handoff doc.
The single entry point supports both fresh and resumed jobs:

```bash
sbatch examples/interact/cooking_gemini/train.sbatch
```

For resumption, set `COOKING_GEMINI_LOAD` to the parent `checkpoints` directory
and `INTERACT_WANDB_RUN_PATH` to `entity/project/run-id`. Use a new output run
directory and wait for the previous W&B writer to exit.

## Monitor

```bash
squeue -j JOB_ID -o '%.18i %.2t %.10M %.10l %R'
tail -n 100 cooking-gemini-JOB_ID.out
cat "$COOKING_GEMINI_RUN_DIR/checkpoints/latest_checkpointed_iteration.txt"
find "$COOKING_GEMINI_RUN_DIR/audit" -maxdepth 1 -name 'speculative-rollout-*.json'
```

`iter_0000002` is the checkpoint after three completed updates. Validation uses
one point at `eval/step=2` (the evaluated rollout ID, corresponding to three
completed updates); episode metrics use one-based train steps while
optimizer metrics use zero-based train steps.

## Failure handling

- Preflight failure: use the emitted evidence. The launcher automatically falls
  back from shared renderer/policy colocation to a dedicated renderer topology.
- Episode failure: one failed speculative attempt is tolerated only if four valid
  attempts remain in that group. Aborted trajectories are never trained on.
- Allocation timeout: resume the latest complete checkpoint. Never infer a
  checkpoint from W&B alone.
- Resume rejection: reconcile checkpoint metadata, W&B history, split hashes,
  and hyperparameters. Do not disable the guard.
- W&B reports the run ID is already in use: verify the single-writer logger
  actor path is enabled; W&B shared mode is not supported for continuations.
- Gemini quota/provider failure: stop and repair credentials/quota; do not turn
  API failures into task rewards.

## Interpretation

Validation has only 20 stochastic attempts, so report counts and uncertainty,
not just percentages. Inspect success, prevention, timeout, wrong serve, burns,
false flags, invalid outputs, turns, and reward together. First-four-of-five
speculation reduces tail latency but may favor shorter episodes; audit receipts
record every accepted, failed, discarded, and cancelled attempt.
