# Unified assistant RL runbook

This checkout extends official THUDM Slime with one assistant-supervision API for
native interactive engines. ScreenSim and CookBench are implemented; VH Streaming
is registered but still requires a Unity lease adapter.

## Invariants

- Preserve each engine's native prompt, images, clocks, action schema, and grader.
- Group GRPO samples by complete `EpisodeSpec`; never mix tasks inside a reward group.
- Train only on eligible terminal episodes. Infrastructure aborts are not reward zero.
- Save learner and sampler state before advancing `resume-state.json`.
- Resume the same W&B run explicitly; never merge engines into one policy curve.
- Keep checkpoints, episodes, browser output, credentials, and `interact-runs/` out of Git.

## Local validation

```bash
cd /gpfs/home/zixianma/interact/slime
source examples/interact/qwen35_env.sh
python -m pytest -q tests/interact
```

The shared interface is in `interact_env/protocol.py`; adapter discovery is in
`interact_env/registry.py`; Slime generation and reward hooks are under
`interact_env/slime_bridge/`.

## Cooking experiment

The frozen split has 40 training and 10 validation scenarios. Each update uses
6 scenario groups x 8 rollouts = 48 episodes. Validation uses 10 scenarios x 4
attempts = 40 episodes at completed updates 3, 6, 9, and 12.

```bash
export COOKING_RUN_DIR=/gpfs/scrubbed/zixianma/checkpoints/web/cooking-full-295856
sbatch --time=08:00:00 --export=ALL,COOKING_RUN_DIR="$COOKING_RUN_DIR" \
  examples/interact/cooking_full4.sbatch
```

The 4-GPU phase layout is:

- rollout: one Vulkan renderer plus three TP1 SGLang policy servers;
- update: offload rollout servers, then use all GPUs as a TP1 x DP4 learner;
- batch script topology assertions fail before training if this layout changes.

Use dependency chaining for a continuation that must start after a live allocation:

```bash
sbatch --dependency=afterany:CURRENT_JOB --time=08:00:00 \
  --export=ALL,COOKING_RUN_DIR="$COOKING_RUN_DIR" \
  examples/interact/cooking_full4.sbatch
```

Monitor only authoritative state and the active Slurm job:

```bash
cat "$COOKING_RUN_DIR/resume-state.json"
squeue -j JOB_ID -o '%.18i %.2t %.10M %.10L %R'
tail -n 100 interact-runs/cooking-full-JOB_ID.out
```

Do not submit a second independent continuation against the same run directory.
After an interrupted rollout, completed policy-bound episodes are reused from the
episode cache. After a committed update, checkpoint pointer, sampler cursor, W&B
identity, and `resume-state.json` must agree or startup fails closed.

## Rewards and tracking

Cooking retains native binary task success as the primary objective. From update
3, the lexicographic tie-break reward is:

```text
task_success + 0.1 * detected_errors / errors_planned
             - 0.01 * min(false_flags, 10)
```

Its failed and successful ranges do not overlap. Log native outcomes and the shaped
training reward separately. The cooking W&B run is
`zixianma/interact-slime-rl/wq9zbcxq`.

ScreenSim uses native in-time success. Its canonical merged W&B history is
`zixianma/interact-slime-rl/d0df345ba8f8`.
