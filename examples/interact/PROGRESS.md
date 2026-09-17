# Unified assistant RL progress

Status snapshot: 2026-09-16 22:49 PDT.

## Environment

| Engine | Adapter | Native parity | RL |
| --- | --- | --- | --- |
| ScreenSim | implemented | CPU and GPU validated | 15 updates complete |
| CookBench | implemented | CPU, renderer, rollout, checkpoint resume validated | update 4 running |
| VH Streaming | planned | pending Unity lease integration | pending |

The implementation is based on official THUDM Slime commit `4c193f1`. It provides
engine-neutral episode/observation/action/result contracts while leaving prompts,
actions, simulation, and grading engine-specific.

## ScreenSim result

Qwen3.5-4B completed 15 GRPO updates. Checkpoint verification passed, language
weights changed, and 297 frozen vision shards remained unchanged.

| Completed update | Validation success | Task-macro success |
| ---: | ---: | ---: |
| 0 | 25/48 (52.1%) | 60.0% |
| 2 | 27/48 (56.3%) | 61.7% |
| 6 | 32/48 (66.7%) | 70.8% |
| 9 | 29/48 (60.4%) | 68.3% |
| 12 | 35/48 (72.9%) | 77.5% |
| 15 | 34/48 (70.8%) | 75.8% |

These are repeated fixed validation attempts, not a statistical significance claim.
The canonical W&B run is
<https://wandb.ai/zixianma/interact-slime-rl/runs/d0df345ba8f8> and has verified
train steps 0–14 and eval steps 0, 2, 6, 9, 12, and 15. The public paired replay is
<https://zixianma.github.io/screensim-validation-replay/>.

## Cooking run

Run root: `/gpfs/scrubbed/zixianma/checkpoints/web/cooking-full-295856`.
Model: Qwen3.5-4B. Human: scripted/programmatic. No paid model API calls.

- Authoritative committed state: 3 completed updates; validation update 3 complete.
- Validation 3: 1/40 success (2.5%); 38 wrong serves, one burn, one win.
- Update 3 learner workload: 7,333 microbatches / 46,382,704 tokens in 3,868 s.
- Update 4 collected 48 fresh episodes and entered a 3,779-microbatch optimizer pass.
- Active allocation: Slurm 298131, 4 H200, scheduled to end 2026-09-17 02:21 PDT.
- Queued continuation: Slurm 298666, 4 H200 / 32 CPU / 240 GiB / 8 h,
  dependency `afterany:298131`.

The active job inherited a TP2 x DP1 learner and therefore uses only two GPUs during
optimization. The queued continuation corrects this to TP1 x DP4. Static checks and
13 targeted tests pass; GPU checkpoint resharding, memory fit, and measured speedup
remain to be verified when job 298666 starts. If TP1 is not viable, use TP2 x DP2,
not the old TP2 x DP1 layout.

Target completion remains 12 updates with validation recorded at 3, 6, 9, and 12.
