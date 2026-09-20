# Unified assistant RL progress

Status snapshot: 2026-09-19 23:30 PDT.

| Engine | Native adapter | RL status |
| --- | --- | --- |
| ScreenSim | implemented | Qwen3.5-4B, 15 updates complete |
| CookBench | implemented | Gemini-user Qwen3.5-4B run active; 6 updates checkpointed, update-6 validation active |
| VH Streaming | planned | pending Unity lease integration |

## CookSim Gemini-user run

The current supported profile uses 120 single-error training scenarios, 10
held-out parent variants, the engine-native `gemini-3.7-flash` classic-novice
user, Qwen3.5-4B, and the `cooking_prevention_turns_v1` reward. Each update has
24 accepted episodes and validation has 20 stochastic episodes.

| Completed updates | Success | Reward | Prevention | Timeout | Turns |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 13/20 (65%) | 0.606 | 25% | 30% | 136.2 |
| 3 | 16/20 (80%) | 0.772 | 35% | 15% | 128.0 |

Update 3 also had 5.05 false flags per episode, 5% wrong serves, no burns, and a
3.8% invalid-response fraction. The improvement is encouraging but not yet
reliable: the user and policy are stochastic and validation has only 20 attempts.

The active experiment is tracked at
<https://wandb.ai/zixianma/interact-slime-rl/runs/kv1kjbup>. This table is a
snapshot, not the final update-9 result. The operational setup—not the live job
IDs or site paths—is captured in
[COOKING_GEMINI_HANDOFF.md](COOKING_GEMINI_HANDOFF.md).

Verified pipeline properties:

- four TP1 policy servers coexist with Vulkan rendering after a stress preflight;
- all four H200s are reused for TP2 x DP2 optimization;
- first-four-of-five speculative groups tolerate one straggler/failure;
- visual tensors spill to disk and materialize per microbatch;
- static DP schedules add zero-loss padding when multi-turn sample counts do not align;
- each completed update has a checkpoint and continuations verify W&B history;
- one Ray logger actor serializes all online W&B writes, including resumed jobs;
- decision-cap timeouts receive the configured turn cost without fabricated
  prevention credit;
- full visual-tensor debug serialization is disabled.

## ScreenSim result

Qwen3.5-4B completed 15 GRPO updates. Checkpoint verification passed and the
vision tower remained frozen.

| Completed update | Validation success | Task-macro success |
| ---: | ---: | ---: |
| 0 | 25/48 (52.1%) | 60.0% |
| 2 | 27/48 (56.3%) | 61.7% |
| 6 | 32/48 (66.7%) | 70.8% |
| 9 | 29/48 (60.4%) | 68.3% |
| 12 | 35/48 (72.9%) | 77.5% |
| 15 | 34/48 (70.8%) | 75.8% |

These are repeated fixed validation attempts, not a statistical significance
claim. The canonical W&B run is
<https://wandb.ai/zixianma/interact-slime-rl/runs/d0df345ba8f8>.
