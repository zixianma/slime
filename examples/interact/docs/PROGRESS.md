# Unified assistant RL progress

Status snapshot: 2026-09-22 PDT.

| Engine | Native adapter | RL status |
| --- | --- | --- |
| ScreenSim | implemented | Qwen3.5-4B, 15 updates complete |
| CookBench | implemented | Gemini-user Qwen3.5-4B continuation and full-suite evaluation tooling validated |
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

The original experiment is tracked at
<https://wandb.ai/zixianma/interact-slime-rl/runs/kv1kjbup>. This table is a
training-curve snapshot rather than a final full-suite result. The operational
setup is captured in
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

## Matched full-suite checkpoint evidence

The baseline-persona ScreenSim update-0/update-12 comparison covers each frozen
scenario once at each checkpoint:

| Split | Update 0 | Update 12 |
| --- | ---: | ---: |
| Train | 5/18 (27.8%) | 8/18 (44.4%) |
| Validation | 4/12 (33.3%) | 3/12 (25.0%) |
| Overall | 9/30 (30.0%) | 11/30 (36.7%) |

This is broad paired evidence with one stochastic rollout per scenario. See the
[evaluation protocol](FULL_SUITE_EVAL.md) and
[public ScreenSim replay](https://zixianma.github.io/interact-rl-replays/screensim-gemini-u0-vs-u12/).

The CookSim full-suite baseline contains all 150 scenarios. Its currently
published update-12 export contains 149/150 and remains labeled partial; it must
not be reported as a complete aggregate.
