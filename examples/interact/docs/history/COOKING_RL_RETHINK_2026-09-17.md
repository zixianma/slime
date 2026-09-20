# Cooking RL rethink (2026-09-17)

> Historical design record. The redesign was implemented and superseded by the
> classic-novice Gemini-user setup in
> [COOKING_GEMINI_HANDOFF.md](../COOKING_GEMINI_HANDOFF.md). Do not launch jobs from
> IDs or paths in this document.

## What was stopped

The Qwen3.5-4B CookSim run was stopped during update 7. Jobs `299530`,
`299663`, and `299666` were cancelled. The last committed checkpoint is update
6; update-7 rollouts are cached, but its optimizer step did not complete. The
W&B point at train step 7 therefore describes rollouts from the update-6 policy,
not an update-7 checkpoint.

The frozen split has 40 training scenarios and 10 held-out development
validation scenarios. Each of the 10 recipe/layout cells contributes four train
variants and one validation variant. A training update sampled 6 scenario
groups x 8 attempts (48 episodes); validation used 10 groups x 4 attempts (40
episodes).

## Why the result is not convincing

Validation success changed from 1/40 at update 3 to 0/40 at update 6. More
importantly, valid error detections fell from 16/76 offered errors to 6/76.
False flags improved from 269 to 242, but invalid assistant outputs increased
from 147/4179 calls to 229/4234 calls. Every wrong-serve episode was missing at
least one required final ingredient.

The current cases are composite exams, not simple learning examples. Each case
contains 2--4 actual errors plus questions, proactive requests, and sometimes a
legitimate plan change or side quest. End-to-end success requires resolving all
consequential errors, so a modest event-level improvement can remain invisible
in binary success. The old reward

```text
success + 0.1 * detected/planned - 0.01 * min(false_flags, 10)
```

also under-rewards the causal chain from timely detection through human
acceptance to prevention.

## Agreed redesign direction

1. Start with single-error cases derived from the existing v5 sub-cases. Keep
   recipes/layout cells and scenario variants split before deriving examples,
   so related train and validation variants cannot leak across the boundary.
2. Preserve a separate composite validation set. Single-error validation
   measures whether the policy learned intervention skills; composite
   validation measures whether those skills compose.
3. Train on engine-verifiable stages: valid/timely detection, convincing the
   user, and preventing the error. Penalize false flags, invalid replies, and
   excessive assistant turns. Keep terminal task success as the dominant term.
4. Replace the scripted cook with the engine-native Gemini user using
   `gemini-3.7-flash`. This changes the environment distribution and incurs paid
   API calls, so it needs a new split/profile version and a separate W&B run.
5. Before a full RL run, validate a small matched rollout set for schema
   validity, user compliance, reward variance, API latency/failure rate, and
   wall-clock/GPU utilization. Do not interpret scripted-user and Gemini-user
   curves as one continuous experiment.

## Proposed reward: `cooking_single_error_gemini_v1`

For an episode with one offered error:

```text
R = success
  + 0.10 * valid_in_time_detection
  + 0.10 * user_convinced
  + 0.10 * error_prevented
  - 0.02 * min(false_flags, 5)
  - 0.10 * invalid_response_fraction
  - 0.10 * min(assistant_turns / 200, 1)
```

All components are computed from the native report. Detection, conviction, and
prevention are zero when the error was not offered; such episodes should
normally be excluded from a single-error training batch and surfaced as a data
quality metric. The shaping range is `[-0.30, 0.30]`, so every success scores at
least `0.70` and every failure at most `0.30`. This keeps task success strictly
dominant while supplying denser causal credit.

The turn penalty is normalized by the fixed 200-decision cap. Its maximum is
0.10, and one extra assistant turn costs 0.0005. This discourages drawn-out
interactions without making short recipes inherently preferable or making a
quick catastrophic termination attractive. Log the unshaped success, each
reward component, raw assistant turns, native ticks, and total shaped reward
separately.

## Retrospective weight validation

We subsequently simplified the candidate to reward only engine-verified
prevention and tested it on the 80 complete held-out episodes from policy weight
versions 4 and 7 (40 each):

```text
R = task_success
  + 0.30 * any_error_prevented
  - 0.02 * min(false_flags, 15)
  - 0.05 * min(assistant_turns / 200, 1)
```

The initially discussed `-0.10 * min(false_flags, 3)` was rejected: 86.25% of
the archived episodes hit its cap, leaving only 9 distinct rounded rewards and
a median reward of -0.30. With the revised `-0.02` cost, none of the 80 episodes
hit the 15-flag cap, there were 65 distinct rounded rewards, and every rollout
group had reward variance. In all 34 within-checkpoint, within-scenario pairs
where prevention differed, the prevented rollout ranked higher; the median
margin was about 0.32. Prevention itself varied in 50% of groups (60% at weight
version 4 and 40% at version 7), while the complete reward varied in 100%.

The revised reward is also bounded so that success remains dominant. Given the
200-turn and 15-flag caps, a failure scores at most 0.30 and a success at least
0.65. These results validate the scalarization and ranking behavior only on the
archived scripted-user composite data. A Gemini-user, single-error smoke is
still required to validate reward frequencies, API behavior, and whether the
policy can causally affect turn count. No Gemini credentials were available in
the inspected environment when this retrospective test was run.

The workspace credential was later found as `GOOGLE_API_KEY` in
`/gpfs/home/zixianma/interact/.env` and mapped to the engine's expected
`GEMINI_API_KEY` without logging its value. Two direct
`gemini-3.7-flash` probes, separated by the API-provided retry interval, both
returned HTTP 429. The reported exhausted quota was the 20-request free-tier
daily project/model quota. No Gemini-user rollout or GPU job was launched; a
quota-enabled key, billing-enabled project, or quota reset is required first.

The key was replaced and a direct model probe then succeeded. Frozen-policy job
`299896` completed one real Qwen3.5-4B-assistant / Gemini-3.7-Flash-user episode
in 6m15s. It used 28 assistant turns, produced no invalid assistant responses,
detected and prevented one of two offered composite errors, made one false flag,
and scored 0.273 under the proposed scalarization. Policy generation consumed
about 197s of the 343s episode runtime; Gemini/environment work accounted for
most of the remainder.

For training, the composite-parent split was frozen before deriving single-error
children. The resulting CPU-rendered Gemini split has 120 training cases and 10
balanced development-validation cases (one per recipe/layout cell), covering all
nine native error labels. A 30-case full held-out audit pool is also retained.
The first proper phase is three updates of 6 groups x 4 attempts (24 episodes),
with 10 x 2 baseline/final validation. The initial two-GPU job `300010` was
cancelled while still pending when the user requested a faster four-GPU run.
Replacement job `300014` requests four H200s, 32 CPUs, and a four-hour ceiling.
During rollouts one selected GPU renders and three host TP=1 policy servers;
after browsers exit, a hard handoff check requires no graphics processes before
all four GPUs perform the update as TP=2 x DP=2. It uses a fresh W&B run.

## Implemented outcome (September 19)

The final profile changed the user to `classic_novice`, extended the native base
budget to 3,600 seconds, kept the 200-decision cap, and added a guarded fourth
TP1 policy server on the renderer GPU. Five attempts are launched per group and
the first four valid completions are trained on. Baseline validation was 13/20
(65%, reward 0.606); update-3 validation was 16/20 (80%, reward 0.772). See the
handoff document for the maintained launcher and caveats.
