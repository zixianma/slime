# CookSim RL with the classic-novice Gemini user

The Qwen3.5-4B assistant's **strict task success increased at update 12** in the original three-pass, 150-scenario evaluation: 32.44% to 42.22% overall (+9.78 percentage points), with increases on both train and validation. A newer, independently sampled validation-only evaluation with complete interaction records increased from 64.44% to 67.78% (+3.33 pp); its full two-judge interaction-quality (AQ) mean across three passes increased from 24.31% to 30.17% (+5.86 pp). These are small proof-of-concept results on fixed scenarios, not evidence of broad task generalization.

## Task success, interaction quality, and detection

Each row uses three stochastic passes of fixed scenario IDs at one checkpoint. Success is CookSim's native `won` outcome. The **original full-suite** run evaluates 120 train and 30 held-out development-validation scenarios per pass. Its AQ is a *selected, matched Gemini-only subset*: 95 of 360 train and nine of 90 validation episodes per checkpoint have usable traces, so these AQ cells do not estimate full-split quality. The **new complete-trace** run reevaluates only the 30 validation scenarios per pass, with native reports and assistant decisions for all 180 episodes; Gemini and GPT scored all 11,151 assistant rounds. The two protocols differ in stochastic rollouts and effective user-model version and must not be treated as one pooled experiment.

| Evaluation | Split | Update | Successes / episodes | Success, mean ± SD | AQ-eligible episodes | AQ judge | AQ | Detected / offered errors | Detection recall |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Original full suite | Train | 0 | 116/360 | 32.22% ± 4.28 pp | 95/360 | Gemini only | 23.5% ± 0.3 pp | — | — |
| Original full suite | Train | 12 | 152/360 | 42.22% ± 2.93 pp | 95/360 | Gemini only | 32.5% ± 1.9 pp | — | — |
| Original full suite | Validation | 0 | 30/90 | 33.33% ± 3.33 pp | 9/90 | Gemini only | 27.1% pooled | — | — |
| Original full suite | Validation | 12 | 38/90 | 42.22% ± 6.94 pp | 9/90 | Gemini only | 31.4% pooled | — | — |
| New complete trace | Validation | 0 | 58/90 | 64.44% ± 6.94 pp | 90/90 | Gemini + GPT | 24.31% ± 2.05 pp | 22/54 | 40.7% |
| New complete trace | Validation | 12 | 61/90 | 67.78% ± 7.70 pp | 90/90 | Gemini + GPT | 30.17% ± 0.27 pp | 15/56 | 26.8% |

Task-success SD is the **sample SD** across three pass percentages. AQ-eligible counts indicate available matched traces. Original full-suite train AQ uses the **population SD** of three pass-level scores (unequal matched-pair coverage); its validation AQ pools nine pairs across only two passes, so it has no comparable pass-level SD. New validation AQ is the mean and **sample SD** of three pass-level scores, each pooling that pass's assistant rounds. Detection recall in the new run is the share of offered native error instances flagged validly; changing exposure (54 versus 56 errors) makes it descriptive rather than an isolated measure of policy improvement. Full original-run detection recall is unavailable because native reports were not retained for every episode. Dashes mark unavailable measurements. AQ measures utterance quality separately from task success using factual grounding, situational relevance, actionable guidance, user intent uptake, and conciseness; a failed factual-grounding gate zeros a round.

The development-validation cases are held-out parent variants from recipe/layout cells also represented in training; the RL split has **no separate frozen test set**. Requests for the pinned `gemini-3.7-flash` user were routed to effective `gemini-3.8-flash` in the new evaluation, so its absolute success rates are not directly comparable to September's. Repeated passes on fixed scenarios measure rollout variability, not uncertainty over new tasks. Three passes are too few for a strong statistical claim. Source artifacts are listed below.

<details>
<summary>Per-pass results and validation AQ coverage</summary>

The new complete-trace validation success counts, for passes 1–3 respectively, are **20/30, 21/30, 17/30** at update 0 and **19/30, 23/30, 19/30** at update 12. All six passes passed strict verification. The full AQ input set contains 90 paired scenario-passes and 11,151 assistant rounds; every round has both judge verdict records, and the official merger passed its 180-episode/11,151-round audit. Two silent rounds had a non-applicable GPT numeric score, so their merged AQ uses the Gemini score, following the scorer's aggregation rule. The round-pooled AQ, which weights longer passes more, is 24.28% at update 0 versus 30.17% at update 12 (+5.90 pp). The machine-readable official AQ summary is at `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-validation-aq-20261019/aq-merged/official-aq-summary.json`, alongside paired verdicts under `aq-merged/sq_v4/frames/classic_novice/`; the run plan is at `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-validation-aq-20261019/RUN_PLAN.md`.

| Pass | AQ u0 | Assistant rounds u0 | AQ u12 | Assistant rounds u12 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 22.56% | 1,917 | 30.00% | 1,932 |
| 2 | 26.57% | 1,781 | 30.04% | 1,858 |
| 3 | 23.82% | 1,701 | 30.48% | 1,962 |

Judge-specific pooled AQ is 18.69% (Gemini) and 29.86% (GPT) at update 0, versus 25.02% and 35.31% at update 12; the improvement is present under either judge. The tracked simulator and grading API spend was $493.37, within the approved $575 total cap.

The original full-suite overall success counts were 146/450 (32.44% ± 4.07 pp) at update 0 and 190/450 (42.22% ± 2.52 pp) at update 12. The original selected-subset Gemini AQ pooled train and validation was 23.7% ± 0.5 pp versus 32.5% ± 1.8 pp across 104 matched pairs per checkpoint (7,080 and 6,510 assistant rounds respectively). Those selected-subset scores do not estimate full-suite AQ.

| Pass | Train success u0 | Train success u12 | Validation success u0 | Validation success u12 | Overall success u0 | Overall success u12 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33/120 | 47/120 | 9/30 | 12/30 | 42/150 | 59/150 |
| 2 | 43/120 | 54/120 | 11/30 | 11/30 | 54/150 | 65/150 |
| 3 | 40/120 | 51/120 | 10/30 | 15/30 | 50/150 | 66/150 |

The standard errors of mean task success are 2.47, 1.92, and 2.35 pp at update 0 for train, validation, and overall; at update 12 they are 1.69, 4.01, and 1.46 pp. Each is sample SD divided by √3. These describe repeated rollout variation on the frozen catalog, not independent scenario sampling.

| Pass | Matched train AQ pairs | Matched validation AQ pairs | Train AQ u0 | Train AQ u12 | Combined AQ u0 | Combined AQ u12 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40 | 8 | 23.7% | 30.3% | 23.9% | 30.4% |
| 2 | 41 | 1 | 23.7% | 34.9% | 24.1% | 34.8% |
| 3 | 14 | 0 | 23.0% | 32.2% | 23.0% | 32.2% |

For the eight matched validation scenarios in pass 1, Gemini AQ is 24.8% at update 0 and 31.5% at update 12. Pass 2 has only one matched validation scenario (34.4% versus 30.7%); pass 3 has none. These descriptive values are not a validation-set AQ estimate.

</details>

<details>
<summary>Other simulator diagnostics</summary>

**New complete-trace validation detection audit.** The native engine offered 54 errors at update 0 and 56 at update 12; valid structured flags fell from 22 to 15. In the 46 matched scenario-pass pairs where an error was offered at both checkpoints, valid detection fell from 18 to 12: 13 pairs lost a detection and seven gained one. Of the 13 losses, six update-12 rollouts had no structured flag attached to the offered error and seven had only wrong-name flags. Across all offered errors, the largest label-level declines were `leave_unattended` (7/8 to 4/8) and `forget_next_step` (3/5 to 1/6); `mis_prep` stayed at 6/12. False-positive flags fell from 517 to 391 across the 90 episodes, while assistant turns rose from 12,701 to 13,191. This pattern points to fewer or misnamed flags, rather than less error exposure or shorter episodes, as the immediate reason measured recall fell. The stochastic trajectories and ten held-out parent cases do not isolate why the policy changed its flagging behavior.

The following are pooled over 450 episodes per checkpoint. Mistake exposure and episode length depend on the policy rollout, so these metrics are descriptive rather than controlled substitutes for task success.

| Update | Prevented / offered errors | Prevention % | False flags | False flags / episode | Assistant turns | Turns / episode | Invalid responses / turns | Invalid % | Timeouts / episodes | Timeout % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 73/151 | 48.3% | 1,076 | 2.39 | 70,582 | 156.85 | 4,335/70,582 | 6.1% | 253/450 | 56.2% |
| 12 | 68/167 | 40.7% | 810 | 1.80 | 68,281 | 151.74 | 1,571/68,281 | 2.3% | 213/450 | 47.3% |

At update 12, timeouts, false flags, and invalid responses fall, while prevention per offered error also falls. The changing number of offered errors and other rollout differences limit causal interpretation of these diagnostics.

</details>

<details>
<summary>Training protocol and source artifacts</summary>

The assistant is Qwen3.5-4B with a frozen vision tower, trained with GRPO. The engine-native `GeminiHuman` uses `gemini-3.7-flash` with the `classic_novice` persona. Training uses 120 parent-disjoint single-error scenarios. One update retains four completed attempts for each of six scenario groups (24 episodes); five attempts are launched per group and the first four valid completions are used. The shaped `cooking_prevention_turns_v1` reward is:

```text
task_success + 0.30 × any_error_prevented
             − 0.02 × min(false_flags, 15)
             − 0.05 × min(assistant_turns / 200, 1)
```

The first-four-of-five collector may favor shorter episodes. Online development validation samples 10 scenarios twice at updates 0, 3, 6, 9, and 12; the **full-suite comparison above** instead uses all 120 train and 30 frozen validation scenarios once per pass. See the [CookSim Gemini handoff](COOKING_GEMINI_HANDOFF.md) and [full-suite evaluation protocol](FULL_SUITE_EVAL.md) for setup and checkpoint accounting.

Verified task-success source: `/gpfs/scrubbed/zixianma/checkpoints/web/full-suite-eval-20260920/repeated-full-suite-summary.json`, its six `cooking/<pass>/audit/result.json` files, and `cooksim-classic-novice-three-pass-table.md` in that directory. The source records were checked for 150 completed scenarios in each pass. AQ source: `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-interaction-quality-20261012/gemini-paired-summary.json`, `interim-results.md`, and the paired verdict artifacts there. The [published CookSim replay](https://zixianma.github.io/interact-rl-replays/cooking-u0-vs-u12/) is useful for trajectory inspection but its older displayed update-12 export contains only 149 of 150 scenarios; the verified tables here use complete repaired passes.

</details>
