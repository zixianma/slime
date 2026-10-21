# CookSim RL with the classic-novice Gemini user

The Qwen3.5-4B assistant's **strict task success increased at update 12** on the frozen 150-scenario suite: 32.44% to 42.22% overall across three passes (+9.78 percentage points). Train and validation both increased on the same frozen scenarios. A newer, independently sampled three-pass validation evaluation with complete interaction records succeeded on 58/90 episodes at update 0 and 61/90 at update 12. Its two-judge interaction-quality (AQ) scoring is in progress. This is a small proof-of-concept RL run, not evidence of broad task generalization.

## New complete-trace validation evaluation

Each checkpoint was evaluated once per pass on the same 30 held-out development-validation scenarios, for three stochastic passes and 90 episodes per checkpoint. Every episode has a native report, assistant-decision record, and training audit; all six passes passed strict trace verification. The numbers below describe this **new validation-only evaluation**, distinct from the earlier full-suite results in the next section.

| Update | Pass 1 successes / episodes | Pass 2 successes / episodes | Pass 3 successes / episodes | Total successes / episodes | Pooled success | Δ pooled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 20/30 | 21/30 | 17/30 | 58/90 | 64.4% | — |
| 12 | 19/30 | 23/30 | 19/30 | 61/90 | 67.8% | +3.3 pp |

The comparison reuses the same scenario IDs but samples new stochastic episodes. The split has 120 training and 30 development-validation scenarios and **no separate frozen test split**; the validation cases are held-out parent variants from recipe/layout cells also represented in training. Requests for the pinned `gemini-3.7-flash` user were routed to effective `gemini-3.8-flash` in this new evaluation, so these success rates should not be combined with the earlier September evaluation as if they shared the same user-model version. All 180 detailed episodes are linked from `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-validation-aq-20261019/RUN_PLAN.md`; two-judge AQ on those episodes is pending.

## Full-suite task success: update 0 versus 12

Each pass evaluates one stochastic episode for each frozen scenario at each checkpoint. The suite contains 120 training and 30 held-out validation scenarios. Success is the native CookSim `won` outcome. The table reports the mean and **sample SD** of the three pass success percentages. Δ is update 12 minus update 0, calculated before rounding.

| Split | Update | Successes / episodes | Scenarios / pass | Mean success ± SD | Δ mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0 | 116/360 | 120 | 32.22% ± 4.28 pp | — |
| Train | 12 | **152/360** | 120 | **42.22% ± 2.93 pp** | **+10.00 pp** |
| Validation | 0 | 30/90 | 30 | 33.33% ± 3.33 pp | — |
| Validation | 12 | **38/90** | 30 | **42.22% ± 6.94 pp** | **+8.89 pp** |
| Overall | 0 | 146/450 | 150 | 32.44% ± 4.07 pp | — |
| Overall | 12 | **190/450** | 150 | **42.22% ± 2.52 pp** | **+9.78 pp** |

The passes reuse the same scenario catalog, so SD measures stochastic rollout variation on fixed tasks, not uncertainty over unseen tasks. The validation scenarios are held-out parent variants from the same recipe/layout condition cells; this is development validation, not an unseen-domain test. The user and policy are stochastic, and three passes are too few for a strong statistical claim.

## Interaction quality: preliminary Gemini-only judgment

AQ scores the assistant's utterances against the observed kitchen state and the cook's needs, independently of task success. CookSim's native five-category rubric grades factual grounding, situational relevance, actionable guidance, user intent uptake, and conciseness; a failed factual-grounding gate zeros a round. The official AQ averages independent Gemini and GPT judgments per round. **Only Gemini judgments are complete for this comparison.**

Usable reports and decision traces exist at both checkpoints for **104 matched scenario-pass pairs**, of the 450 full-suite episode instances per checkpoint. This selected subset cannot estimate full-suite AQ. It contains 95 train and only nine validation pairs; the validation-only row below is a descriptive pooled score, not a three-pass estimate.

| AQ scope | Matched / full episode instances | Passes covered | Assistant rounds u0 | Assistant rounds u12 | Gemini AQ u0 | Gemini AQ u12 | AQ Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 95/360 | 3/3 | 6,521 | 6,094 | 23.5% ± 0.3 pp | 32.5% ± 1.9 pp | +9.0 pp |
| Validation only, partial | 9/90 | 2/3 | 559 | 416 | 27.1% (pooled) | 31.4% (pooled) | +4.3 pp |
| Train + validation | 104/450 | 3/3 | 7,080 | 6,510 | 23.7% ± 0.5 pp | 32.5% ± 1.8 pp | +8.8 pp |

For train and train + validation, each pass pools all assistant rounds in its matched episodes; the table gives the mean and **population SD** of the three pass-level scores. Pair coverage differs by pass (48, 42, 14), so the SD also reflects changing scenario composition. The validation-only row instead pools all 559 update-0 and 416 update-12 graded rounds across its nine matched pairs: eight pairs in pass 1, one in pass 2, and none in pass 3. Its +4.3 pp difference is **not comparable as a three-pass AQ estimate** and is highly sensitive to the available cases and episode lengths. All 208 paired episodes and 13,590 assistant rounds received Gemini verdicts. The exact agent-visible image was recovered for every graded decision; the upstream scorer used its recorded-evidence fallback for kitchen state. The missing GPT component prevents an official CookSim AQ claim.

<details>
<summary>Per-pass results and validation AQ coverage</summary>

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
