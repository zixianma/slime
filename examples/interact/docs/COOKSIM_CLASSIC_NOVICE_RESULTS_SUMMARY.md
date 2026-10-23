# CookSim RL with the classic-novice Gemini user

The Qwen3.5-4B assistant's **strict task success increased at update 12** in the original three-pass, 150-scenario evaluation: 32.44% to 42.22% overall (+9.78 percentage points), with increases on both train and validation. A newer, independently sampled validation-only evaluation with complete interaction records increased from 64.44% to 67.78% (+3.33 pp); its full two-judge interaction-quality (AQ) mean across three passes increased from 24.31% to 30.17% (+5.86 pp), while detection under the current upstream metric fell from 23/54 (42.6%) to 16/56 (28.6%). These are small proof-of-concept results on fixed scenarios, not evidence of broad task generalization.

## Task success, interaction quality, and detection

Task success and detection use three stochastic passes of fixed scenario IDs at each checkpoint. Success is CookSim's native `won` outcome. The **original full-suite** run evaluates 120 train and 30 held-out development-validation scenarios per pass. Its AQ is a *selected, matched Gemini-only subset*: 95 of 360 train and nine of 90 validation episodes per checkpoint have usable traces, so these AQ cells do not estimate full-split quality. The **new complete-trace** run reevaluates only the 30 validation scenarios per pass, with native reports and assistant decisions for all 180 episodes; Gemini and GPT scored all 11,151 assistant rounds. Its detection cells use the current upstream metric, while the original full-suite detection cells remain an audit-based proxy because most detailed traces are unavailable. The two protocols also differ in stochastic rollouts and effective user-model version and must not be treated as one pooled experiment.

| Evaluation | Split | Update | Successes / episodes | Success, mean ± SD | AQ-eligible episodes | AQ judge | AQ | Detected / offered errors | Detection recall, pooled | Detection, mean ± SD |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Original full suite | Train | 0 | 116/360 | 32.22% ± 4.28 pp | 95/360 | Gemini only | 23.5% ± 0.3 pp | 61/124 | 49.2% | 49.1% ± 7.0 pp |
| Original full suite | Train | 12 | 152/360 | 42.22% ± 2.93 pp | 95/360 | Gemini only | 32.5% ± 1.9 pp | 60/138 | 43.5% | 43.8% ± 6.8 pp |
| Original full suite | Validation | 0 | 30/90 | 33.33% ± 3.33 pp | 9/90 (2 passes) | Gemini only | 29.6% ± 4.8 pp | 13/27 | 48.1% | 49.0% ± 11.9 pp |
| Original full suite | Validation | 12 | 38/90 | 42.22% ± 6.94 pp | 9/90 (2 passes) | Gemini only | 31.1% ± 0.4 pp | 10/29 | 34.5% | 34.9% ± 6.7 pp |
| Original full suite | Overall | 0 | 146/450 | 32.44% ± 4.07 pp | 104/450 | Gemini only | 23.7% ± 0.5 pp | 74/151 | 49.0% | 49.0% ± 4.4 pp |
| Original full suite | Overall | 12 | 190/450 | 42.22% ± 2.52 pp | 104/450 | Gemini only | 32.5% ± 1.8 pp | 70/167 | 41.9% | 42.1% ± 4.9 pp |
| New complete trace | Validation | 0 | 58/90 | 64.44% ± 6.94 pp | 90/90 | Gemini + GPT | 24.31% ± 2.05 pp | 23/54 | 42.6% | 43.0% ± 14.5 pp |
| New complete trace | Validation | 12 | 61/90 | 67.78% ± 7.70 pp | 90/90 | Gemini + GPT | 30.17% ± 0.27 pp | 16/56 | 28.6% | 28.7% ± 3.4 pp |

Task-success and detection SDs are **sample SDs** across three pass percentages. Pooled detection divides total detected by total offered errors; it differs slightly from the unweighted pass mean because denominators vary. AQ-eligible counts indicate available matched traces. Original full-suite AQ uses the **population SD** of pass-level round-pooled scores: three passes for train and overall, but only **two for validation** (eight matched episodes in pass 1, one in pass 2, none in pass 3). Its validation mean ± SD is descriptive and not comparable to three-pass variability; the corresponding nine-episode round-pooled AQ is 27.1% at update 0 and 31.4% at update 12. New complete-trace validation AQ is the mean and **sample SD** of three pass-level scores, each pooling that pass's assistant rounds. New complete-trace detection uses the current upstream rule: a correctly named error counts inside its window or within 30 ticks after commitment. Original full-suite recall uses the retained audit counters for all 900 episodes: its numerator counts a valid flag **or** prevention, so it is an audit-based proxy for strict in-window recall, not a full application of the current rule. The 369 original native reports still available match the audit counters exactly, and none show prevention without a valid flag; the other 531 episodes cannot be re-scored. Detection rates across the two evaluations therefore use different definitions, and changing error exposure across checkpoints makes each rate descriptive rather than an isolated measure of policy improvement. AQ measures utterance quality separately from task success using factual grounding, situational relevance, actionable guidance, user intent uptake, and conciseness; a failed factual-grounding gate zeros a round.

**Metric-version note.** These rollouts used CookBench commit `f6ea8d2` (2026-09-11), not the later upstream simulator. On 2026-10-23, `hellomuffin/cook-bench-engine` main was `467554acf25b2fab2b695fcb2ca14ed4fcae6683`; its `tools/eval_v2_sweep.py` is byte-identical to the available 2026-10-12 upstream snapshot. The new complete-trace headline rows apply that current metric to the saved reports. The original strict in-window valid-flag counts were **22/54 (40.7%) → 15/56 (26.8%)**; the current rule adds one late detection at each update. This is a post hoc metric rescore, not a rerun with the current simulator. The [rescore script](/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-validation-aq-20261019/rescore_detection_upstream.py) and [machine-readable summary](/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-validation-aq-20261019/current-upstream-detection-summary.json) record the code hash, per-pass counts, and paired-case audit.

## Reward design

For the `cooking_prevention_turns_v1` training reward,

$$
R = S + 0.30P - 0.02\min(F,15) - 0.05\min(T/200,1),
$$

where $S$ indicates task success, $P$ indicates prevention of the episode's planned error, $F$ is the number of false-positive flags, and $T$ is the number of assistant turns. Each training episode has exactly one planned error, although it may end before that error is offered. Task success remains dominant by construction: unsuccessful completed episodes score at most 0.30, while successful ones score at least 0.65. The prevention bonus encourages timely intervention; false-flag and turn penalties discourage speculative calls and unnecessarily long exchanges. A valid flag without prevention earns no separate bonus, so this reward does not directly optimize detection recall.

**Interpretation of the detection tradeoff.** In the complete-trace validation, update 12 improved task success and AQ but detected fewer offered errors than update 0 (16/56, 28.6%, versus 23/54, 42.6%). Strict in-window valid flags also fell (22/54 to 15/56), despite similar numbers of offered errors and longer conversations. Among 46 scenario-pass pairs with an offered error at both checkpoints, 14 lost a current-metric detection and eight gained one. Under the reward above, a more selective flagging policy could reduce false positives (517 to 391 across 90 episodes) and improve task completion while missing uncertain errors. This is a plausible explanation, not a causal finding: the stochastic rollouts do not establish that the reward caused the change in flagging behavior.

The development-validation cases are held-out parent variants from recipe/layout cells also represented in training; the RL split has **no separate frozen test set**. Requests for the pinned `gemini-3.7-flash` user were routed to effective `gemini-3.8-flash` in the new evaluation, so its absolute success rates are not directly comparable to September's. Repeated passes on fixed scenarios measure rollout variability, not uncertainty over new tasks. Three passes are too few for a strong statistical claim. Source artifacts are listed below.

<details>
<summary>Per-pass results and validation AQ coverage</summary>

The new complete-trace validation success counts, for passes 1–3 respectively, are **20/30, 21/30, 17/30** at update 0 and **19/30, 23/30, 19/30** at update 12. All six passes passed strict verification. The full AQ input set contains 90 paired scenario-passes and 11,151 assistant rounds; every round has both judge verdict records, and the official merger passed its 180-episode/11,151-round audit. Two silent rounds had a non-applicable GPT numeric score, so their merged AQ uses the Gemini score, following the scorer's aggregation rule. The round-pooled AQ, which weights longer passes more, is 24.28% at update 0 versus 30.17% at update 12 (+5.90 pp). The machine-readable official AQ summary is at `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-validation-aq-20261019/aq-merged/official-aq-summary.json`, alongside paired verdicts under `aq-merged/sq_v4/frames/classic_novice/`; the run plan is at `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-validation-aq-20261019/RUN_PLAN.md`.

| Pass | AQ u0 | Assistant rounds u0 | Detected / offered u0 | AQ u12 | Assistant rounds u12 | Detected / offered u12 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 22.56% | 1,917 | 10/19 (52.6%) | 30.00% | 1,932 | 5/17 (29.4%) |
| 2 | 26.57% | 1,781 | 5/19 (26.3%) | 30.04% | 1,858 | 6/19 (31.6%) |
| 3 | 23.82% | 1,701 | 8/16 (50.0%) | 30.48% | 1,962 | 5/20 (25.0%) |

Judge-specific pooled AQ is 18.69% (Gemini) and 29.86% (GPT) at update 0, versus 25.02% and 35.31% at update 12; the improvement is present under either judge. The tracked simulator and grading API spend was $493.37, within the approved $575 total cap.

The original selected-subset Gemini AQ covers 104 matched pairs per checkpoint (7,080 and 6,510 assistant rounds at updates 0 and 12, respectively). Its overall AQ row does not estimate full-suite AQ.

| Pass | Train success u0 | Train success u12 | Validation success u0 | Validation success u12 | Overall success u0 | Overall success u12 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 33/120 | 47/120 | 9/30 | 12/30 | 42/150 | 59/150 |
| 2 | 43/120 | 54/120 | 11/30 | 11/30 | 54/150 | 65/150 |
| 3 | 40/120 | 51/120 | 10/30 | 15/30 | 50/150 | 66/150 |

Original full-suite detection counts by pass are **17/38, 24/42, 20/44** on train and **5/8, 4/10, 4/9** on validation at update 0; at update 12 they are **22/46, 20/42, 18/50** on train and **3/8, 3/11, 4/10** on validation. Headline recall pools detected and offered errors over the three passes, rather than averaging rates with different denominators.

The standard errors of mean task success are 2.47, 1.92, and 2.35 pp at update 0 for train, validation, and overall; at update 12 they are 1.69, 4.01, and 1.46 pp. Each is sample SD divided by √3. These describe repeated rollout variation on the frozen catalog, not independent scenario sampling.

| Pass | Matched train AQ pairs | Matched validation AQ pairs | Train AQ u0 | Train AQ u12 | Combined AQ u0 | Combined AQ u12 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40 | 8 | 23.7% | 30.3% | 23.9% | 30.4% |
| 2 | 41 | 1 | 23.7% | 34.9% | 24.1% | 34.8% |
| 3 | 14 | 0 | 23.0% | 32.2% | 23.0% | 32.2% |

For the eight matched validation scenarios in pass 1, Gemini AQ is 24.8% at update 0 and 31.5% at update 12. Pass 2 has only one matched validation scenario (34.4% versus 30.7%); pass 3 has none. The headline two-pass mean weights these passes equally, whereas round-pooled AQ weights their graded assistant rounds. Neither estimates full-validation AQ or a three-pass validation SD.

</details>

<details>
<summary>Other simulator diagnostics</summary>

**New complete-trace validation detection audit.** On the 46 matched scenario-pass pairs with an offered error at both checkpoints, current-metric detection was 19 at update 0 and 13 at update 12. Across all offered errors, the largest label-level declines were `leave_unattended` (7/8 to 4/8) and `forget_next_step` (4/5 to 1/6); `mis_prep` stayed at 6/12. Assistant turns rose from 12,701 to 13,191 across the 90 episodes, ruling out shorter conversations as a simple explanation for the lower recall. The ten held-out parent cases limit generalization beyond this validation catalog.

The following are pooled over 450 episodes per checkpoint. Mistake exposure and episode length depend on the policy rollout, so these metrics are descriptive rather than controlled substitutes for task success.

| Update | Prevented / offered errors | Prevention % | False flags | False flags / episode | Assistant turns | Turns / episode | Invalid responses / turns | Invalid % | Timeouts / episodes | Timeout % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 73/151 | 48.3% | 1,076 | 2.39 | 70,582 | 156.85 | 4,335/70,582 | 6.1% | 253/450 | 56.2% |
| 12 | 68/167 | 40.7% | 810 | 1.80 | 68,281 | 151.74 | 1,571/68,281 | 2.3% | 213/450 | 47.3% |

At update 12, timeouts, false flags, and invalid responses fall, while prevention per offered error also falls. The changing number of offered errors and other rollout differences limit causal interpretation of these diagnostics.

</details>

<details>
<summary>Training protocol and source artifacts</summary>

The assistant is Qwen3.5-4B with a frozen vision tower, trained with GRPO. The engine-native `GeminiHuman` uses `gemini-3.7-flash` with the `classic_novice` persona. Training uses 120 parent-disjoint single-error scenarios derived from `cases_composite_v4.json` (the file resides under the engine's `bench/v5` directory). These results are for the v4 case suite and do not require rerunning to remain valid on that suite; evaluating the redesigned v5 cases would be a separate experiment. One update retains four completed attempts for each of six scenario groups (24 episodes); five attempts are launched per group and the first four valid completions are used. The reward formula and rationale appear above the diagnostic sections.

The first-four-of-five collector may favor shorter episodes. Online development validation samples 10 scenarios twice at updates 0, 3, 6, 9, and 12; the **full-suite comparison above** instead uses all 120 train and 30 frozen validation scenarios once per pass. See the [CookSim Gemini handoff](COOKING_GEMINI_HANDOFF.md) and [full-suite evaluation protocol](FULL_SUITE_EVAL.md) for setup and checkpoint accounting.

Verified task-success source: `/gpfs/scrubbed/zixianma/checkpoints/web/full-suite-eval-20260920/repeated-full-suite-summary.json`, its six `cooking/<pass>/audit/result.json` files, and `cooksim-classic-novice-three-pass-table.md` in that directory. The source records were checked for 150 completed scenarios in each pass. AQ source: `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-interaction-quality-20261012/gemini-paired-summary.json`, `interim-results.md`, and the paired verdict artifacts there. The [published CookSim replay](https://zixianma.github.io/interact-rl-replays/cooking-u0-vs-u12/) is useful for trajectory inspection but its older displayed update-12 export contains only 149 of 150 scenarios; the verified tables here use complete repaired passes.

</details>
