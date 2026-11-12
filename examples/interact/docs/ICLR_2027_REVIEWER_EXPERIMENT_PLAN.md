# ICLR 2027 reviews: RL evidence and proposed experiments

**Current execution and results:** [RL rebuttal status, 12 November 2026](ICLR_2027_RL_REBUTTAL_STATUS_20261112.md). The dated launch notes below are historical snapshots; use the linked status page for current completed, pending, and blocked work.

The reviewers' central RL question is whether training produces better interactive guidance that transfers beyond the training user and scenarios. The highest-value response is to (1) analyze the existing paired CookSim traces at error opportunities, (2) evaluate the same frozen checkpoints with a different user and genuinely unseen tasks and error schedules, and (3) ablate the reward. These are proposed experiments, not completed results. Reviewer excerpts below are verbatim from the three reviews supplied by the authors for [Submission 13980](https://openreview.net/forum?id=ZGrIJ7f8rl).

**Execution status, 2026-11-10.** The no-cost [complete-trace analysis](/gpfs/scrubbed/zixianma/interact/rl-review-analysis-20261110/REPORT.md) is complete, including a prepared blinded annotation sample. The [cross-user dataset and transfer backend](/gpfs/scrubbed/zixianma/interact/reviewer_rl_transfer_20261110/TRANSFER_READINESS.md) and [four-arm reward ablation](/gpfs/scrubbed/zixianma/interact/slime/examples/interact/REVIEWER_REWARD_ABLATION.md) are prepared and mock/unit tested. A [new 19-case catalog](/gpfs/scrubbed/zixianma/interact/rl-review-heldout-cooksim-20261110/REPORT.md) uses four built-in recipes absent from RL training; all 19 cases passed a native scripted loader audit, while paired checkpoint evaluation remains. [Bounded launch plans](/gpfs/scrubbed/zixianma/interact/reviewer_rl_transfer_20261110/LAUNCH_GATES.md) cover the second persona, v5 error shift, and one genuinely recipe-disjoint case. The prior successful six-job validation used 36.9 H200 GPU-hours and $223.41 in simulator calls, giving a basis for the new baseline-persona plan below. One strictly matched seed of four reward arms is about 372 H200 GPU-hours before failures, frozen evaluation, and API calls. The v5 catalog tests changed errors but reuses the trained recipes and layouts.

The first wave was approved on 2026-11-10 with a **72 H200 GPU-hour and $440 Gemini-user API ceiling**: baseline-persona three-pass evaluation (48 H200-hours, $300), v5 error-shift pilot (4, $30), recipe-disjoint pilot (4, $30), and one-update reward-cost pilot (16, $80). Slurm jobs `447361` (baseline stage 1), `447362` (v5 pilot), `447363` (recipe-disjoint pilot), and `447360` (reward pilot) were submitted under the `vlm` account and were pending at the first scheduler check. Baseline stage 2 depends on verified completion of stage 1; its remaining allocation is included in the 48-hour cap. These runs cover pilots and a second-persona evaluation, not the full reward ablation, independent seeds, full new-task suite, alternative user-model family, or AQ rejudging. Supervisors must count failed or canceled runtime against the same caps.

CPU-only dependent jobs `447376` (baseline stage-2 gate), `447378` (v5 verifier), `447379` (recipe-disjoint verifier), and `447383` (reward-pilot verifier) were also submitted. They can verify artifacts and, for the baseline gate, advance a successful first stage; they do not provide autonomous diagnosis and repair of unexpected failures. Initial queue estimates were around 2026-11-13 and may move. No GPU time or Gemini-user API spend had accrued at that check.

On 2026-11-10, the user requested `vlm` urgent QOS. The `vlm` association permits `urgent`; Slurm reports the same usage factor as `normal`. All eight queued first-wave jobs were updated in place to `urgent`, and future stage-2 submissions are configured to request it. Immediately afterward, all four GPU studies remained pending but scheduler-projected starts moved to the evening of 2026-11-10. These are estimates, not allocations or results.

**Budget clarification, 2026-11-11.** The user approved **up to $500 in aggregate Gemini API credits across all rebuttal experiments**, inclusive of prior usage and outstanding reservations. This replaces the $440 aggregate API ceiling; it is not an additional $500. The existing experiment-specific API limits ($300 + $30 + $30 + $80 = $440) and the 72 H200 GPU-hour ceiling remain in force. The remaining $60 is unassigned API headroom, not authorization for additional GPU allocation or an unspecified experiment. Each study must retain a durable ledger; aggregate accounting must include failed attempts and unsettled reservations before further API allocation. The Gemini API has reported `gemini-3.8-flash` as the effective user model even where `gemini-3.7-flash` was requested, so paper claims must state the effective model.

## Evidence already available

| CookSim evaluation | Split | Update 0 successes / episodes | Update 0 success | Update 12 successes / episodes | Update 12 success | Comparability limit |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Original full suite | Validation | 30/90 | 33.33% | 38/90 | 42.22% | Three stochastic passes of 30 fixed cases; original engine and effective user model. |
| New complete-trace evaluation | Validation | 58/90 | 64.44% | 61/90 | 67.78% | Three new passes of the same 30 case IDs; changed rollout conditions and effective user model. |

The complete-trace evaluation also found two-judge interaction quality of 24.31% at update 0 and 30.17% at update 12, while current-metric detection recall fell from 23/54 (42.6%) to 16/56 (28.6%). A subsequent [audit of the saved traces](/gpfs/scrubbed/zixianma/interact/rl-review-analysis-20261110/REPORT.md) found more assistant turns (12,701 to 13,191) but 27.0% fewer spoken words (436,563 to 318,896). Thus a change in verbosity remains a plausible part of reviewer jwae's concern; the small success gain alone does not establish better guidance. The two evaluation rows use different stochastic rollouts and effective user-model versions; they must not be pooled or treated as replication of one effect. CookSim validation holds out parent variants but shares recipe/layout cells with training. See the [CookSim results and metric definitions](COOKSIM_CLASSIC_NOVICE_RESULTS_SUMMARY.md).

ScreenSim already has a task-disjoint frozen test set. Across four 12-update training conditions, none shows a clear scenario-weighted frozen-test success gain; see the [ScreenSim results](SCREENSIM_NEW_TASK_RESULTS_SUMMARY.md). It should be reported as a limitation, not as positive evidence of RL transfer. Three evaluation passes of one checkpoint measure rollout variability, not variation across independent RL training runs.

## Reviewer comments and matching experiments

### 1. Does the policy guide better, or just optimize speech penalties? — highest priority

> **Reviewer jwae:** “When RL is used to improve the assistant, the reward penalizes false flags and turn count, so the gain could come from talking less rather than from guiding better.”

> **Reviewer jwae:** “How do you rule out that the gain comes from talking less rather than guiding better?”

**Existing-trace analysis, partly complete.** The [reproducible audit](/gpfs/scrubbed/zixianma/interact/rl-review-analysis-20261110/REPORT.md) measures speech, error-aligned windows, detection/prevention, false flags, and five AQ axes on all 90 matched validation scenario-pass IDs per checkpoint. It finds 13 native detections lost versus seven gained among 46 pairs with an error offered at both checkpoints, and more concise utterances before and after error release. Conversation append order shows that every episode with a native valid flag has a subsequent simulated-user `accept` move and `vlm_fix:literal` action. This is simulator-protocol behavior, not proof of real-user compliance or a causal path to success. A [blinded 12-pair annotation sample](/gpfs/scrubbed/zixianma/interact/rl-review-analysis-20261110/ANNOTATION_PROTOCOL.md) is prepared but unrated. Any outcome contrast is descriptive because the policies induce different trajectories and error exposure.

**Reward ablation, new RL runs.** With the same initial checkpoint, training cases, user model, rollout budget, and independent seeds, compare (a) success-only reward, (b) success plus prevention, (c) success plus prevention minus false-flag penalty, and (d) the full reward including turn penalty. Evaluate all checkpoints under one frozen protocol. This isolates whether the turn and false-flag terms drive success, brevity, detection recall, or AQ. Report all outcomes rather than selecting the highest success checkpoint post hoc.

### 2. Does the gain transfer to another user or different scenarios? — highest priority

> **Reviewer jwae:** “The authors may also consider evaluating the trained policy against a second user or, even better, with participants to show its improvement. Testing the trained policy on a different scenario could also be helpful to show its generalizability, as it is impossible to cover all the scenarios from the real-world in the simulator during training.”

> **Reviewer jwae:** “Also, it forces the improvement on the same frozen scenarios.”

> **Reviewer snbV:** “For the GRPO experiment, does the learned improvement persist under a different simulated-user model or on scenarios with different error distributions?”

> **Reviewer jwae:** “Does the trained policy generalize to held-out scenarios, a second simulator, or human participants?”

**Frozen-checkpoint transfer matrix, no new RL run.** Evaluate CookSim update 0 and update 12 on the same predeclared cases with (a) the training persona, (b) a second persona, and (c) if supported, a user model from a different model family. A persona change tests behavioral robustness; a different model family more directly tests dependence on the training simulator. Pin and record the *effective* model version and all prompts. Use identical case IDs and planned errors across checkpoints, with repeated stochastic passes. Report per-persona success, AQ dimensions, detection, false flags, and speech; never collapse the user conditions into one number.

**New-task and error-shift test.** Freeze a CookSim test catalog whose recipes/task families are absent from RL training and validation; keep it untouched while choosing checkpoints and analysis rules. Within that catalog, predeclare an in-distribution error schedule and shifted error types, timing, or frequencies. Report success by task family and error type, with counts and percentages in separate columns. A new pass over the existing 30 validation IDs does **not** answer the new-task question. Do not claim a cross-user effect if only the persona prompt changes while the underlying VLM remains the same.

**Human transfer, conditional on ethics clearance.** A small counterbalanced within-participant comparison of frozen update-0 and update-12 assistants would directly test user behavior and task outcomes. Before recruitment, document ethics-board status or exemption, consent, compensation, and protection of free-text data as requested by reviewer jwae. This should not hold up the simulator and trace analyses.

### 3. Show the closed-loop mechanism after an error — high priority, mostly existing traces

> **Reviewer 13H4:** “However, the paper lacks explicit analysis or evaluation on these dynamics once an out-of-plan event is triggered, nor how different models differ in handling the evolving interaction.”

> **Reviewer 13H4:** “The reported metrics capture some of it, but they do not characterize, e.g., how recovery unfolds after an error commits, how guidance quality changes under persona push-back across turns, or how an assistant’s behavior shifts as the user diverges from the plan.”

Construct event-aligned trajectories for update 0 and update 12: event offered → assistant observation → warning/advice → user response/action → environment state → recovery or failure. Quantify transition probabilities and elapsed ticks at each stage, stratified by error type and persona. Include post-commit recovery and user push-back where recorded. Put several complete, paired examples next to the aggregate; do not infer a mechanism solely from a favorable example.

### 4. Establish reproducibility of the RL effect — high priority if making a training claim

The reviews do not explicitly request multiple training seeds, but this is necessary to distinguish a reproducible learning effect from a favorable checkpoint. Train at least three independent seeds for the selected CookSim condition under the same budget and evaluate each on the same transfer matrix. Report the distribution of *per-seed* update-12 minus update-0 effects. For uncertainty over tasks, resample or cluster by task/scenario, not by individual repeated rollout; three passes of the same fixed cases are not three independent task samples.

## Other reviewer requests that should accompany the RL response

| Reviewer comment (verbatim) | Concrete analysis or experiment |
| --- | --- |
| **snbV:** “How stable are assistant rankings under changes to the user simulator?” | Re-evaluate a small, preselected set of assistants under two user-model families and report rank changes with per-task uncertainty. This is broader than the update-0/update-12 transfer test. |
| **jwae:** “Why a binary rubric? How well does the LLM judge agree with human raters?” | Blind human annotation of a stratified sample of utterances on each AQ axis; report per-axis agreement, disagreements, and results with an ordinal rubric. Keep axis scores separate from the aggregate. |
| **jwae:** “What motivated the five axes, and does varying each one produce a measurable change in simulated behavior?” | Perturb one persona axis at a time on matched scenarios and quantify predefined behaviors, including speech rate, response latency, action narration, warning compliance, and error rate. Compare those behaviors with the existing participant data where valid. |
| **13H4:** “It is recommended to report any-success (and ideally time-to-completion) alongside in-time success for all assistants in all three environments.” | Recompute and report any-success, in-time success, time-to-completion, and timeout counts under the same evaluation protocol. |
| **13H4:** “Since the ground-truth scene graph is available, an ablation giving the assistant partial or full environmental state would disentangle whether perception or task understanding is the bottleneck.” | On fixed cases, compare visual-only, partial-state, and full-state observation while holding the assistant and user fixed. |

## Execution order and reporting rule

1. **First:** audit and analyze the existing complete CookSim traces, including error-aligned dynamics and speech. This can answer part of the “talking less” concern without new rollouts.
2. **Next, in parallel where approved resources allow:** run the frozen-checkpoint cross-user/new-task transfer matrix and prepare the reward ablation. Pin one evaluation protocol before looking at results.
3. **Then:** complete the reward ablation and independent-seed runs if the transfer results justify a stronger RL claim. Treat human transfer as a separate, ethics-dependent study.

The rebuttal should distinguish what is established from what remains open: the original CookSim run shows improvement on its fixed suite, the complete-trace validation shows a smaller success gain with higher AQ and lower detection recall, and ScreenSim does not show a clear frozen-test success gain. No single aggregate score establishes broadly transferable interactive assistance.
