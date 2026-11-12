# CookSim RL trace analysis for the ICLR reviews

**Finding.** Update 12 is not a uniformly quieter policy. On the 90 complete-trace validation episodes per checkpoint it produced **more assistant turns** (13,191 versus 12,701) and **more nonempty utterances** (12,572 versus 11,780), but **27.0% fewer spoken words** (318,896 versus 436,563). This supports a narrower version of reviewer jwae's concern: the policy became more concise, and the observed success gain cannot yet be attributed to better guidance. Strict success rose only 58/90 to 61/90 in this evaluation; native valid detection and prevention fell. These are stochastic trajectories on 30 fixed validation scenarios repeated three times, not a new-task test or independent training seeds.

| Measure | Update 0 count | Update 0 rate | Update 12 count | Update 12 rate | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Strict success / episodes | 58/90 | 64.4% | 61/90 | 67.8% | Net +3 paired successes; 13 gains and 10 losses. |
| Assistant turns | 12,701 | — | 13,191 | — | More turns, so turn suppression alone does not explain success. |
| Nonempty assistant utterances | 11,780 | — | 12,572 | — | More speech events. |
| Spoken words | 436,563 | — | 318,896 | — | 27.0% fewer words; mean words per nonempty utterance 37.1 to 25.4. |
| False flags | 517 | — | 391 | — | Fewer speculative flags. |
| Offered planned errors / episodes | 54/90 | 60.0% | 56/90 | 62.2% | Exposure depends on the rollout. |
| Prevention / offered errors | 22/54 | 40.7% | 14/56 | 25.0% | Native `injections[].prevented`. |
| Native valid flag / offered errors | 22/54 | 40.7% | 15/56 | 26.8% | `report.flagged_valid`. |
| Current-upstream detection / offered errors | 23/54 | 42.6% | 16/56 | 28.6% | Adds a 30-tick post-commit grace period. |
| Committed errors / offered errors | 33/54 | 61.1% | 36/56 | 64.3% | A worse descriptive error trajectory. |
| Success after committed error | 25/33 | 75.8% | 24/36 | 66.7% | Outcome after commitment, **not** a direct measure of corrective guidance. |

The error-aligned windows show the same verbosity change near the intervention opportunity. Among offered-error episodes, the first 30 ticks after release contain **723 assistant messages and 25,965 words** at update 0 versus **728 messages and 19,083 words** at update 12. That is 35.9 versus 26.2 words per message. The preceding 30-tick windows contain 672 messages/23,293 words versus 711 messages/17,992 words. Message counts are similar or higher, while utterances are shorter both before and after error release. The release windows are different stochastic episodes; these comparisons are descriptive.

Among the **46 scenario-pass pairs with an offered error at both checkpoints**, prevention changed from yes to no in 13 pairs and no to yes in six (five both yes, 22 both no). Native valid flag changed from yes to no in 13 and no to yes in seven (five both yes, 21 both no). The upstream detection rescore, which includes a 30-tick post-commit grace period, separately found 14 lost detections and eight gained. The differing native and rescore tallies should not be mixed. Two illustrative paired IDs with **success gains but lost native detection** are pass 1 `s1_b3_hard_nops_medium_map_2_leave_unattended` and pass 3 `s1_b3_hard_nops_hard_map_1_serve_raw`; they are useful for qualitative review but do not establish a beneficial mechanism. Conversely, pass 1 `s1_b3_easy_nops_medium_map_4_omit_subgoal` lost both success and native detection.

### Interaction-quality axes

Each cell below is the round-count-weighted mean of saved per-episode category aggregates from the official two-judge AQ artifacts. These axes are **not** interchangeable with strict task success; applicability differs by rubric item, and `listens` in particular can be sparse.

| AQ axis | Update 0 | Update 12 | Change |
| --- | ---: | ---: | ---: |
| Factual grounding (`truthful`) | 43.0% | 52.2% | +9.3 pp |
| Situational relevance (`sensible`) | 67.2% | 65.3% | −1.8 pp |
| Actionable guidance (`helpful`) | 35.9% | 38.8% | +2.9 pp |
| User intent uptake (`listens`) | 30.6% | 34.4% | +3.8 pp |
| Conciseness (`economy`) | 38.7% | 46.9% | +8.1 pp |

The largest gains are factual grounding and conciseness; situational relevance declines slightly. This is consistent with an AQ gain driven substantially by shorter, more grounded statements, not proof of better error intervention. The official aggregate AQ rises 24.31% to 30.17% as a mean of three pass-level scores. The axis numbers above are pooled by round and therefore use a different aggregation rule from that headline.

### What remains unresolved

The saved `conversation` array **does** preserve append order within a tick. The rollout code appends the assistant record before requesting and appending the next simulated-human response. In all **22/22** offered-error episodes with a native valid flag at update 0 and **15/15** at update 12, the first human record after that flagged assistant entry is an `accept` move, including when both share the same tick. The first subsequent logged action is labeled `vlm_fix:literal` in all 37 cases; it completed in 22/22 update-0 and 14/15 update-12 episodes (the remaining action was superseded). Thus the **logged dialogue-to-action sequence** can be reconstructed. It is a simulator-protocol observation, not evidence that real users would comply, nor proof that the warning was necessary for success. Actions may also have been queued or influenced by earlier guidance, and `accept` is a simulator move label rather than an independent human judgment.

For a qualitative check, [12 matched offered-error pairs](blinded_annotation_samples.jsonl) were selected deterministically, with three each from the four success-transition strata (both fail, update 0 only, update 12 only, both succeed). The two arms are randomly swapped within each pair; checkpoint labels, success outcomes, and detection verdicts are withheld from the annotation file and kept in a separate [coordinator key](blinded_annotation_key.json). This is a **balanced diagnostic sample**, not a prevalence sample. Annotators need relevant saved decision frames before scoring visual grounding; the text excerpts alone support only dialogue-level observations. The [annotation protocol](ANNOTATION_PROTOCOL.md) specifies independent ratings for warning timing, actionability, user response, and post-error repair. No human ratings have been collected yet. This analysis also does not isolate the effect of the reward terms; the planned matched-budget reward ablation is required.

### Reproducibility and sources

Run `python3 analyze_traces.py` in this directory. It reads the 180 native reports and 180 official paired AQ artifacts using the [manifest](/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-validation-aq-20261019/aq-inputs/manifest.jsonl), then writes [episode metrics](episode_metrics.jsonl) and [summary JSON](summary.json). `spoken_words` uses a whitespace-independent regex word count over the assistant text in `report.conversation`; the native turn count equals the number of assistant conversation entries and matches the decision records. Event windows are `[release−30, release)` and `[release, release+30)` in simulation ticks. This rescore uses only existing data and no paid judge calls. For current-upstream detection, see the separate [official rescore](/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-validation-aq-20261019/current-upstream-detection-summary.json). The effective user model in this October run was routed from requested Gemini 3.7 Flash to Gemini 3.8 Flash, so do not combine these figures with the September full-suite evaluation.
