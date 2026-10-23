# ScreenSim RL on the new task library

On the primary scenario-weighted strict-success measure, **none of the four completed 12-update runs shows a clear frozen-test gain**. The only positive test change is one additional success in 90 rollouts for the classic-novice user without a manual. On the full training catalog, only baseline/no-manual increases in strict success (+1.0 percentage point); the other three methods decline. Baseline-user training without a manual improves development success by 3/96 rollouts over three passes, but its three-pass test result falls slightly. With a manual throughout, both user personas finish below their update-0 frozen-test results. GPT-judged interaction quality improves for baseline + manual on dev and test despite lower task success; baseline + no manual is nearly flat on dev and declines on test. Gemini judges baseline + no manual lower at update 12 on both splits.

## Strict success at updates 0 and 12

The assistant is Qwen3.5-4B and the simulated user is Gemini 3.7 Flash. “Manual” means the assistant receives the task manual in **training, development, and test**. Each method has separate update-0 and update-12 rows. Strict success requires the requested phone state by the deadline without lasting unrequested changes. Δ is update 12 minus update 0 in percentage points (pp), calculated before rounding; **bold ↑** marks a numerical gain, not a statistically established improvement. Dashes in update-0 delta cells mean the comparison is not applicable.

| Method (user; assistant manual) | Update | Train count | Train success, mean ± SD | Train Δ pp | Dev count | Dev success, mean ± SD | Dev Δ pp | Frozen-test count | Frozen-test success, mean ± SD | Test Δ pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 90/408 | 22.1% ± 2.0 pp | — | 23/96 | 24.0% ± 11.8 pp | — | 25/90 | 27.8% ± 6.9 pp | — |
| Baseline; no manual | 12 | **94/408** | **23.0% ± 1.8 pp** | **↑ +1.0** | **26/96** | **27.1% ± 9.5 pp** | **↑ +3.1** | 23/90 | 25.6% ± 1.9 pp | −2.2 |
| Baseline; manual | 0 | 85/408 | 20.8% ± 2.6 pp | — | 19/96 | 19.8% ± 1.8 pp | — | 25/90 | 27.8% ± 10.7 pp | — |
| Baseline; manual | 12 | 54/408 | 13.2% ± 0.7 pp | −7.6 | 6/96 | 6.3% ± 0.0 pp | −13.5 | 15/90 | 16.7% ± 3.3 pp | −11.1 |
| Classic novice; no manual | 0 | 11/408 | 2.7% ± 0.8 pp | — | 2/96 | 2.1% ± 1.8 pp | — | 3/90 | 3.3% ± 3.3 pp | — |
| Classic novice; no manual | 12 | 9/408 | 2.2% ± 0.0 pp | −0.5 | **4/96** | **4.2% ± 1.8 pp** | **↑ +2.1** | **4/90** | **4.4% ± 1.9 pp** | **↑ +1.1** |
| Classic novice; manual | 0 | 39/408 | 9.6% ± 2.0 pp | — | 4/96 | 4.2% ± 1.8 pp | — | 8/90 | 8.9% ± 1.9 pp | — |
| Classic novice; manual | 12 | 26/408 | 6.4% ± 1.8 pp | −3.2 | **6/96** | **6.3% ± 3.1 pp** | **↑ +2.1** | 5/90 | 5.6% ± 3.8 pp | −3.3 |

The SD is the **sample SD of pass-level success percentages**. Every development cell now uses three passes of 32 scenarios (96 rollouts); every frozen-test cell uses three passes of 30 scenarios (90 rollouts). Each available train comparison uses three passes of 136 scenarios (408 rollouts). Repeated passes revisit the same scenarios and therefore measure rollout variability, not uncertainty across new tasks. Differences of one or two successes are especially fragile. The two personas also differ in difficulty, so comparing their raw rates does not isolate the manual’s effect.

For baseline/no-manual development, the pass-level success counts are **5/32 → 8/32**, **12/32 → 12/32**, and **6/32 → 6/32**: the entire net gain comes from the original pass. The original pass used engine stamp `66ef7e6941`; the repeats use `a6249808e1` because the copied engine forces `manual.py` to return an empty string. The original function already returned an empty string for every task in this development split, and all other engine-stamped files match; the assistant had no manual in all three passes.

## Interaction quality across four conditions

**AQ measures assistant-utterance quality separately from task success.** ScreenSim's primary GPT-5.4-mini judge scored every saved dev and frozen-test episode in all four conditions using the native five-category rubric with a factual-grounding gate. The primary AQ rises for baseline + manual on both splits, while baseline + no manual is nearly flat on dev and declines on test. Classic novice + no manual declines; classic novice + manual changes little.

| User; assistant manual | Split | Scenarios / pass | Passes | Episodes per update | Assistant rounds u0 | Assistant rounds u12 | GPT AQ u0, mean ± SD | GPT AQ u12, mean ± SD | AQ Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | Dev | 32 | 3 | 96 | 1,269 | 1,325 | 30.0% ± 0.6 pp | 29.7% ± 2.9 pp | −0.3 pp |
| Baseline; no manual | Frozen test | 30 | 3 | 90 | 1,015 | 1,112 | 36.4% ± 1.7 pp | 31.4% ± 1.3 pp | −5.1 pp |
| Baseline; manual | Dev | 32 | 3 | 96 | 972 | 516 | 42.1% ± 2.8 pp | 49.8% ± 2.7 pp | +7.6 pp |
| Baseline; manual | Frozen test | 30 | 3 | 90 | 889 | 429 | 43.8% ± 2.3 pp | 50.4% ± 1.7 pp | +6.5 pp |
| Classic novice; no manual | Dev | 32 | 3 | 96 | 2,359 | 2,456 | 27.0% ± 3.2 pp | 22.1% ± 1.5 pp | −4.9 pp |
| Classic novice; no manual | Frozen test | 30 | 3 | 90 | 2,110 | 2,621 | 27.9% ± 3.1 pp | 26.9% ± 0.6 pp | −1.0 pp |
| Classic novice; manual | Dev | 32 | 3 | 96 | 1,958 | 2,372 | 40.2% ± 1.4 pp | 40.0% ± 1.3 pp | −0.2 pp |
| Classic novice; manual | Frozen test | 30 | 3 | 90 | 1,895 | 2,307 | 40.1% ± 1.9 pp | 41.1% ± 1.7 pp | +1.0 pp |

The independent Gemini judge gives a sensitivity check. The scorer requested Gemini 3.7 Flash, which [Google now routes to Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/changelog?release=09-18-2026); the effective judge version is therefore 3.8 for these judgments. Its values should not be treated as a pinned 3.7 replication.

| User; assistant manual | Split | Gemini AQ u0, mean ± SD | Gemini AQ u12, mean ± SD | AQ Δ |
| --- | --- | ---: | ---: | ---: |
| Baseline; no manual | Dev | 31.6% ± 1.1 pp | 29.1% ± 3.5 pp | −2.5 pp |
| Baseline; no manual | Frozen test | 40.5% ± 1.8 pp | 33.2% ± 1.7 pp | −7.3 pp |
| Baseline; manual | Dev | 44.9% ± 2.6 pp | 56.8% ± 4.1 pp | +11.9 pp |
| Baseline; manual | Frozen test | 52.0% ± 1.4 pp | 59.8% ± 0.9 pp | +7.8 pp |
| Classic novice; no manual | Dev | 23.5% ± 1.3 pp | 20.1% ± 2.1 pp | −3.4 pp |
| Classic novice; no manual | Frozen test | 25.5% ± 2.1 pp | 24.9% ± 0.5 pp | −0.6 pp |
| Classic novice; manual | Dev | 40.8% ± 1.5 pp | 40.2% ± 0.9 pp | −0.6 pp |
| Classic novice; manual | Frozen test | 39.2% ± 2.4 pp | 41.7% ± 2.2 pp | +2.5 pp |

AQ is the mean of pass-level scores, each pooling its rubric-applicable assistant rounds; SD is the **population SD** across three passes, and Δ uses unrounded means. Update 0 and 12 use the same scenario IDs within each pass; assistant-round counts differ with trajectory length. The original four-condition batch contains 1,360 judged episodes and 23,817 assistant rounds. The two additional baseline/no-manual dev passes at each update add 128 judged episodes and 1,788 assistant rounds, yielding 1,488 judged episodes and 25,605 assistant rounds across both batches. Both judges returned verdicts for all of these episodes and rounds. One GPT-judged round in the original batch had no applicable rubric item and is excluded from its AQ denominator; Gemini had an applicable score for every round. Complete verification, per-judge summaries, and verdicts are under `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-interaction-quality-nomanual-20261019/` and `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-baseline-nomanual-dev-repeats-20261021/`.

## No-manual users: native error detection

The native `detected` verdict is the union of the simulated person's call and the engine's flag verdict; it is **not** an assistant-only explicit-naming metric. Recall divides detected fired mistakes by all fired mistakes. Pass SD is the sample SD of the three pass-level recall percentages, while pooled recall uses the total counts; changing error exposure across rollouts limits causal interpretation.

| User | Split | Update | Detected / fired | Pooled recall | Pass mean ± SD | False flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | Dev | 0 | 125/137 | 91.2% | 91.2% ± 1.1 pp | 118 |
| Baseline | Dev | 12 | 133/140 | 95.0% | 94.9% ± 4.7 pp | 198 |
| Baseline | Frozen test | 0 | 129/151 | 85.4% | 85.4% ± 3.2 pp | 77 |
| Baseline | Frozen test | 12 | 134/151 | 88.7% | 88.7% ± 6.1 pp | 122 |
| Classic novice | Dev | 0 | 51/52 | 98.1% | 97.6% ± 4.1 pp | 288 |
| Classic novice | Dev | 12 | 40/43 | 93.0% | 93.3% ± 6.7 pp | 395 |
| Classic novice | Frozen test | 0 | 42/52 | 80.8% | 80.3% ± 10.4 pp | 203 |
| Classic novice | Frozen test | 12 | 52/61 | 85.2% | 85.1% ± 10.1 pp | 353 |

Classic-novice counts were recomputed from the fired beats and `detected` flags in all 96 development episodes per update and all 90 corrected no-manual frozen-test episodes per update. The development aggregate's older test episodes are excluded because that evaluation supplied a manual on test. The fired-error denominators differ between checkpoints because stochastic trajectories expose different opportunities.

The calculation matches the native `detected / n_fired` formula in `screensim/interact/composite_run_v2.py`. On 2026-10-23, this file, `screensim/interact/grade.py`, and `tools/sample_summary.py` were byte-identical to `hellomuffin/screensim-engine` main at `6f566fa7c47b3a4772909cae5e618c9085557d9c`. The saved episodes were generated earlier under their stamped engine revisions; this code check verifies the detection metric, not that the full simulator was rerun at the current upstream head.

## Other evaluation metrics

**Task-macro strict success** averages each task’s scenario success rate, then weights the tasks equally. It can move differently from the scenario-weighted primary metric because tasks have different numbers of scenarios.

| Method | Update | Train task-macro % | Train Δ pp | Dev task-macro % | Dev Δ pp | Frozen-test task-macro % | Test Δ pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 23.1% | — | 24.2% | — | 28.8% | — |
| Baseline; no manual | 12 | **23.9%** | **↑ +0.8** | **27.3%** | **↑ +3.0** | **31.8%** | **↑ +3.0** |
| Baseline; manual | 0 | 22.4% | — | 20.7% | — | 29.3% | — |
| Baseline; manual | 12 | 13.9% | −8.6 | 6.1% | −14.6 | 21.2% | −8.1 |
| Classic novice; no manual | 0 | 3.2% | — | 2.5% | — | 4.5% | — |
| Classic novice; no manual | 12 | 2.8% | −0.4 | **4.0%** | **↑ +1.5** | **7.1%** | **↑ +2.5** |
| Classic novice; manual | 0 | 10.2% | — | 4.5% | — | 13.1% | — |
| Classic novice; manual | 12 | 7.2% | −3.0 | **6.6%** | **↑ +2.0** | 6.6% | −6.6 |

**Goal attainment** checks whether the final phone state satisfies the requested goal, regardless of the deadline or lasting unrequested changes. It is therefore less stringent than strict success. Counts and percentages are kept in separate columns; `n/r` means the detailed diagnostic was not recorded, whereas `—` means no full train-split checkpoint evaluation was run.

| Method | Update | Train goal count | Train goal % | Train Δ pp | Dev goal count | Dev goal % | Dev Δ pp | Frozen-test goal count | Frozen-test goal % | Test Δ pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 124/408 | 30.4% | — | 28/96 | 29.2% | — | 31/90 | 34.4% | — |
| Baseline; no manual | 12 | **132/408** | **32.4%** | **↑ +2.0** | **41/96** | **42.7%** | **↑ +13.5** | 29/90 | 32.2% | −2.2 |
| Baseline; manual | 0 | 123/408 | 30.1% | — | 33/96 | 34.4% | — | 31/90 | 34.4% | — |
| Baseline; manual | 12 | 83/408 | 20.3% | −9.8 | 9/96 | 9.4% | −25.0 | 22/90 | 24.4% | −10.0 |
| Classic novice; no manual | 0 | 21/408 | 5.1% | — | 5/96 | 5.2% | — | 4/90 | 4.4% | — |
| Classic novice; no manual | 12 | **24/408** | **5.9%** | **↑ +0.7** | **6/96** | **6.3%** | **↑ +1.0** | 4/90 | 4.4% | 0.0 |
| Classic novice; manual | 0 | 62/408 | 15.2% | — | 12/96 | 12.5% | — | 14/90 | 15.6% | — |
| Classic novice; manual | 12 | **64/408** | **15.7%** | **↑ +0.5** | 12/96 | 12.5% | 0.0 | 14/90 | 15.6% | 0.0 |

**On-time goal success** requires the requested state by the simulator's deadline but permits lasting unrequested changes. It separates deadline misses from the extra-change failures excluded by strict success. The frozen-test count stays at 14/90 for classic novice with a manual while strict success falls from 8/90 to 5/90; more of its on-time goals therefore end with extra changes at update 12.

| Method | Update | Train count | Train on-time goal % | Train Δ pp | Dev count | Dev on-time goal % | Dev Δ pp | Frozen-test count | Frozen-test on-time goal % | Test Δ pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 115/408 | 28.2% | — | 27/96 | 28.1% | — | 30/90 | 33.3% | — |
| Baseline; no manual | 12 | 127/408 | 31.1% | +2.9 | **38/96** | **39.6%** | **↑ +11.5** | 29/90 | 32.2% | −1.1 |
| Baseline; manual | 0 | 120/408 | 29.4% | — | 33/96 | 34.4% | — | 30/90 | 33.3% | — |
| Baseline; manual | 12 | 79/408 | 19.4% | −10.0 | 8/96 | 8.3% | −26.0 | 21/90 | 23.3% | −10.0 |
| Classic novice; no manual | 0 | 20/408 | 4.9% | — | 5/96 | 5.2% | — | 4/90 | 4.4% | — |
| Classic novice; no manual | 12 | 23/408 | 5.6% | +0.7 | 6/96 | 6.3% | +1.0 | 4/90 | 4.4% | 0.0 |
| Classic novice; manual | 0 | 61/408 | 15.0% | — | 11/96 | 11.5% | — | 14/90 | 15.6% | — |
| Classic novice; manual | 12 | 62/408 | 15.2% | +0.2 | 12/96 | 12.5% | +1.0 | 14/90 | 15.6% | 0.0 |

<details>
<summary>Additional simulator diagnostics (update 0 → 12)</summary>

**Error prevention** is the number of fired mistakes detected without a lasting relevant effect, divided by all fired mistakes. Exposure varies with rollout behavior, so it is descriptive rather than a controlled success comparison. Counts are fired opportunities, not episodes. `n/r` means the per-episode report was not recoverable.

| Method | Update | Train prevented/fired | Train % | Train Δ pp | Dev prevented/fired | Dev % | Dev Δ pp | Frozen-test prevented/fired | Test % | Test Δ pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 431/600 | 71.8% | — | 93/137 | 67.9% | — | 98/151 | 64.9% | — |
| Baseline; no manual | 12 | 450/603 | 74.6% | +2.8 | **111/140** | **79.3%** | **↑ +11.4** | 96/151 | 63.6% | −1.3 |
| Baseline; manual | 0 | 473/643 | 73.6% | — | 98/137 | 71.5% | — | 104/160 | 65.0% | — |
| Baseline; manual | 12 | 356/617 | 57.7% | −15.9 | 68/137 | 49.6% | −21.9 | 84/161 | 52.2% | −12.8 |
| Classic novice; no manual | 0 | 157/200 | 78.5% | — | 42/52 | 80.8% | — | 32/52 | 61.5% | — |
| Classic novice; no manual | 12 | 147/185 | 79.5% | +1.0 | 30/43 | 69.8% | −11.0 | 45/61 | 73.8% | +12.2 |
| Classic novice; manual | 0 | 359/433 | 82.9% | — | 80/98 | 81.6% | — | 77/106 | 72.6% | — |
| Classic novice; manual | 12 | 321/401 | 80.0% | −2.9 | 70/88 | 79.5% | −2.1 | 81/106 | 76.4% | +3.8 |

F1 summarizes detection precision and recall for error opportunities that actually fired. False flags count incorrect assistant error calls; turns are assistant turns per rollout. These are descriptive diagnostics: opportunity exposure and episode length can differ across checkpoints. Cells show the mean over the saved split rollouts. `n/r` means the diagnostic was not recorded.

| Method | Update | Train mean F1 | Dev mean F1 | Frozen-test mean F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 0.687 | 0.624 | 0.723 |
| Baseline; no manual | 12 | 0.607 | 0.582 | 0.684 |
| Baseline; manual | 0 | 0.752 | 0.639 | 0.831 |
| Baseline; manual | 12 | 0.680 | 0.505 | 0.667 |
| Classic novice; no manual | 0 | 0.242 | 0.293 | 0.247 |
| Classic novice; no manual | 12 | 0.199 | 0.163 | 0.253 |
| Classic novice; manual | 0 | 0.537 | 0.559 | 0.558 |
| Classic novice; manual | 12 | 0.470 | 0.468 | 0.515 |

| Method | Update | Train mean false flags | Dev mean false flags | Frozen-test mean false flags |
| --- | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 0.828 | 1.229 | 0.856 |
| Baseline; no manual | 12 | 1.517 | 2.063 | 1.356 |
| Baseline; manual | 0 | 0.578 | 0.542 | 0.200 |
| Baseline; manual | 12 | 0.257 | 0.250 | 0.167 |
| Classic novice; no manual | 0 | 2.627 | 3.000 | 2.256 |
| Classic novice; no manual | 12 | 3.456 | 4.115 | 3.922 |
| Classic novice; manual | 0 | 1.216 | 1.000 | 1.167 |
| Classic novice; manual | 12 | 1.446 | 1.802 | 1.867 |

| Method | Update | Train mean assistant turns | Dev mean assistant turns | Frozen-test mean assistant turns |
| --- | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 12.57 | n/r | 13.56 |
| Baseline; no manual | 12 | 13.09 | n/r | 13.51 |
| Baseline; manual | 0 | 12.93 | 12.20 | 12.50 |
| Baseline; manual | 12 | 15.34 | 12.83 | 12.76 |
| Classic novice; no manual | 0 | 26.78 | 27.88 | 28.57 |
| Classic novice; no manual | 12 | 25.58 | 28.04 | 32.47 |
| Classic novice; manual | 0 | 23.62 | 24.02 | 25.46 |
| Classic novice; manual | 12 | 24.98 | 28.49 | 28.70 |

</details>

<details>
<summary>Task split, training protocol, and source artifacts</summary>

The task-disjoint library has 50 training tasks (136 scenarios), 11 development tasks (32 scenarios), and 11 frozen-test tasks (30 scenarios). The 12-update runs sampled six training scenarios eight times per update (48 rollouts/update), using `screensim_prevention_turns_v1` shaped reward. Development and test used `screensim_intime_success_v1`. The classic-novice/no-manual run also continued to update 24. The same task IDs appear for both personas within each split, but the simulated user behavior differs.

Verified artifacts on the project filesystem:

| Result | Artifact |
| --- | --- |
| Baseline/no-manual dev, original pass | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-baseline-training-20261004/completion-verification.json` |
| Baseline/no-manual dev, three-pass result | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-baseline-nomanual-dev-repeats-20261021/native-dev-summary.json` (new passes and independent verifier in the same directory) |
| Baseline/no-manual dev, three-pass AQ | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-baseline-nomanual-dev-repeats-20261021/completion-verification.json` (`gpt-summary.json`, `gemini-summary.json`, and verdicts in the same directory) |
| Baseline/no-manual test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-no-manual-consistent-20261004/baseline-test/summary.json` |
| Baseline/manual dev and test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-baseline-manual-20261006/final-dev-test-3pass/summary.json` |
| Baseline/manual train update 0 | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-baseline-manual-20261006/pre-rl-train-u0-8pass/screensim/update-0-pass-01/audit/result.json` (plus passes 02–03 in the same directory) |
| Baseline/manual train update 12 | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-new-task-train-eval-remaining-20261008/baseline-manual-u12/screensim/update-12-pass-01/audit/result.json` (plus passes 02–03 in the same directory) |
| Classic-novice/no-manual dev | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-u0-u12-final-eval-3pass-20261003/summary.json` (use its dev split only; its test split supplied a manual) |
| Classic-novice/no-manual test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-classic-novice-no-manual-test-u0-u12-u24-3pass-20261006/summary.json` (corrected no-manual test) |
| Classic-novice/manual dev and test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-classic-novice-manual-20261004/final-dev-test-3pass/summary.json` ([public results page](https://zixianma.github.io/interact-rl-replays/screensim-classic-novice-manual-u0-vs-u12/)) |
| Baseline/no-manual and classic-novice/manual train | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-new-task-train-eval-3pass-20261006/summary.json` |
| Baseline/manual and classic-novice/no-manual train | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-new-task-train-eval-remaining-20261008/summary.json` |

Task-macro values are the mean of the per-pass task-macro success values for each split. Goal attainment, F1, false flags, and turns are pooled from the corresponding per-pass `audit/result.json` scenario records, except the baseline/no-manual development diagnostics, whose first-pass native reports were recovered from the original training-run episode directory and combined with two new complete audit passes. On-time goal success and error prevention come from native `episodes/<episode_id>/report.json` files joined to those scenario records by episode ID; all included reports matched the audit strict-success verdicts. The corrected classic-novice/no-manual test summary records the path to each result, including a repaired update-12 pass under `recovery-1`; the secondary metrics follow those paths. Baseline/no-manual development assistant-turn means remain unreported because the original pass has no comparable audit turn counts. Engine revisions match between updates 0 and 12 within each condition but differ across some conditions; cross-condition differences should not be read as isolated effects of persona, manual, or split.

The six classic-novice/no-manual train passes reached their initial wall-time limits after saving 748 of 816 complete episodes. The remaining 68 scenarios were replayed with the same checkpoints and frozen catalog; each reconstructed 136-scenario pass passed the independent verifier at `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-new-task-train-eval-remaining-20261008/verify.py`. The new train-eval allocation used **46.81 of 49.5 approved H200 GPU-hours**, including recovery.

</details>

<details>
<summary>Initial reward-signal coverage with the assistant manual</summary>

The update-0 policy was evaluated eight times per training scenario, matching the optimizer’s eight-rollout group size. Positive reward occurred at least once on most scenarios, but this did not ensure improvement after RL.

| User | Positive-reward scenarios, count | % | Positive-reward tasks, count | % | Positive-reward rollouts, count | % | Strict-success rollouts, count | % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 128/136 | 94.1% | 50/50 | 100.0% | 869/1,088 | 79.9% | 207/1,088 | 19.0% |
| Classic novice | 110/136 | 80.9% | 45/50 | 90.0% | 620/1,088 | 57.0% | 90/1,088 | 8.3% |

Source reports: `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-baseline-manual-20261006/pre-rl-train-u0-8pass/signal-report.json` and `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-classic-novice-manual-20261004/pre-rl-train-u0-8pass/signal-report.json`.

</details>

<details>
<summary>Update-24 continuation and earlier mixed-manual test</summary>

The classic-novice/no-manual run continued from update 12 to update 24. In this consistent no-manual condition, the update-24 frozen-test result is five more successes than update 0 on the same 30 scenarios over three stochastic passes; it is a separate 24-update comparison.

| Update | Split | Success count | Success, mean ± SD |
| ---: | --- | ---: | ---: |
| 24 | Development | 2/96 | 2.1% ± 3.6 pp |
| 24 | Frozen test | 8/90 | 8.9% ± 5.1 pp |

An earlier evaluation of this run inadvertently supplied a manual **only at test time**, although training and development used no manual:

| Test context | Update | Success count | Success, mean ± SD |
| --- | ---: | ---: | ---: |
| Manual only at test | 0 | 4/90 | 4.4% ± 3.8 pp |
| Manual only at test | 12 | 11/90 | 12.2% ± 1.9 pp |
| Manual only at test | 24 | 8/90 | 8.9% ± 1.9 pp |

These independent passes are not paired with the consistent no-manual test passes in the main table and do not estimate a causal manual effect. The corrected no-manual test results are in the main table and the update-24 table above.

Sources: `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-classic-novice-no-manual-test-u0-u12-u24-3pass-20261006/summary.json`, `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-u0-u12-final-eval-3pass-20261003/summary.json`, and `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-u24-final-eval-3pass-20261004/summary.json`.

</details>

These fixed, small task sets make the results a proof of concept for RL interaction in ScreenSim. They do not establish a generalization benefit from RL or a causal effect of the manual.
