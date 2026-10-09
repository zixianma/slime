# ScreenSim RL on the new task library

On the primary scenario-weighted strict-success measure, **none of the four completed 12-update runs shows a clear frozen-test gain**. The only positive test change is one additional success in 90 rollouts for the classic-novice user without a manual. On the full training catalog, only baseline/no-manual increases in strict success (+1.0 percentage point); the other three methods decline. Baseline-user training without a manual improves development success in its single evaluation pass, but its three-pass test result falls slightly. With a manual throughout, both user personas finish below their update-0 frozen-test results.

## Strict success at updates 0 and 12

The assistant is Qwen3.5-4B and the simulated user is Gemini 3.7 Flash. “Manual” means the assistant receives the task manual in **training, development, and test**. Each method has separate update-0 and update-12 rows. Strict success requires the requested phone state by the deadline without lasting unrequested changes. Δ is update 12 minus update 0 in percentage points (pp), calculated before rounding; **bold ↑** marks a numerical gain, not a statistically established improvement. Dashes in update-0 delta cells mean the comparison is not applicable.

| Method (user; assistant manual) | Update | Train count | Train success % | Train Δ pp | Dev count | Dev success % | Dev Δ pp | Frozen-test count | Frozen-test success % | Test Δ pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 90/408 | 22.1% | — | 5/32 | 15.6% | — | 25/90 | 27.8% | — |
| Baseline; no manual | 12 | **94/408** | **23.0%** | **↑ +1.0** | **8/32** | **25.0%** | **↑ +9.4** | 23/90 | 25.6% | −2.2 |
| Baseline; manual | 0 | 85/408 | 20.8% | — | 19/96 | 19.8% | — | 25/90 | 27.8% | — |
| Baseline; manual | 12 | 54/408 | 13.2% | −7.6 | 6/96 | 6.3% | −13.5 | 15/90 | 16.7% | −11.1 |
| Classic novice; no manual | 0 | 11/408 | 2.7% | — | 2/96 | 2.1% | — | 3/90 | 3.3% | — |
| Classic novice; no manual | 12 | 9/408 | 2.2% | −0.5 | **4/96** | **4.2%** | **↑ +2.1** | **4/90** | **4.4%** | **↑ +1.1** |
| Classic novice; manual | 0 | 39/408 | 9.6% | — | 4/96 | 4.2% | — | 8/90 | 8.9% | — |
| Classic novice; manual | 12 | 26/408 | 6.4% | −3.2 | **6/96** | **6.3%** | **↑ +2.1** | 5/90 | 5.6% | −3.3 |

The baseline/no-manual **development** cells use one complete pass of 32 scenarios per checkpoint. Every other development cell uses three passes (96 rollouts); every frozen-test cell uses three passes (90 rollouts). Each available train comparison uses three passes of 136 scenarios (408 rollouts). Repeated passes revisit the same scenarios and therefore measure rollout variability, not uncertainty across new tasks. Differences of one or two successes are especially fragile. The two personas also differ in difficulty, so comparing their raw rates does not isolate the manual’s effect.

## Other evaluation metrics

**Task-macro strict success** averages each task’s scenario success rate, then weights the tasks equally. It can move differently from the scenario-weighted primary metric because tasks have different numbers of scenarios.

| Method | Update | Train task-macro % | Train Δ pp | Dev task-macro % | Dev Δ pp | Frozen-test task-macro % | Test Δ pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 23.1% | — | 16.7% | — | 28.8% | — |
| Baseline; no manual | 12 | **23.9%** | **↑ +0.8** | **25.8%** | **↑ +9.1** | **31.8%** | **↑ +3.0** |
| Baseline; manual | 0 | 22.4% | — | 20.7% | — | 29.3% | — |
| Baseline; manual | 12 | 13.9% | −8.6 | 6.1% | −14.6 | 21.2% | −8.1 |
| Classic novice; no manual | 0 | 3.2% | — | 2.5% | — | 4.5% | — |
| Classic novice; no manual | 12 | 2.8% | −0.4 | **4.0%** | **↑ +1.5** | **7.1%** | **↑ +2.5** |
| Classic novice; manual | 0 | 10.2% | — | 4.5% | — | 13.1% | — |
| Classic novice; manual | 12 | 7.2% | −3.0 | **6.6%** | **↑ +2.0** | 6.6% | −6.6 |

**Goal attainment** checks whether the final phone state satisfies the requested goal, regardless of the deadline or lasting unrequested changes. It is therefore less stringent than strict success. Counts and percentages are kept in separate columns; `n/r` means the detailed diagnostic was not recorded, whereas `—` means no full train-split checkpoint evaluation was run.

| Method | Update | Train goal count | Train goal % | Train Δ pp | Dev goal count | Dev goal % | Dev Δ pp | Frozen-test goal count | Frozen-test goal % | Test Δ pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 124/408 | 30.4% | — | n/r | n/r | n/r | 31/90 | 34.4% | — |
| Baseline; no manual | 12 | **132/408** | **32.4%** | **↑ +2.0** | n/r | n/r | n/r | 29/90 | 32.2% | −2.2 |
| Baseline; manual | 0 | 123/408 | 30.1% | — | 33/96 | 34.4% | — | 31/90 | 34.4% | — |
| Baseline; manual | 12 | 83/408 | 20.3% | −9.8 | 9/96 | 9.4% | −25.0 | 22/90 | 24.4% | −10.0 |
| Classic novice; no manual | 0 | 21/408 | 5.1% | — | 5/96 | 5.2% | — | 4/90 | 4.4% | — |
| Classic novice; no manual | 12 | **24/408** | **5.9%** | **↑ +0.7** | **6/96** | **6.3%** | **↑ +1.0** | 4/90 | 4.4% | 0.0 |
| Classic novice; manual | 0 | 62/408 | 15.2% | — | 12/96 | 12.5% | — | 14/90 | 15.6% | — |
| Classic novice; manual | 12 | **64/408** | **15.7%** | **↑ +0.5** | 12/96 | 12.5% | 0.0 | 14/90 | 15.6% | 0.0 |

### Training reward

The available complete three-pass train-catalog checkpoint evaluations each have 408 rollouts over all 136 training scenarios. Training uses the shaped `screensim_prevention_turns_v1` reward; development/test use binary strict reward, so their reported mean reward would duplicate strict success and is not directly comparable to the training-reward numbers below.

| Method | Update | Train positive-reward count | Train positive-reward % | Reward Δ pp | Train mean shaped reward | Mean reward Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 299/408 | 73.3% | — | 0.3598 | — |
| Baseline; no manual | 12 | **306/408** | **75.0%** | **↑ +1.7** | **0.3625** | **↑ +0.0026** |
| Baseline; manual | 0 | 332/408 | 81.4% | — | 0.3680 | — |
| Baseline; manual | 12 | 280/408 | 68.6% | −12.7 | 0.2566 | −0.1113 |
| Classic novice; no manual | 0 | 116/408 | 28.4% | — | 0.0262 | — |
| Classic novice; no manual | 12 | 93/408 | 22.8% | −5.6 | 0.0025 | −0.0237 |
| Classic novice; manual | 0 | 256/408 | 62.7% | — | 0.1983 | — |
| Classic novice; manual | 12 | 224/408 | 54.9% | −7.8 | 0.1477 | −0.0506 |

<details>
<summary>Additional simulator diagnostics (update 0 → 12)</summary>

F1 summarizes detection precision and recall for error opportunities that actually fired. False flags count incorrect assistant error calls; turns are assistant turns per rollout. These are descriptive diagnostics: opportunity exposure and episode length can differ across checkpoints. Cells show the mean over the saved split rollouts. `n/r` means the diagnostic was not recorded and `—` means the full train-split evaluation was not run.

| Method | Update | Train mean F1 | Dev mean F1 | Frozen-test mean F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 0.687 | n/r | 0.723 |
| Baseline; no manual | 12 | 0.607 | n/r | 0.684 |
| Baseline; manual | 0 | 0.752 | 0.639 | 0.831 |
| Baseline; manual | 12 | 0.680 | 0.505 | 0.667 |
| Classic novice; no manual | 0 | 0.242 | 0.293 | 0.247 |
| Classic novice; no manual | 12 | 0.199 | 0.163 | 0.253 |
| Classic novice; manual | 0 | 0.537 | 0.559 | 0.558 |
| Classic novice; manual | 12 | 0.470 | 0.468 | 0.515 |

| Method | Update | Train mean false flags | Dev mean false flags | Frozen-test mean false flags |
| --- | ---: | ---: | ---: | ---: |
| Baseline; no manual | 0 | 0.828 | n/r | 0.856 |
| Baseline; no manual | 12 | 1.517 | n/r | 1.356 |
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
| Baseline/no-manual dev | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-baseline-training-20261004/completion-verification.json` |
| Baseline/no-manual test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-no-manual-consistent-20261004/baseline-test/summary.json` |
| Baseline/manual dev and test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-baseline-manual-20261006/final-dev-test-3pass/summary.json` |
| Baseline/manual train update 0 | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-baseline-manual-20261006/pre-rl-train-u0-8pass/screensim/update-0-pass-01/audit/result.json` (plus passes 02–03 in the same directory) |
| Baseline/manual train update 12 | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-new-task-train-eval-remaining-20261008/baseline-manual-u12/screensim/update-12-pass-01/audit/result.json` (plus passes 02–03 in the same directory) |
| Classic-novice/no-manual dev | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-u0-u12-final-eval-3pass-20261003/summary.json` (use its dev split only; its test split supplied a manual) |
| Classic-novice/no-manual test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-classic-novice-no-manual-test-u0-u12-u24-3pass-20261006/summary.json` (corrected no-manual test) |
| Classic-novice/manual dev and test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-classic-novice-manual-20261004/final-dev-test-3pass/summary.json` ([public results page](https://zixianma.github.io/interact-rl-replays/screensim-classic-novice-manual-u0-vs-u12/)) |
| Baseline/no-manual and classic-novice/manual train | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-new-task-train-eval-3pass-20261006/summary.json` |
| Baseline/manual and classic-novice/no-manual train | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-new-task-train-eval-remaining-20261008/summary.json` |

Task-macro values are the mean of the per-pass task-macro success values for each split. Goal attainment, F1, false flags, and turns are pooled from the corresponding per-pass `audit/result.json` scenario records. The corrected classic-novice/no-manual test summary records the path to each result, including a repaired update-12 pass under `recovery-1`; the secondary metrics follow those paths. Baseline/no-manual development saved only compact metrics, so its detailed simulator diagnostics are unavailable.

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

| Update | Split | Success count | Success % |
| ---: | --- | ---: | ---: |
| 24 | Development | 2/96 | 2.1% |
| 24 | Frozen test | 8/90 | 8.9% |

An earlier evaluation of this run inadvertently supplied a manual **only at test time**, although training and development used no manual:

| Test context | Update | Success count | Success % |
| --- | ---: | ---: | ---: |
| Manual only at test | 0 | 4/90 | 4.4% |
| Manual only at test | 12 | 11/90 | 12.2% |
| Manual only at test | 24 | 8/90 | 8.9% |

These independent passes are not paired with the consistent no-manual test passes in the main table and do not estimate a causal manual effect. The corrected no-manual test results are in the main table and the update-24 table above.

Sources: `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-classic-novice-no-manual-test-u0-u12-u24-3pass-20261006/summary.json`, `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-u0-u12-final-eval-3pass-20261003/summary.json`, and `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-u24-final-eval-3pass-20261004/summary.json`.

</details>

These fixed, small task sets make the results a proof of concept for RL interaction in ScreenSim. They do not establish a generalization benefit from RL or a causal effect of the manual.
