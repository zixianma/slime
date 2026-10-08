# ScreenSim RL on the new task library

On the primary scenario-weighted strict-success measure, **none of the four completed 12-update runs shows a clear frozen-test gain**. The only positive test change is one additional success in 90 rollouts for the classic-novice user without a manual. Baseline-user training without a manual improves development success in its single evaluation pass, but its three-pass test result falls slightly. With a manual throughout, both user personas finish below their update-0 frozen-test results.

## Strict success at updates 0 and 12

The assistant is Qwen3.5-4B and the simulated user is Gemini 3.7 Flash. “Manual” means the assistant receives the task manual in **training, development, and test**. Each cell compares update **0 → 12** for the same method and split. Strict success requires the requested phone state by the deadline without lasting unrequested changes. Parentheses show the percentage-point (pp) change; **bold ↑** marks a numerical gain, not a statistically established improvement. A dash means no full-split checkpoint evaluation was run.

| Method (user; assistant manual) | Train count | Train success % (Δ pp) | Dev count | Dev success % (Δ pp) | Frozen-test count | Frozen-test success % (Δ pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | 90/408 → **94/408** | 22.1% → **23.0% (↑ +1.0)** | 5/32 → **8/32** | 15.6% → **25.0% (↑ +9.4)** | 25/90 → 23/90 | 27.8% → 25.6% (−2.2) |
| Baseline; manual | — | — | 19/96 → 6/96 | 19.8% → 6.3% (−13.5) | 25/90 → 15/90 | 27.8% → 16.7% (−11.1) |
| Classic novice; no manual | — | — | 2/96 → **4/96** | 2.1% → **4.2% (↑ +2.1)** | 3/90 → **4/90** | 3.3% → **4.4% (↑ +1.1)** |
| Classic novice; manual | 39/408 → 26/408 | 9.6% → 6.4% (−3.2) | 4/96 → **6/96** | 4.2% → **6.3% (↑ +2.1)** | 8/90 → 5/90 | 8.9% → 5.6% (−3.3) |

The baseline/no-manual **development** cells use one complete pass of 32 scenarios per checkpoint. Every other development cell uses three passes (96 rollouts); every frozen-test cell uses three passes (90 rollouts). The two available train comparisons use three passes of 136 scenarios (408 rollouts). Repeated passes revisit the same scenarios and therefore measure rollout variability, not uncertainty across new tasks. Differences of one or two successes are especially fragile. The two personas also differ in difficulty, so comparing their raw rates does not isolate the manual’s effect.

## Other evaluation metrics

**Task-macro strict success** averages each task’s scenario success rate, then weights the tasks equally. It can move differently from the scenario-weighted primary metric because tasks have different numbers of scenarios. Each cell is update **0 → 12**, with its change in pp.

| Method | Train task-macro success % (Δ pp) | Dev task-macro success % (Δ pp) | Frozen-test task-macro success % (Δ pp) |
| --- | ---: | ---: | ---: |
| Baseline; no manual | 23.1% → **23.9% (↑ +0.8)** | 16.7% → **25.8% (↑ +9.1)** | 28.8% → **31.8% (↑ +3.0)** |
| Baseline; manual | — | 20.7% → 6.1% (−14.6) | 29.3% → 21.2% (−8.1) |
| Classic novice; no manual | — | 2.5% → **4.0% (↑ +1.5)** | 4.5% → **7.1% (↑ +2.5)** |
| Classic novice; manual | 10.2% → 7.2% (−3.0) | 4.5% → **6.6% (↑ +2.0)** | 13.1% → 6.6% (−6.6) |

**Goal attainment** checks whether the final phone state satisfies the requested goal, regardless of the deadline or lasting unrequested changes. It is therefore less stringent than strict success. Counts and percentages are kept in separate columns; `n/r` means the detailed diagnostic was not recorded, whereas `—` means no full train-split checkpoint evaluation was run.

| Method | Train goal count | Train goal % | Dev goal count | Dev goal % | Frozen-test goal count | Frozen-test goal % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline; no manual | 124/408 → **132/408** | 30.4% → **32.4% ↑** | n/r | n/r | 31/90 → 29/90 | 34.4% → 32.2% |
| Baseline; manual | — | — | 33/96 → 9/96 | 34.4% → 9.4% | 31/90 → 22/90 | 34.4% → 24.4% |
| Classic novice; no manual | — | — | 5/96 → **6/96** | 5.2% → **6.3% ↑** | 4/90 → 4/90 | 4.4% → 4.4% |
| Classic novice; manual | 62/408 → **64/408** | 15.2% → **15.7% ↑** | 12/96 → 12/96 | 12.5% → 12.5% | 14/90 → 14/90 | 15.6% → 15.6% |

### Training reward

Only two methods have complete three-pass train-catalog checkpoint evaluations. Each checkpoint has 408 rollouts over all 136 training scenarios. Training uses the shaped `screensim_prevention_turns_v1` reward; development/test use binary strict reward, so their reported mean reward would duplicate strict success and is not directly comparable to the training-reward numbers below.

| Method | Train positive-reward count | Train positive-reward % | Train mean shaped reward |
| --- | ---: | ---: | ---: |
| Baseline; no manual | 299/408 → **306/408** | 73.3% → **75.0% ↑** | 0.3598 → **0.3625 ↑** |
| Classic novice; manual | 256/408 → 224/408 | 62.7% → 54.9% | 0.1983 → 0.1477 |

<details>
<summary>Additional simulator diagnostics (update 0 → 12)</summary>

F1 summarizes detection precision and recall for error opportunities that actually fired. False flags count incorrect assistant error calls; turns are assistant turns per rollout. These are descriptive diagnostics: opportunity exposure and episode length can differ across checkpoints. Cells show the mean over the saved split rollouts. `n/r` means the diagnostic was not recorded and `—` means the full train-split evaluation was not run.

| Method | Train mean F1 | Dev mean F1 | Frozen-test mean F1 |
| --- | ---: | ---: | ---: |
| Baseline; no manual | 0.687 → 0.607 | n/r | 0.723 → 0.684 |
| Baseline; manual | — | 0.639 → 0.505 | 0.831 → 0.667 |
| Classic novice; no manual | — | 0.293 → 0.163 | 0.247 → 0.253 |
| Classic novice; manual | 0.537 → 0.470 | 0.559 → 0.468 | 0.558 → 0.515 |

| Method | Train mean false flags | Dev mean false flags | Frozen-test mean false flags |
| --- | ---: | ---: | ---: |
| Baseline; no manual | 0.828 → 1.517 | n/r | 0.856 → 1.356 |
| Baseline; manual | — | 0.542 → 0.250 | 0.200 → 0.167 |
| Classic novice; no manual | — | 3.000 → 4.115 | 2.256 → 3.922 |
| Classic novice; manual | 1.216 → 1.446 | 1.000 → 1.802 | 1.167 → 1.867 |

| Method | Train mean assistant turns | Dev mean assistant turns | Frozen-test mean assistant turns |
| --- | ---: | ---: | ---: |
| Baseline; no manual | 12.57 → 13.09 | n/r | 13.56 → 13.51 |
| Baseline; manual | — | 12.20 → 12.83 | 12.50 → 12.76 |
| Classic novice; no manual | — | 27.88 → 28.04 | 28.57 → 32.47 |
| Classic novice; manual | 23.62 → 24.98 | 24.02 → 28.49 | 25.46 → 28.70 |

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
| Classic-novice/no-manual dev | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-u0-u12-final-eval-3pass-20261003/summary.json` (use its dev split only; its test split supplied a manual) |
| Classic-novice/no-manual test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-classic-novice-no-manual-test-u0-u12-u24-3pass-20261006/summary.json` (corrected no-manual test) |
| Classic-novice/manual dev and test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-classic-novice-manual-20261004/final-dev-test-3pass/summary.json` ([public results page](https://zixianma.github.io/interact-rl-replays/screensim-classic-novice-manual-u0-vs-u12/)) |
| Full training-task evaluation | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-new-task-train-eval-3pass-20261006/summary.json` |

Task-macro values are the mean of the per-pass task-macro success values for each split. Goal attainment, F1, false flags, and turns are pooled from the corresponding per-pass `audit/result.json` scenario records. The corrected classic-novice/no-manual test summary records the path to each result, including a repaired update-12 pass under `recovery-1`; the secondary metrics follow those paths. Baseline/no-manual development saved only compact metrics, so its detailed simulator diagnostics are unavailable.

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

| Update-24 split | Success count | Success % |
| --- | ---: | ---: |
| Development | 2/96 | 2.1% |
| Frozen test | 8/90 | 8.9% |

An earlier evaluation of this run inadvertently supplied a manual **only at test time**, although training and development used no manual:

| Split | Update 0 count | Update 0 % | Update 12 count | Update 12 % | Update 24 count | Update 24 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen test, manual only at test | 4/90 | 4.4% | **11/90** | **12.2%** | 8/90 | 8.9% |

These independent passes are not paired with the consistent no-manual test passes in the main table and do not estimate a causal manual effect. The corrected no-manual test results are in the main table and the update-24 table above.

Sources: `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-classic-novice-no-manual-test-u0-u12-u24-3pass-20261006/summary.json`, `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-u0-u12-final-eval-3pass-20261003/summary.json`, and `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-u24-final-eval-3pass-20261004/summary.json`.

</details>

These fixed, small task sets make the results a proof of concept for RL interaction in ScreenSim. They do not establish a generalization benefit from RL or a causal effect of the manual.
