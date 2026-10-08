# ScreenSim RL on the new task library

Across the four completed 12-update runs, **none shows a clear frozen-test gain**. The only positive test change is one additional success in 90 rollouts for the classic-novice user without a manual. Baseline-user training without a manual improves development success in its single evaluation pass, but its three-pass test result falls slightly. With a manual throughout, both user personas finish below their update-0 frozen-test results.

## Strict success at updates 0 and 12

The assistant is Qwen3.5-4B and the simulated user is Gemini 3.7 Flash. “Manual” means the assistant receives the task manual in **training, development, and test**. Each row compares the same persona, manual condition, and task split at two checkpoints. Strict success requires the requested phone state by the deadline without lasting unrequested changes. **Δ** is update 12 minus update 0 in percentage points (pp); **↑ bold** marks a numerical gain, not a statistically established improvement.

| User | Manual | Split | Update 0 count | Update 0 % | Update 12 count | Update 12 % | Δ (pp) |
| --- | :---: | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | No | Development | 5/32 | 15.6% | **8/32** | **25.0%** | **↑ +9.4** |
| Baseline | No | Frozen test | 25/90 | 27.8% | 23/90 | 25.6% | −2.2 |
| Baseline | Yes | Development | 19/96 | 19.8% | 6/96 | 6.3% | −13.5 |
| Baseline | Yes | Frozen test | 25/90 | 27.8% | 15/90 | 16.7% | −11.1 |
| Classic novice | No | Development | 2/96 | 2.1% | **4/96** | **4.2%** | **↑ +2.1** |
| Classic novice | No | Frozen test | 3/90 | 3.3% | **4/90** | **4.4%** | **↑ +1.1** |
| Classic novice | Yes | Development | 4/96 | 4.2% | **6/96** | **6.3%** | **↑ +2.1** |
| Classic novice | Yes | Frozen test | 8/90 | 8.9% | 5/90 | 5.6% | −3.3 |

The baseline/no-manual **development** row uses one complete pass of 32 scenarios per checkpoint. Every other development row uses three passes (96 rollouts); every frozen-test row uses three passes (90 rollouts). Repeated passes revisit the same scenarios and therefore measure rollout variability, not uncertainty across new tasks. Differences of one or two successes are especially fragile. The two personas also differ in difficulty, so comparing their raw rates does not isolate the manual’s effect.

## Full training-task evaluation

Only two conditions have completed full train-catalog checkpoint evaluations. Each checkpoint was evaluated in three passes over all 136 training scenarios (408 rollouts). Positive shaped reward is a training-signal measure; it is **not** strict task success.

| User | Manual | Measure | Update 0 count | Update 0 % | Update 12 count | Update 12 % | Δ (pp) |
| --- | :---: | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | No | Strict success | 90/408 | 22.1% | **94/408** | **23.0%** | **↑ +1.0** |
| Baseline | No | Positive shaped reward | 299/408 | 73.3% | **306/408** | **75.0%** | **↑ +1.7** |
| Classic novice | Yes | Strict success | 39/408 | 9.6% | 26/408 | 6.4% | −3.2 |
| Classic novice | Yes | Positive shaped reward | 256/408 | 62.7% | 224/408 | 54.9% | −7.8 |

<details>
<summary>Task split, training protocol, and source artifacts</summary>

The task-disjoint library has 50 training tasks (136 scenarios), 11 development tasks (32 scenarios), and 11 frozen-test tasks (30 scenarios). The 12-update runs sampled six training scenarios eight times per update (48 rollouts/update), using `screensim_prevention_turns_v1` shaped reward. Development and test used `screensim_intime_success_v1`. The classic-novice/no-manual run also continued to update 24. The same task IDs appear for both personas within each split, but the simulated user behavior differs.

Verified artifacts on the project filesystem:

| Result | Artifact |
| --- | --- |
| Baseline/no-manual dev | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-baseline-training-20261004/completion-verification.json` |
| Baseline/no-manual test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-no-manual-consistent-20261004/baseline-test/summary.json` |
| Baseline/manual dev and test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-baseline-manual-20261006/final-dev-test-3pass/summary.json` |
| Classic-novice/no-manual dev and test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-classic-novice-no-manual-test-u0-u12-u24-3pass-20261006/summary.json` |
| Classic-novice/manual dev and test | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-library-gemini37-classic-novice-manual-20261004/final-dev-test-3pass/summary.json` ([public results page](https://zixianma.github.io/interact-rl-replays/screensim-classic-novice-manual-u0-vs-u12/)) |
| Full training-task evaluation | `/gpfs/scrubbed/zixianma/checkpoints/web/screensim-new-task-train-eval-3pass-20261006/summary.json` |

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
