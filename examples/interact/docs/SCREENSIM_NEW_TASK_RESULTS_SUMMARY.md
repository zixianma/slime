# ScreenSim RL on the new task library

These experiments train a Qwen3.5-4B assistant with a Gemini 3.7 Flash simulated user. The new task-disjoint library has 50 training tasks (136 scenarios), 11 development tasks (32 scenarios), and 11 frozen-test tasks (30 scenarios). Each of the 12-update runs samples six training scenarios eight times per update, using the shaped `screensim_prevention_turns_v1` reward. Development and test results use strict `screensim_intime_success_v1`: the requested phone state must be reached on time without lasting unrequested changes. The two user personas, **baseline** and **classic novice**, are separate conditions; all comparisons within a persona use the same task IDs.

## Strict scenario success

| Simulated user | Assistant manual during training and evaluation | Development, update 0 → 12 | Frozen test, update 0 → 12 |
| --- | --- | ---: | ---: |
| Baseline | No | 15.6% → 25.0% | 27.8% → 25.6% |
| Baseline | Yes | 19.8% → 6.3% | 27.8% → 16.7% |
| Classic novice | No | 2.1% → 4.2% | 3.3% → 4.4% |
| Classic novice | Yes | 4.2% → 6.3% | 8.9% → 5.6% |

The no-manual classic-novice run continued to update 24: development success was **2.1%** and frozen-test success was **8.9%**. Baseline/no-manual development numbers come from one complete pass of 32 scenarios per checkpoint. Every other development cell and every test cell above uses three independent passes, totaling 96 and 90 rollouts per checkpoint, respectively.

The baseline-user/manual run began with the highest development success among the manual-enabled conditions, but **fell on both development and frozen test after RL**. Classic novice with a manual improved slightly on development and fell on test. Baseline without a manual improved on its one-pass development measure, while its three-pass frozen-test mean was nearly unchanged. Thus these completed 12-update runs provide no clear frozen-test gain. The differing user personas also change rollout difficulty, so percentages across personas should not be read as an isolated manual effect.

## Training-task results and initial reward signal

Three complete train-catalog passes per checkpoint evaluated all 136 training scenarios in each pass. For baseline without a manual, strict train success was **22.1% → 23.0%** at updates 0 → 12 (90/408 → 94/408). For classic novice with a manual, it was **9.6% → 6.4%** (39/408 → 26/408). Their positive shaped-reward rollouts were **299/408 → 306/408** and **256/408 → 224/408**, respectively.

Before the baseline-user/manual RL run, the frozen update-0 policy was rolled out eight times on every training scenario, matching the RL group size. It received positive shaped reward on **128/136 scenarios** and **50/50 tasks** at least once, with **869/1,088** positive-reward rollouts and **207/1,088** strict successes. The corresponding classic-novice/manual sweep had positive reward on **110/136 scenarios** and **45/50 tasks**, with **620/1,088** positive-reward rollouts and **90/1,088** strict successes. Broad initial reward coverage therefore did not ensure improvement after RL.

An earlier classic-novice run trained and evaluated on development **without** an assistant manual, but its original test evaluator supplied a manual only at test time. That mixed-context test reported **4.4%, 12.2%, and 8.9%** at updates 0, 12, and 24. A corrected no-manual test rerun found **3.3%, 4.4%, and 8.9%** at the same checkpoints, as shown above. The two test conditions use independent stochastic passes and should be labeled separately.

These are small fixed task sets: three passes measure variation in rollouts on the same 32 development or 30 test scenarios, not uncertainty over a larger population of tasks. Success differences of a few scenarios should therefore be interpreted cautiously. The checkpoint comparisons describe this proof-of-concept run and do not establish a generalization benefit from RL or a causal effect of the manual.
