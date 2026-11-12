# CookSim cross-persona evaluation of the classic-novice-trained policy

The frozen Qwen3.5-4B assistant trained with a **classic-novice** simulated cook showed **higher machine-judged interaction quality but no task-success gain** on either alternate user persona. On the same 30 held-out CookSim validation cases, update 0 and update 12 succeeded equally often with the **classic-expert** user (83/90 episodes each); with the **baseline** user, success was 82/90 versus 80/90. Each checkpoint–persona cell has three stochastic passes, so 90 episodes reuse **30 distinct cases**, not 90 independent tasks. Here a case is the fixed task configuration and scheduled events; an episode is one execution of that case with a persona and assistant, following the [public benchmark terminology](https://github.com/SalesforceAIResearch/Interactive-Visual-Gym/blob/main/docs/benchmark_protocol.md#terminology). The three-pass estimates are too small to establish equivalence or a reliable decline.

| Evaluation user | Checkpoint | Pass 1 successes | Pass 2 successes | Pass 3 successes | Total successes | Episodes | Success, pass mean ± SD | AQ-scored episodes | AQ, pass mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | Update 0 | 29 | 27 | 26 | 82 | 90 | 91.1% ± 5.1 pp | 90 | 10.24% ± 1.45 pp |
| Baseline | Update 12 | 27 | 26 | 27 | 80 | 90 | 88.9% ± 1.9 pp | 90 | 14.83% ± 2.71 pp |
| Classic expert | Update 0 | 27 | 28 | 28 | 83 | 90 | 92.2% ± 1.9 pp | 90 | 2.88% ± 0.60 pp |
| Classic expert | Update 12 | 29 | 27 | 27 | 83 | 90 | 92.2% ± 3.8 pp | 90 | 8.86% ± 3.64 pp |

Each pass contains 30 scenarios, so the pass columns give counts out of 30; the success percentage and SD summarize the three pass percentages.

The paired pass-mean success change is **−2.2 percentage points** with the baseline user and **0.0 points** with the classic-expert user. In contrast, Assistance Quality (AQ) rose **+4.59 points** with the baseline user and **+5.98 points** with the classic-expert user; all three paired passes improved for each persona. AQ is the Gemini-only SQ v4 mean of per-utterance rubric scores, pooling assistant turns within each pass and averaging the three pass scores equally. Its five axes assess factual grounding, situational relevance, actionable guidance, user-intent uptake, and conciseness. The AQ gains do not imply task-success gains.

Both transfer evaluations kept the 30-case scenario catalog, assistant checkpoints, seeds, renderer, and simulator service fixed while changing the persona. The service reported **Gemini 3.8 Flash** as the effective simulated-user model throughout. These tests probe persona sensitivity within one model service, not transfer to a different user-model family or to human cooks. Policy-dependent trajectories change which errors are offered, and the two personas produce different conversations, so their absolute success and AQ values are descriptive comparisons rather than isolated persona effects.

<details>
<summary>Native interaction diagnostics</summary>

The native audit records a valid detection only when an error is offered and the assistant flags it; prevention is a separate outcome. Error exposure varies with the trajectory. Across the three passes, detection and prevention fell at update 12 under both alternate personas, while false flags fell and assistant turns rose. These are descriptive episode totals, not causal estimates of the reward terms.

| Evaluation user | Checkpoint | Detected | Offered | Detection recall | Prevented | Prevention / offered | False flags | Assistant turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | Update 0 | 32 | 62 | 51.6% | 31 | 50.0% | 531 | 8,609 |
| Baseline | Update 12 | 20 | 61 | 32.8% | 18 | 29.5% | 350 | 8,853 |
| Classic expert | Update 0 | 22 | 61 | 36.1% | 21 | 34.4% | 466 | 8,009 |
| Classic expert | Update 12 | 12 | 67 | 17.9% | 12 | 17.9% | 252 | 8,539 |

All four rows contain 90 episodes. Detection and prevention denominators count offered errors, not all planned errors or episodes. The pass-level SDs above describe stochastic reruns on the frozen catalog; they do not quantify uncertainty over new scenarios or human users.

</details>

<details>
<summary>Interaction-quality detail</summary>

All 180 classic-expert episode verdicts and 15,056 assistant rounds passed a strict completeness audit. The effective judge was Gemini 3.8 Flash for every graded round. Each cell below pools rounds within a pass; the headline table gives the mean and sample SD across three passes. Differences between personas are descriptive because their conversations differ.

| Pass | Update 0 rounds | Update 0 AQ | Update 12 rounds | Update 12 AQ | Paired change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,387 | 3.23% | 2,808 | 4.97% | +1.75 pp |
| 2 | 2,537 | 3.22% | 2,528 | 12.20% | +8.98 pp |
| 3 | 2,360 | 2.18% | 2,436 | 9.39% | +7.21 pp |

The classic-expert paired AQ change is +5.98 ± 3.77 pp (mean ± sample SD of the three pass differences). For diagnostic context, the pass-mean axis scores changed as follows; AQ is a separate rubric aggregate, not the arithmetic mean of these five percentages.

| AQ axis | Update 0 | Update 12 | Change |
| --- | ---: | ---: | ---: |
| Truthful | 50.47% | 62.75% | +12.28 pp |
| Sensible | 46.50% | 42.00% | −4.50 pp |
| Helpful | 29.40% | 30.56% | +1.16 pp |
| Listens | 8.91% | 12.48% | +3.57 pp |
| Economy | 10.06% | 20.26% | +10.20 pp |

</details>

<details>
<summary>Evidence, checks, and resource accounting</summary>

- Baseline success, per-pass counts, and recovery lineage: `/gpfs/scrubbed/zixianma/interact/reviewer_rl_transfer_20261110/BASELINE_PERSONA_TRANSFER_RESULT.md`. Baseline Gemini-only AQ and judge audit: `/gpfs/scrubbed/zixianma/interact/reviewer_rl_transfer_20261110/BASELINE_PERSONA_AQ_RESULT.md`. Frozen baseline traces: /gpfs/scrubbed/zixianma/checkpoints/web/cooksim-reviewer-baseline-persona-20261110/.
- Frozen classic-expert catalog and exact persona-only change: /gpfs/scrubbed/zixianma/interact/reviewer_rl_transfer_20261110/classic-expert-v4-validation/manifest.json. All six native trace directories, Slurm receipts, and API ledger: /gpfs/scrubbed/zixianma/checkpoints/web/cooksim-reviewer-classic-expert-20261111/.
- The strict six-pass transfer verifier checked all 30 scenarios per checkpoint and pass, checkpoint identity, dataset hash, native trace artifacts, one effective user model, and API accounting. It returned TRANSFER_VERIFIED with 16,523 metered Gemini responses, 16,528 calls including unsettled reservations, and $126.66 estimated simulator spend. The six completed two-H200 jobs consumed 25.99 H200 GPU-hours, under the approved 48-hour ceiling.
- Classic-expert AQ verdicts: `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-reviewer-classic-expert-20261111/aq-inputs/sq_v4/frames/classic_expert/`; usage ledger: `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-reviewer-classic-expert-20261111/aq-usage.jsonl`. The strict verifier checked 180 paired episode verdicts, all 15,056 graded assistant rounds, and a single effective judge model. The ledger recorded 15,072 judge calls and $125.15 estimated judge spend, within the approved $160 AQ ceiling.
- The original classic-novice validation AQ is a judge-matched reference in the baseline AQ report; its absolute AQ values are not directly paired with the alternate-persona conversations.

</details>
