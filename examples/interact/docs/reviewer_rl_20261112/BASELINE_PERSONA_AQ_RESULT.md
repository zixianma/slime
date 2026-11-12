# CookSim cross-persona interaction quality: baseline user

On the same 30 validation scenarios in each of three stochastic passes, Gemini-only SQ v4 interaction quality (AQ) rose from **10.24% ± 1.45 pp** at update 0 to **14.83% ± 2.71 pp** at update 12, a **+4.59 pp paired mean change**. All three paired passes improved. This AQ gain did **not** carry over to task success with this user: the separately verified success rates were 91.1% versus 88.9% (−2.2 pp). AQ is a machine-judged assistant-response score, not a human preference or task-success measure.

| User simulator | Update | Validation episodes | Assistant rounds | AQ, pass mean ± sample SD | Task success, pass mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| Baseline persona | 0 | 90 | 6,363 | 10.24% ± 1.45 pp | 91.1% ± 5.1 pp |
| Baseline persona | 12 | 90 | 6,385 | 14.83% ± 2.71 pp | 88.9% ± 1.9 pp |

The AQ values pool assistant rounds **within each pass** and give the three passes equal weight. The paired AQ differences were +9.29, +2.09, and +2.40 pp (mean +4.59 pp; sample SD 4.07 pp). The judge was Gemini 3.8 Flash for every graded round. One malformed judge response was resumed and regraded; the final audit verified all **180 episode verdicts and 12,748 assistant rounds**. Tracked AQ judge spend was **$114.48**, within the $190 AQ subcap and the aggregate rebuttal Gemini cap.

| AQ axis, pass-mean percentage | Update 0 | Update 12 | Change |
|---|---:|---:|---:|
| Truthful | 45.79% | 58.56% | +12.77 pp |
| Sensible | 66.01% | 60.36% | −5.65 pp |
| Helpful | 33.98% | 36.82% | +2.84 pp |
| Listens | 15.60% | 17.82% | +2.22 pp |
| Economy | 31.13% | 40.35% | +9.22 pp |

For a judge-matched reference, the original classic-novice validation traces, rescored here from their existing Gemini 3.8 Flash verdicts with the same pass-level aggregation, yield **18.94% ± 1.80 pp** at update 0 and **25.02% ± 0.05 pp** at update 12. Both personas show an AQ increase, but their absolute scores are not interchangeable: the user policies and stochastic trajectories differ, and this is not a randomized human transfer study. The established classic-novice round-pooled numbers (18.69% and 25.02%) use a different weighting across passes.

<details>
<summary>Per-pass AQ and evidence</summary>

| Pass | Update 0 episodes | Update 0 rounds | Update 0 AQ | Update 12 episodes | Update 12 rounds | Update 12 AQ | Paired change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 2,162 | 8.61% | 30 | 2,002 | 17.90% | +9.29 pp |
| 2 | 30 | 2,161 | 10.70% | 30 | 2,213 | 12.79% | +2.09 pp |
| 3 | 30 | 2,040 | 11.40% | 30 | 2,170 | 13.79% | +2.40 pp |

Strict verdict verifier and summary: [`summarize_baseline_aq.py`](summarize_baseline_aq.py). Inputs, per-round judgments, and usage ledger: `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-reviewer-baseline-persona-20261110/aq-inputs/` and `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-reviewer-baseline-persona-20261110/aq-usage.jsonl`. Task-success verification: [`BASELINE_PERSONA_TRANSFER_RESULT.md`](BASELINE_PERSONA_TRANSFER_RESULT.md). Classic-novice source: `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-validation-aq-20261019/aq-inputs/sq_v4/frames/classic_novice/`.

</details>
