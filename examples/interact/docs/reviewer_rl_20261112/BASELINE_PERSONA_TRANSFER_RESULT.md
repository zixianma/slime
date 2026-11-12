# CookSim cross-user checkpoint evaluation: baseline persona

On the same 30 frozen CookSim validation scenarios, the policy trained with the classic-novice simulated user did **not show a success gain** when evaluated with the baseline simulated-user persona. Across three complete paired passes, update 0 succeeded in 82/90 episodes (91.1% mean across passes, 5.1 percentage-point pass-level SD), while update 12 succeeded in 80/90 (88.9%, 1.9-point SD). The mean paired change was −2.2 percentage points (5.1-point SD across three pass differences). These 90 episodes reuse 30 scenarios three times; they are not 90 distinct scenarios. With only three passes, the small difference should not be interpreted as a reliable decline.

| Frozen checkpoint | Pass 1 success | Pass 2 success | Pass 3 success | Total successful episodes | Pass-mean success ± SD |
|---|---:|---:|---:|---:|---:|
| Update 0 | 29/30 | 27/30 | 26/30 | 82/90 | 91.1% ± 5.1 pp |
| Update 12 | 27/30 | 26/30 | 27/30 | 80/90 | 88.9% ± 1.9 pp |

All six passes completed one episode per frozen scenario with full native reports, assistant decisions, and training audits. The result verifier checked the exact dataset hash, checkpoint labels and indices, scenario pairing, one shared CookSim engine revision (`c92b6caad0`), metered user-model responses, and the shared API cap. The requested user-model name was `gemini-3.7-flash`; all recorded API responses reported effective model `gemini-3.8-flash`. The result therefore tests transfer between two persona settings of this user-model service, not transfer to a different model family or to humans.

<details>
<summary>Provenance, recovery, and resource accounting</summary>

- Frozen 30-case catalog and hash: `baseline-persona-v4-validation/cooking-all.jsonl` and `manifest.json`.
- Checkpoints: `baseline-launch/checkpoints.json`; update 0 is the frozen Qwen3.5-4B model, and update 12 loads checkpoint index 11 of the classic-novice CookSim RL run.
- Results and native traces: `/gpfs/scrubbed/zixianma/checkpoints/web/cooksim-reviewer-baseline-persona-20261110/cooking/update-{0,12}-rep{1,2,3}/`.
- Stage 1 update-0 pass was recovered after an accounting-ID collision. Twenty fully completed episodes were preserved; only the ten missing cases were rerun with their original sampling indices, then merged and strictly reverified. A separate stage-2 SGLang port collision consumed 2.5 GPU minutes before targeted replacement. These failures and all attempts remain in the Slurm lineage.
- The study consumed 30.528 H200-hours across successful and failed attempts, under the approved 48 H200-hour ceiling. Metered Gemini response usage was $163.28; conservative unsettled reservations add $49.55, for $212.83 committed under the study's $300 API subcap. The aggregate rebuttal API ceiling is $500 across all experiments.

</details>
