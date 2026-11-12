# RL rebuttal experiments: results and execution status (12 November 2026)

**Controlled result.** The classic-novice-trained CookSim Qwen3.5-4B assistant (update 12) improved Gemini-judged assistance quality on two alternate personas, but did **not** improve their task success. Each persona/checkpoint used three stochastic passes of the **same 30 held-out cases** (90 episodes, 30 distinct task configurations). The [public benchmark protocol](https://github.com/SalesforceAIResearch/Interactive-Visual-Gym/blob/main/docs/benchmark_protocol.md#terminology) defines a case as a task configuration with scheduled human events and an episode as one execution with a persona and assistant. The effective simulated-user service was Gemini 3.8 Flash throughout. See the [complete cross-persona table](COOKSIM_CLASSIC_NOVICE_CROSS_PERSONA_RESULTS.md) for counts, pass-level SD, detection denominators, and judge protocol.

| Evaluation user | Update 0 success | Update 12 success | Success change | Update 0 AQ | Update 12 AQ | AQ change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline persona | 82/90 (91.1%) | 80/90 (88.9%) | −2.2 pp | 10.24% ± 1.45 pp | 14.83% ± 2.71 pp | +4.59 pp |
| Classic expert | 83/90 (92.2%) | 83/90 (92.2%) | 0.0 pp | 2.88% ± 0.60 pp | 8.86% ± 3.64 pp | +5.98 pp |

AQ is Assistance Quality, the Gemini-only SQ v4 per-utterance rubric aggregate. Its uncertainty is the sample SD of three pass scores, not a confidence interval over new tasks. The original classic-novice AQ and success results use different conversations; do not pool them with the alternate-persona rows.

| Other rebuttal experiment | Status | Result or next gate |
| --- | --- | --- |
| Existing-trace analysis | Complete | Update 12 used more assistant turns (12,701→13,191) but 27.0% fewer spoken words (436,563→318,896). Among 46 scenario-pass pairs with an offered error at both checkpoints, 13 native detections were lost and seven gained. These are descriptive, not a causal reward-term ablation. [Analysis snapshot](reviewer_rl_20261112/TRACE_ANALYSIS_REPORT.md). |
| Changed-error-distribution v5 pilot | Complete, one case per checkpoint | Both checkpoints succeeded (1/1 each), but only one of two planned errors was offered in either rollout. No error-recovery or transfer claim follows. [Pilot table](reviewer_rl_20261112/PILOT_RESULTS.md). |
| Recipe-disjoint sushi pilot | Complete, one case per checkpoint | Update 0 succeeded (1/1); update 12 timed out (0/1). Neither rollout offered its planned error. [Pilot table](reviewer_rl_20261112/PILOT_RESULTS.md). |
| Recipe-disjoint full catalog | Frozen and loader-audited; policy evaluation pending | Nineteen cases across four recipes absent from the RL train/validation grid; all 19 triggered their planned error under a scripted user. This does not establish exposure under a stochastic VLM user. [Catalog report](reviewer_rl_20261112/HELDOUT_CATALOG_REPORT.md). |
| One-update full-reward training pilot | Failed during checkpoint save; fix tested, no replacement result | The pilot completed 20 initial-eval and 24 training rollouts and 1,159 actor microbatches, then hit an `AF_UNIX` socket-path limit. No valid update-1 checkpoint or post-training evaluation exists. The shorter temporary path passed smoke tests. [Recovery report](reviewer_rl_20261112/REWARD_PILOT_RECOVERY.md). |
| Four-arm matched reward ablation | Not launched | Compare success-only, +prevention, −false flags, and −turns with matched training settings, then evaluate on frozen held-out cases. The historical estimate is about 372 H200 GPU-hours for one seed across four arms, before failures and evaluation. |
| Different user-model family | Paired pilot submitted 12 November, results pending | One classic-novice validation case at both checkpoints, changing the user backend to pinned GPT-5.4 Mini; maximum 4 H200 GPU-hours and $20 OpenAI API. A one-case pilot is a compatibility/cost check, not a transfer estimate. |
| Blinded human interaction ratings | Twelve matched error-window pairs selected; unrated | Collect independent 0/1/2/NA axis ratings, including visual evidence, then compare with AQ judge scores. [Annotation protocol](reviewer_rl_20261112/ANNOTATION_PROTOCOL.md). |

The frozen ScreenSim task-disjoint test has no clear scenario-weighted success gain across its four 12-update training conditions; see the [ScreenSim results](SCREENSIM_NEW_TASK_RESULTS_SUMMARY.md). Human-participant transfer has not been run and would require the appropriate study clearance. Neither three stochastic passes nor machine-judged AQ supplies uncertainty across independent RL training seeds or human users.

<details>
<summary>Execution limits and evidence</summary>

- The new user authorization raises aggregate rebuttal API ceilings to **$1,000 Gemini** and **$200 OpenAI**, including earlier usage. Study-level cost guards remain separate. Conservative committed Gemini usage before new launches was about **$632.16**, including unsettled reservations; this is an estimate from the existing ledgers, not a provider invoice.
- The OpenAI pilot is Slurm array `459159`, two 2-H200 jobs with a one-hour limit per task. Its shared ledger has a $20 study cap and a 100-user-call limit per episode. This report does not claim that queued jobs have produced results.
- Completed baseline transfer and AQ accounting: [transfer report](reviewer_rl_20261112/BASELINE_PERSONA_TRANSFER_RESULT.md) and [AQ report](reviewer_rl_20261112/BASELINE_PERSONA_AQ_RESULT.md). The [cross-persona table](COOKSIM_CLASSIC_NOVICE_CROSS_PERSONA_RESULTS.md) includes classic-expert provenance and judgment audit.
- The source snapshots above preserve the original study accounting and local artifact pointers. Some relative links inside snapshots refer to the original workspace rather than this published directory; the frozen result claims and counts are reproduced here and in the cross-persona table. No credentials, raw human-study data, or private rollout transcripts are published in this directory.

</details>
