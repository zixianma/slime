# Assistant RL experiment setups

This records the immutable data and rollout units used by the ScreenSim and
CookSim experiments. A scenario group is one native `EpisodeSpec`; an episode
is one stochastic policy attempt within that group.

## ScreenSim scripted-human run

- Engine source: `screensim-engine` at `9451adb` (the revision frozen in the
  manifest; historical results must continue to use that checkout/revision).
- Split: `screensim_rl_v2_18train_12val`; manifest SHA-256
  `a29464102278d784a06a6b5a57ab6cb3dd53b2da6a74015bab54f8687a3d96f9`.
- Train: 6 whole tasks, 18 scenarios (three certified composites per task).
- Validation: 5 disjoint whole tasks, 12 scenarios.
- Human/persona: deterministic `scripted_human_v1`, baseline persona.
- Observation/reward: two ordered frames; `screensim_intime_success_v1`.
- Update: 6 scenario groups x 8 attempts = 48 episodes; therefore all 18
  training scenarios are covered every three updates.
- Validation: 12 scenario groups x 4 attempts = 48 episodes, every 3 updates.
- Sampling: policy temperature 0.8, top-p 1, max 512 new tokens. The dataset is
  sequential (`rollout_shuffle=False`) and its offset is checkpointed.
- Learner: Qwen3.5-4B, frozen vision tower, GRPO, global batch 48, LR `5e-7`.

## CookSim scripted-human run

- Engine data: `cook-bench-engine/bench/v5`; split version
  `cooking-rl-v1-20260914-budget-v2`.
- Train: 40 scenarios: 10 recipe/layout condition cells x 4 variants.
- Validation: 10 held-out scenario variants: one per condition cell. Recipes
  and layouts overlap, so this is development validation, not unseen-domain
  testing.
- Human/persona: programmatic scripted cook, baseline persona; deterministic
  native seed 0. Observation is rendered frames and reward is
  `native_outcome_v1` task success.
- Update: 6 scenario groups x 8 attempts = 48 episodes. With sequential
  sampling, the 40 scenarios repeat after 6 full updates plus 4 groups.
- Validation: 10 scenario groups x 4 attempts = 40 episodes, every 3 updates.
- Policy/learner: Qwen3.5-4B, frozen vision tower, GRPO, global batch 48, LR
  `5e-7`; four-GPU learner topology is TP=2 x DP=2.

## ScreenSim Gemini-human run

- Engine source: upstream `hellomuffin/screensim-engine` at `869d8e9`; this has
  v3 grading, stricter success, and revised task/persona behavior.
- Human: engine-native `PlanHuman` with `gemini-3.7-flash`, in free mode. The
  local Qwen policy remains the assistant. This uses paid Gemini API calls.
- Use a new revision-pinned manifest and a separate W&B run. Never merge these
  measurements into the scripted-human curve: both engine semantics and the
  human distribution changed.
- Gemini generation is stochastic and has no fixed replay seed in this setup;
  validation therefore measures the joint policy/user interaction with more
  variance than scripted-human validation.
- Train/eval split: 18/12 scenarios. One update is six groups x eight attempts;
  validation is 12 groups x four attempts at updates 0, 3, 6, 9, and 12.
- The earlier job `299664` completed only a two-episode, one-update smoke test
  with zero successes and no validation. The first full curve is a separate run.

## CookSim Gemini-human single-error run (current)

- Split: `cooking-single-error-gemini37-classic-novice-budget-v2-20260919`;
  120 training scenarios and 10 balanced development-validation scenarios.
  Parent composite variants are split before single-error derivation.
- Validation uses two attempts per scenario (20 episodes) at updates 0, 3, 6,
  and 9. It is development validation, not unseen-layout/recipe testing.
- Human: engine-native `GeminiHuman` with `gemini-3.7-flash`, configured as the
  benchmark's `classic_novice`; calls are stochastic and paid. Assistant and
  learner are Qwen3.5-4B.
- Reward: `cooking_prevention_turns_v1` = task success + `0.30` if any error is
  prevented - `0.02 * min(false_flags, 15)` -
  `0.05 * min(assistant_turns / 200, 1)`.
- Learner unit: 6 scenario groups x 4 accepted attempts = 24 episodes, GRPO,
  global batch 24, LR `5e-7`, frozen vision tower. All four GPUs update as
  TP=2 x DP=2 after a hard renderer-process handoff check.
- Episode budget: 3,600 native wall seconds, doubled by the engine for composite
  episodes, plus a hard 200 assistant-decision cap.
- Rollout topology: four TP1 SGLang servers, one per H200; the selected Vulkan
  renderer GPU also hosts its policy server only after a shared-GPU preflight.
- Straggler mitigation: launch five attempts for each group and train
  on its first four complete attempts. At most 24 trajectories run at once
  (six per server); the spare is cancelled and fully closed once four attempts
  complete. Exact validation remains non-speculative. This may favor shorter
  trajectories, so audit receipts record accepted/cancelled attempts and raw
  turn counts remain a required interpretation metric.
- Resume: save every update and reuse the same W&B run only after checkpoint,
  history, split, and hyperparameter verification.
