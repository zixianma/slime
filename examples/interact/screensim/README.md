# ScreenSim

This package contains ScreenSim data preparation, smoke/evaluation entry points,
and Slurm launchers. Model-specific Qwen3.5 training and continuation machinery
lives under `../models/qwen35/`; generic tracking is under `../tracking/`.

Historical experiment results and split semantics are documented in
[`../docs/GPU_VALIDATION.md`](../docs/GPU_VALIDATION.md). Do not mix ScreenSim and
CookSim W&B run identities.

For the supported engine-native Gemini-user experiment, follow the
[step-by-step reproduction guide](../docs/SCREENSIM_GEMINI_REPRO.md). The
`gemini_train.sbatch` launcher supports both fresh and checkpoint-resumed runs.
For one-rollout coverage of every frozen train and validation scenario, use the
[full-suite evaluation guide](../docs/FULL_SUITE_EVAL.md).
