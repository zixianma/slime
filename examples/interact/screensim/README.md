# ScreenSim

This package contains ScreenSim data preparation, smoke/evaluation entry points,
and Slurm launchers. Model-specific Qwen3.5 training and continuation machinery
lives under `../models/qwen35/`; generic tracking is under `../tracking/`.

Historical experiment results and split semantics are documented in
[`../docs/GPU_VALIDATION.md`](../docs/GPU_VALIDATION.md). Do not mix ScreenSim and
CookSim W&B run identities.
