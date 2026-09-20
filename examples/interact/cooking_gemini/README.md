# CookSim Gemini-user RL

This directory is the maintained CookSim training implementation:

- `train.sbatch` — fresh and resumed four-H200 entry point;
- `run.sh` — frozen Slime/Qwen3.5-4B arguments;
- `train.py` — split, topology, and resume invariants;
- `prepare_split.py` — deterministic parent-disjoint single-error split;
- `speculative_rollout.py` — first-four-of-five rollout collection;
- `metrics.py` — episode-weighted train and validation metrics;
- `render_policy_preflight.py` — renderer/policy colocation stress gate.
- `validate_reward.py` — retrospective reward-ranking audit utility.

Read [the complete handoff](../COOKING_GEMINI_HANDOFF.md) before running it.
Older cooking scripts in the parent directory are historical scripted-user,
calibration, or profiling paths and are not interchangeable with this setup.
