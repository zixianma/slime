# CookSim Gemini-user RL

This directory is the maintained CookSim training implementation:

- `train.sbatch` — fresh and resumed four-H200 entry point;
- `run.sh` — frozen Slime/Qwen3.5-4B arguments;
- `train.py` — split, topology, and resume invariants;
- `prepare_split.py` — deterministic parent-disjoint single-error split;
- `speculative_rollout.py` — first-four-of-five rollout collection;
- `metrics.py` — episode-weighted train and validation metrics;
- `render_policy_preflight.py` — renderer/policy colocation stress gate;
- `cooking_episode_cache.py` and `cooking_recover_episodes.py` — durable
  trajectory caching and recovery;
- `cooking_gpu_handoff.py` and `cooking_render_probe.py` — renderer ownership
  and GPU handoff checks;
- `prepare_cooking_rl.py` and `prepare_split.py` — immutable parent and
  Gemini-user split builders;
- `validate_reward.py` — retrospective reward-ranking audit utility.

Read [the complete handoff](../docs/COOKING_GEMINI_HANDOFF.md) before running it.
Older cooking scripts under `../archive/cooking_calibration/` are historical
scripted-user/calibration paths and are not interchangeable with this setup.
