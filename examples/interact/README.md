# Unified interactive RL environments

This package connects native assistant-supervision simulators to official
[THUDM/Slime](https://github.com/THUDM/slime). The policy receives each engine's
native assistant prompt and ordered images; the simulated person, world state,
clock, grading, and artifacts remain inside the engine.

## Start here

| Goal | Guide | Status |
| --- | --- | --- |
| Reproduce ScreenSim + Gemini-user RL | [ScreenSim Gemini reproduction](docs/SCREENSIM_GEMINI_REPRO.md) | Supported; full update-12 experiment in progress |
| Reproduce CookSim + Gemini-user RL | [CookSim Gemini reproduction](docs/COOKING_GEMINI_REPRO.md) | Supported and validated through checkpoint continuation |
| Operate or recover a run | [Runbook](docs/RUNBOOK.md) | Supported |
| Inspect frozen experiment definitions | [Experiment setups](docs/EXPERIMENT_SETUPS.md) | Current |
| Review verified evidence | [Progress](docs/PROGRESS.md) and [GPU validation](docs/GPU_VALIDATION.md) | Current |

Gemini-user experiments make paid, stochastic API calls. Keep them on separate
W&B curves from scripted-user experiments; they measure a different joint
policy/user distribution.

## Layout

- `screensim/` and `cooking_gemini/`: engine-specific data and launch code.
- `models/`: model-specific Slime launch and metric hooks.
- `tracking/`: credential isolation, single-writer W&B, and resume guards.
- `common/` and `runtime/`: shared preparation and runtime helpers.
- `configs/`: centralized engine and GPU-layout configurations.
- `tools/`: profiling, comparison, and rollout visualization.
- `docs/`: current operational records; `docs/history/` is not launchable.
- `archive/`: superseded calibration experiments retained for provenance.

The stable boundary is `EpisodeSpec -> Observation -> Action -> EpisodeResult`.
Adapters intercept only the native assistant call. They must not disclose hidden
plans, oracle state, or grader-only fields to the policy.

## Local validation

Use the Qwen runtime for the complete interaction suite:

```bash
source examples/interact/models/qwen35/qwen35_env.sh
pytest -q tests/interact
```

For a fast edit loop, run the tests named in the relevant reproduction guide.
Production launchers additionally fail closed on split hashes, engine revisions,
checkpoint metadata, W&B history, and run identity.

## Implementation status

- ScreenSim: native frames, PlanHuman/scripted humans, native grading, Slime RL.
- CookSim: native rendering, Gemini novice user, speculative straggler handling,
  versioned reward, Slime RL.
- VH Streaming: protocol design only; no production adapter yet.

Generated splits, checkpoints, episode artifacts, API keys, and local W&B state
are deliberately excluded from Git.
