# Full-suite checkpoint evaluation

This package evaluates one policy rollout for every frozen training and
development-validation scenario. Checkpoint selection comes from a versioned
JSON specification; the Slurm launchers contain no experiment checkpoint paths.

See [`../docs/FULL_SUITE_EVAL.md`](../docs/FULL_SUITE_EVAL.md) for dataset
preparation, checkpoint-spec examples, launch commands, result semantics, and
replay generation.
