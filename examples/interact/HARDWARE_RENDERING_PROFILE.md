# Cooking hardware-renderer experiment

User requested hardware rendering validation, followed by one rendering GPU plus
one policy GPU. This is a rollout throughput experiment, **not an optimizer run**.
No paid model API calls; human remains scripted, model is local Qwen3.5-4B.

## Validation

- Capability job 295535: one H200, 8 CPUs, 10-minute ceiling; completed in 94s.
  Vulkan and EGL both reported NVIDIA H200 and passed red-pixel readback with
  software fallback disabled. Default backend did not create a context.
- Cooking correctness job 295536: one H200, 8 CPUs, 10-minute ceiling;
  completed in 4m46s. Six matched no-op decisions (ticks 0..10), 32 image pairs.
  Prompts, ticks, and image counts matched. Mean absolute pixel difference
  across pairs was 0.994/255; max per-image mean was 1.116/255.
  Inspected final software/hardware images showed distinct, corresponding
  top-down and ego views, not the historical duplicated-camera failure.
  Hardware context remained valid. This is a short correctness smoke test,
  not proof across every task or a long-running stress test.
- Software steps ~2.2s. Hardware had 43.6s reset, 7.35s first step, then
  0.138–0.172s for the next four steps. Do not omit cold-start cost or extrapolate
  single-environment latency directly to eight concurrent environments.

Artifacts: `/gpfs/scrubbed/zixianma/checkpoints/web/render-probe-295535/`
and `/gpfs/scrubbed/zixianma/checkpoints/web/render-check-295536/`.

## Throughput job

Initial job 295629 was cancelled after 42 seconds before inference/rollouts:
Slurm inherited the parent two-GPU request in each step and blocked the second
step. Fixed by specifying both `--gpus=1` and `--gpus-per-task=1` per step.
Replacement 295631 has a 28-minute ceiling, 2 H200, 16 CPUs, 240 GiB.
Together these stay below the original 1 GPU-hour profiling ceiling.
`profile_dedicated_renderer.sbatch` runs two exclusive one-GPU/eight-CPU steps:

- Policy: local SGLang server, TP1, max four running requests, existing settings.
- Renderer: eight concurrent cooking environments using explicit Vulkan mode.

Each step records its GPU UUID; the controller rejects non-isolated assignments
or overlapping renderer/policy devices. Every hardware-mode assistant callback
checks the actual cooking context for NVIDIA hardware and context loss.
The renderer must not silently fall back to SwiftShader.

Same four calibration cases, 16-turn prefixes, model and sampling parameters as
the original scaling trial. Sampling uses original scenario group keys so the
new renderer configuration does not itself change sampling seeds. Different
batching and pixels can still change trajectories. All time including browser
startup and inference server startup counts toward allocation cost.

Compare completed turns per elapsed GPU-hour, phase throughput, step/inference
latencies, episode startup, failures and per-device utilization against 295509
(2 inference GPUs / 16 CPUs) and 295510 (4 inference GPUs / 32 CPUs). Different
nodes and a single trial per setup limit causal claims. Current runs have no
learner; single-GPU optimizer memory/performance is not established here.

Output: `/gpfs/scrubbed/zixianma/checkpoints/web/render-policy-295631/`.
Do not interpret policy-step shutdown after the renderer finishes as an RL
failure; it is a background inference server explicitly stopped by the job.

## Eight-request follow-up

Job 295631 completed successfully: 810 turns, zero failures, 1506.75s cooking
phase; Slurm elapsed 26m07s including startup/shutdown. Mean environment step
0.183s, HTTP roundtrip 14.332s, server queue 6.749s. Phase throughput 967.65
turns/GPU-hour; allocation-inclusive throughput about 930 turns/GPU-hour.

User approved the recommended batching test for 1 or 2 hours; selected the
smaller one-hour ceiling. Submitted job 295683 using
`profile_render_batch8.sbatch`: 2 H200, 16 CPUs, 240 GiB, max 2 GPU-hours.
Only the policy server request limit changes from 4 to 8. Same eight concurrent
environments, four cases, 16-turn prefixes and sampling configuration.
This remains explicitly rollout-only, with no optimizer updates or W&B training
curve. Switching the renderer GPU back into the two-GPU learner is not yet
integrated or validated. Results go to
`/gpfs/scrubbed/zixianma/checkpoints/web/render-policy-295683/`.

## Four-request stability retest

295683 failed after 23m15s with an HTTP read timeout (240s), after 304
completed turns. Mean request latency 22.96s despite mean queue time 0.042s;
no explicit OOM/crash appeared in server logs. Underlying cause unconfirmed.
User approved retesting the recommended four-request configuration.

Submitted 295755: same one-hour ceiling, 2 H200 / 16 CPUs / 240 GiB,
one dedicated renderer plus one policy GPU, max 2 GPU-hours. Launch:
`sbatch --job-name=cooking-render1-policy1-batch4 examples/interact/profile_render_batch8.sbatch 4`.
The script now accepts request limit 4 or 8 (default remains 8 for compatibility).
Only request concurrency changes relative to 295683; timeout and workload remain
unchanged. Still no learner updates or full-episode RL claims.
Artifacts: `/gpfs/scrubbed/zixianma/checkpoints/web/render-policy-295755/`.
