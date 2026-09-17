# Cooking versus ScreenSim rollout timing

2026-09-14: CPU recorded-action replay, not fresh model inference. Raw measurements:
`interact-runs/rollout-profile-cpu-v1/summary.json` and per-worker
`worker_timing.jsonl`. Same local Qwen3.5 processor, two Torch threads, sequential
episodes on the login host. Browser CPU affinity was not constrained; this is NOT
a 16-versus-32 CPU scaling measurement. Cancellation at the requested prefix is
intentional; these prefixes are not training examples or success evaluations.

| Case | Timed turns | Environment step (s) | Browser API, steady window (s) | Image decode (s) | Template + processor (s) |
|---|---:|---:|---:|---:|---:|
| Cooking master/hard | 12 | 11.959 | 11.847 | .0183 | .1042 |
| Cooking medium/medium | 12 | 9.971 | 9.973 | .0173 | .1027 |
| ScreenSim parent_phone_readability | 8 | .1088 | .0946 | .0089 | .0227 |
| ScreenSim standby_bedside | 5 | .1218 | .1042 | .0105 | .0224 |

Browser API means cover worker segments before decisions 1..N-1; parent step
means cover all N responses, including the final transition. They are overlapping
measurements with slightly different windows, NOT additive components. Browser
calls include browser IPC, 3D rendering or HTML capture, and image export; they
are not GPU-kernel timings. Audit writes averaged .004-.007s cooking and .0015s
ScreenSim. Startup was 17-18s cooking and 2.4-6.1s ScreenSim, excluded above.
Cooking observations averaged 6.67 images / ~5,000 prompt tokens over these early
prefixes; ScreenSim averaged 1.8-1.9 images / ~2,200-2,600 tokens.

The result establishes a large renderer-side difference on this host. It does
not establish the rendering backend or the exact breakdown of the previous H200
job. Cooking selects Vulkan when an available ICD library can load, otherwise
SwiftShader; Vulkan selection alone does not prove hardware rendering. A new
opt-in browser backend probe records the actual WebGL renderer for future runs.
No rendering resolution, cadence, style, model response limits, or task horizons
were changed by profiling.

Instrumentation (`INTERACT_PROFILE=1`) now retains client preprocessing,
HTTP round-trip, environment step and audit timings. Server timing fields are
retained when returned; enable SGLang metrics for queue and inference timestamps.
Client HTTP round-trip is not pure model time. Missing server fields must remain
unknown, not be inferred from an episode-wide average. Shared-hook tests with
mock inference passed (3 selected tests); Python compilation/diff checks passed.

## Approved scaling measurement

The initially proposed 2-GPU/32-CPU allocation was invalid: Tillicum requires at
most 8 CPUs per full GPU. The user approved two separately billed allocations:

- **295509**: 2 H200s, 16 CPUs, 240 GiB, at most 1 hour (2 GPU-hours), g011.
- **295510**: 4 H200s, 32 CPUs, 240 GiB, at most 30 minutes (2 GPU-hours), g022.

Both run eight concurrent environments, one TP1 Qwen3.5-4B SGLang server per GPU,
same model and deterministic per-case/attempt/turn sampling seeds, 512 response
token cap, and frozen ordered workloads in `configs/scaling_profile_v1.json`.
Each splits its available rollout time equally between cooking then ScreenSim.
Profiles stop at 16-turn prefixes (or native termination), never train on these
prefixes, and report completed turns/segments/full episodes separately.
This is live policy inference, not replaying recorded actions. Queues advance as
workers become available; batching and renderer differences may change exact
trajectories. These jobs test combined resource scaling, not CPU-only causality.

Server startup, client encode/request/environment times, native browser/API audit
times, actual WebGL renderer, and GPU utilization (5-second samples) are retained.
Queue/prefill/inference metadata is enabled. Browser timings include API overhead;
server timestamps are not pure CUDA-kernel measurements. Compare per-engine
progress, equal-time steady throughput and actual allocated GPU-hours; mixed
totals at different engine phases are misleading. Startup counts toward cost.

Artifacts: `/gpfs/scrubbed/zixianma/checkpoints/web/engine-scaling-JOBID/`.
Logs: `interact-runs/scaling-JOBID.out`. Read the current comparison with
`../.venv/bin/python examples/interact/report_scaling.py` from Slime.
Each job also writes a comparison snapshot when it finishes. No optimizer
updates or paid model API calls. No additional compute beyond these two jobs.
