# Unified native assistant environments

## Source and current status

This is the active implementation, based directly on official
[THUDM/Slime](https://github.com/THUDM/slime), pinned at
`4c193f1f37509cca70f0e88807a9305b70f63f4e`, branch `unified-assistant-env`.
The checkout is `/gpfs/home/zixianma/interact/slime` and the public fork is
[zixianma/slime](https://github.com/zixianma/slime). Targeted Slime changes support
multimodal storage, Qwen VL models, resumable W&B logging, and safe weight-transfer
reconnection. See [RUNBOOK.md](RUNBOOK.md) for operation and
[PROGRESS.md](PROGRESS.md) for the concise verified status.

The sibling `slime-cooking/` OpenWebRL-based prototype is reference only, **not
the project base**. Its GPU launcher and its trainer-specific assumptions do not
apply to this checkout. Only isolated native cooking code and tested token-accounting
helpers were carried over; there are no OpenWebRL imports in this implementation.

The environment API is engine-neutral. Cooking and ScreenSim have native adapters;
VH remains an explicit pending entry, not a working integration. **ScreenSim is the
first GPU/RL target**, using Qwen2.5-VL-3B-Instruct. Native ScreenSim direct/IPC prompt
parity passes, as do silent/oracle controls (native F1 0/1). Two real screenshot-based
SGLang trajectories completed through the official Slime hook, both with F1 zero.
Subsequent official Slime training completed two optimizer steps, including a
nonzero-gradient update, checkpoint saving, weight synchronization and a separate
checkpoint-resume run. This is a pipeline smoke test, not measured benchmark gain.
See [GPU_VALIDATION.md](GPU_VALIDATION.md) for results, artifacts and limitations.

| Layer | Shared responsibility | Engine-specific responsibility |
| --- | --- | --- |
| `EpisodeSpec` | Engine/task/seed identity and private configuration | Human/persona, native timing/observation/reward configuration |
| `Observation` / `Decision` | Allowlisted system/user/images, ordered images, content hash, decision ID | Native prompt construction, disclosure policy, timestamp unit, action schema |
| `Action` | Raw assistant generation tied to the exact episode/decision | JSON parsing, typed flags, memory fields and response delivery |
| `Environment` / worker | Async reset/step/close, isolated process, protocol checks, bounded wait, owned cleanup | Native scheduler, human turns, physical execution and render resources |
| `EpisodeResult` | Terminal/truncated eligibility, reward version/components, artifact references | Native grading and task-specific reward calculation |
| Slime bridge | Exact token/image/logprob accounting, trajectory identity, GRPO grouping | No engine-specific branches |

This is **assistant supervision** in all three domains, not a unified physical-action
Gym environment. The simulated person remains inside the environment. Each adapter
preserves its native episode, prompt builder, memory policy, flag taxonomy and clocks.

## Adapter boundaries

- **Cooking**: native `e2e_stepin_rollout.run`, using `sup_client.sup_call` after
  the native system/user/frame request is constructed. Owns its renderer/browser
  when requested. Engine unchanged at `f6ea8d2f677ca2f92759e13affd23208cd2d33b9`.
- **ScreenSim**: preserve `AssistantV2.turn` and intercept its client's
  `text(prompt, images=...)` boundary. Do not serialize the oracle-only `dev`,
  `ir`, or `open_beats` callback arguments into observations. Uses native scripted
  hands plus a deterministic dialogue-human fixture accepting native-verified
  corrections. This is a separately named `scripted_human_v1` profile, not the
  Gemini-human/vLLM-JSON-mode baseline. Ordered screenshots and native F1 are retained.
- **VH (next)**: intercept `iv_protocol.assist_frames` after `sup_turn` constructs
  the native prompt and ordered first-person/map images. Preserve the currently
  omitted text-map arguments for initial parity; any repair needs a new profile.
  A Unity instance must be leased, not silently shared or killed by generic cleanup.

`registry.py` advertises implementation status and resource requirements. It does
not yet allocate resource pools. Attempting an unimplemented adapter fails before
spawning a worker. New adapters implement `NativeAdapter.run(spec, session)`, call
`session.request(Observation(...), tick=..., time_unit=..., action_schema=...)`,
and return `EpisodeResult`. The same protocol supports different time units and
native response schemas. A test-only independent adapter exercises that boundary;
it is not a substitute for testing ScreenSim or Unity.

## Run cooking on CPU

The existing Python-3.12 dependency overlay and Playwright installation are reused
for CPU testing only; neither the shared training runtime nor active baseline is
modified. `env.sh` points `PYTHONPATH` at this official checkout and isolates caches.
For another installation, install `requirements.txt` plus the additional packages
in `examples/interact/requirements.txt`, supply Playwright and the engine checkout,
and set the engine's `config.engine_root` explicitly if it is not a sibling directory.
That is not a portable GPU-stack installation recipe.

```bash
cd /gpfs/home/zixianma/interact/slime
source examples/interact/env.sh
python -m pytest -q -o addopts='' tests/interact
python -m interact_env.smoke --spec examples/interact/configs/cooking.json
python -m interact_env.smoke --spec examples/interact/configs/cooking_frames.json --max-decisions 2
python examples/interact/prepare.py \
  --spec examples/interact/configs/cooking.json \
  --output interact-runs/prepared-cooking
```

The preparation command requires a new output directory and writes a JSONL task
file plus source/package provenance. Repeat `--spec` for additional tasks/configurations
as their adapters become available. Training always groups continuations of the
same complete episode specification; do not normalize across mixed engine cases.

The cooking profile uses a scripted human, baseline persona, native v5 scheduler,
and explicitly fixed zero-tick assistant delay. This keeps inference/IPC wall time
out of simulator time. It is a test/training variant, not measured-latency evaluation.
No paid model API calls occur. Native Gemini human mode is not validated here.

## Official Slime integration

Add these hooks to a separately validated upstream model/training recipe:

```text
--custom-generate-function-path interact_env.slime_bridge.generate.generate
--custom-reward-post-process-path interact_env.slime_bridge.rewards.normalize
--custom-config-path examples/interact/configs/rollout.yaml
--prompt-data interact-runs/prepared-cooking/tasks.jsonl
--input-key prompt --metadata-key metadata
--advantage-estimator grpo --rollout-top-p 1
--rollout-batch-size 1 --n-samples-per-prompt 2 --global-batch-size 2
--use-rollout-logprobs
```

This is a **hook/config fragment, not a complete GPU launcher**. Supply checkpoint,
model provider, parallelism, context/response limits and resource configuration
from a validated official-Slime recipe. The generator deliberately rejects partial
rollout, group-RM and nucleus sampling until their contracts are implemented.

Official Slime now uses `Sample.rollout_id` for multiple training samples from one
episode. Every assistant turn preserves that parent ID and gets a distinct sample
index. The custom reward hook normalizes the terminal reward once per episode within
a prompt group, then broadcasts its advantage to the turns. It rejects missing
turns and mixed episode/reward configurations.

Upstream `rollout_mask_sums` supplies **one token-weighted mean per episode** across
its turns and microbatches. We do not copy OpenWebRL's `1 / num_turns` weighting or
dynamic-global-batch patch. Global batch size counts two episodes, not 152 turns.
Equal-turn weighting would be a separate loss choice. Standard flattened rollout
statistics may still be turn-weighted; use episode reports for benchmark means.

Actual sampled token IDs and likelihoods come from SGLang. Only response tokens
receive loss; no forced end token, newline, or fabricated likelihood is appended.
Native parsing changes only the engine action. Context overflow, inference abort,
invalid likelihoods and infrastructure failures fail the smoke rather than become
negative task rewards. Partial/uncertain delivery is not automatically replayed.

## Reward and validation limits

ScreenSim also exposes the native **in-time success** objective:
`success = int(goal_ok and final_tick <= 2 * reference_plan_end_tick + 8)`.
This deadline is measured in simulator ticks, not inference wall-clock seconds.
`screensim_f1_v1` remains the historical/default smoke reward. The opt-in
`screensim_intime_success_v1` directly rewards this binary success value; use
`configs/screensim_intime.json` with the preparation command and pass the resulting
task file as `--prompt-data`. No unvalidated F1/success mixture is silently applied.
All six completed training/resume episodes in job 289525 had in-time success zero,
including the F1=0.667 episode. Detection improvement alone is not task success.

The silent scripted-human cooking control still wins in **151 ticks / 76 assistant
decisions**, despite two offered errors and zero detections. Therefore success-only
reward does not establish useful assistance.

The optional experimental `assistant_detection_v1` is
`0.5 * won + 0.5 * detected_errors / planned_errors - min(0.1 * false_flags, 1)`.
It scores that silent rollout 0.5. `native_outcome_v1` remains an outcome baseline.
Neither is claimed to replace CookSim's official evaluation. QA quality, persuasion,
persona satisfaction and latency quality require separate validation/scoring.
Reward calculation is inside the cooking adapter, not the common trainer bridge.

CPU tests cover native cooking/ScreenSim direct-call/IPC parity, cancellation and concurrent workers,
observation allowlisting, adapter-independent free-text actions/fractional timestamps,
real rendered images and Qwen image-token alignment, exact completion token handling,
the complete hook with **mock inference**, and real official-Slime train-data conversion
and episode reward normalization. CPU tests do not establish model inference or learning;
the separate GPU rollout probe establishes real inference but not an optimizer update.

## Separate W&B project

The ScreenSim launcher now enables tracking in **`interact-slime-rl`**. It uses
`INTERACT_WANDB_MODE=offline` by default until cloud authentication and destination
approval are available. Set `INTERACT_WANDB_MODE=online` and
`INTERACT_WANDB_ENTITY=<approved-entity>` for future approved training runs. Use a
saved `wandb login` or `WANDB_API_KEY` environment variable; never pass a key through
`--wandb-key`, since upstream records the parsed arguments as run configuration.
The launcher clears inherited run/resume/sweep IDs so it cannot attach to a baseline
run, and keeps local W&B data in that training attempt's scrubbed directory.

`interact_env.slime_bridge.metrics.log_rollout` adds **one vote per episode**, not
per assistant turn. Metrics are split by engine and reward version. For example:
`rollout/episodes/screensim/screensim_f1_v1/in_time_success` and the sibling `f1`,
`goal_ok`, `reward`, `false_flags` and `final_tick` metrics. Upstream training loss,
gradient norms, learning rates, logprob differences and performance logging remain
enabled. Optional components with missing values expose their sample counts.

Completed stages can be backfilled with `examples/interact/backfill_wandb.py`.
It rejects failed stages and incomplete episode histories, uses a fresh run ID,
and writes only scalar metrics plus allowlisted provenance—not prompts, images,
transcripts, checkpoint files, raw logs or credentials. Job 289525's three optimizer
steps were backfilled offline under
`interact-runs/screensim-289525/wandb-backfill-v2/`. They are explicitly marked as
historical data, not a new training run. Cloud synchronization has not occurred.

The user-provided `/gpfs/projects/krishna/zixianma/OpenWebRL/.env` authenticates as
W&B entity `zixianma`. `with_wandb_env.py` reads only W&B authentication from that
file into the worker environment; it does not source shell code or import the
baseline project/run identity or other provider credentials. The approved
`screensim_intime_test.sbatch` targets `zixianma/interact-slime-rl`: three fresh
GRPO batches of four episodes with `screensim_intime_success_v1`, saving the final
checkpoint. It requests 2 H200 / 16 CPU / 240 GiB for at most 30 minutes (one
GPU-hour). Submitted as job **290990** on **g004** after explicit approval;
all 23 preflight tests passed. The job completed in **7m25s**, saved its final
checkpoint, and released both GPUs. All **12 episodes had zero in-time success**,
so all three gradient norms were zero: pipeline validation, not learning progress.
Live tracking (finished, all three steps verified against local records):
[screensim-qwen25vl3b-290990-intime-small](https://wandb.ai/zixianma/interact-slime-rl/runs/ujzg3ikd).
This is a fresh online run, separate from the historical offline backfill and the
OpenWebRL baseline. See `GPU_VALIDATION.md` for verified outcomes.

## GPU readiness and next milestones

### Frozen-policy Qwen comparison (job 292136)

The user approved one ScreenSim-only allocation of **3 H200, 24 CPUs, 240 GiB,
2 hours maximum / 6 GPU-hours**. Submitted as **292136**, node **g021**, scheduler
estimate **$5.40 maximum**. All 26 CPU tests and all three processor/model-class
preflight checks passed. `screensim_compare.sbatch` owns three independent
one-GPU workers, one per Qwen2.5-VL-3B-Instruct, Qwen3-VL-4B-Instruct and
Qwen3.5-4B checkpoint. This performs **no optimizer updates** and uses **no paid
model API**. Cloud scalar metrics go to `zixianma/interact-slime-rl` as three new
runs in a shared job group. Credentials are loaded only inside the allocation.

Live runs for job 292136:
[Qwen2.5-VL-3B](https://wandb.ai/zixianma/interact-slime-rl/runs/1b3cc2cddfa24fe38e14b817993f7809),
[Qwen3-VL-4B](https://wandb.ai/zixianma/interact-slime-rl/runs/5ce69208d63b4bc1b0be31f38dc49219),
[Qwen3.5-4B](https://wandb.ai/zixianma/interact-slime-rl/runs/fb09894d9a594cebb701135985598981).
All three loaded successfully and began native episodes. Local results are under
`screensim-qwen-comparison/job-292136/model-{0,1,2}/`; each worker writes a final
`summary.json`, and the owning controller writes `controller-status.json` before
exiting. These are **running comparisons, not final results**. For a read-only
cross-model coverage and per-scenario confidence/contrast report:

```bash
source examples/interact/env.sh
python examples/interact/report_comparison.py --job-dir \
  /gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison/job-292136
```

`prepare_comparison.py` pins public model revisions and certifies the split before
GPU submission. The frozen manifest is under
`/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison/manifest.json`.
The 7 training tasks have 21 templates, 16 attempts each; the 4 held-out validation
tasks have 9 templates, 8 attempts each: **408 episodes per model / 1,224 total**.
Validation tasks are `lock_screen_lockdown`, `siri_lock_screen_privacy`,
`rent_share_standard` and `rent_landlord_instant`. All variants of a task stay
together. The human is `scripted_human_v1`, baseline persona only—not the complete
three-persona benchmark. Environment seed remains 0; different model samples are
repeated attempts, not new initial states. Validation informs model selection and
is not a pristine final test set.

### Approved RL split revision (18 train / 12 validation)

The subsequent Qwen3.5-4B RL pilot uses a separate, frozen split:
`/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-v2-18train-12val/manifest.json`.
Its SHA-256 is `a29464102278d784a06a6b5a57ab6cb3dd53b2da6a74015bab54f8687a3d96f9`.
All three `hearing_setup` scenarios move to validation to add accessibility
coverage, leaving **6 training tasks / 18 scenarios** and **5 validation tasks /
12 scenarios**. Whole tasks remain disjoint. Native episode specs and rewards
are unchanged. The original comparison manifest and historical W&B split labels
must not be rewritten. This remains development validation, not an untouched test.

`prepare_rl_split.py` builds this version without modifying its source and rejects
an existing output directory. Beside the manifest, `train.jsonl` and
`validation.jsonl` contain Slime-compatible prompts and private episode metadata.
Offline attempt counts are intentionally not an RL schedule.

First pilot: user-approved job **292972**, with runtime/GPU validation in progress:

- Qwen3.5-4B, native in-time-success reward, frozen vision, GRPO with 8 samples
  per scenario, 6 scenarios per rollout batch, initially 3 rollout/train cycles
  covering all 18 training scenarios. Start with LR 5e-7.
- Fixed pre/post evaluation on all 12 validation scenarios, 4 attempts each;
  identical evaluation decoding and seeds. No validation gradients. Log
  checkpoint-specific task-macro success, not cumulative means across checkpoints.
- Separate W&B run in `zixianma/interact-slime-rl`, recording split version/hash,
  model revision, optimizer progress, group reward variation and infrastructure
  failures. Missing or interrupted evaluations are not zero-success observations.
- Approved ceiling: 2 H200 GPUs, 16 CPUs, 240 GiB RAM, 2 hours (4 GPU-hours), no
  paid model APIs. Slurm estimated $3.60; job 292972 is on g001.
- Slime includes native Qwen3.5-VL support, but the tested Qwen2.5 RL runtime's
  SGLang 0.5.6.post2 lacks Qwen3.5. Prepare a separate runtime; do not upgrade the
  shared baseline. GPU checks must cover loading/export, multimodal logprob
  consistency, actual parameter updates, inference weight refresh and checkpoint
  save/restore before treating the pilot as a validated training run. Runtime
  failures may prevent completion of the planned cycles within the ceiling.

The job-scoped `qwen35_controller.py` owns every training worker and bounds setup,
stage execution and cleanup. Queue/status logs are in
`interact-runs/qwen35-rl-292972/`; large artifacts are in
`/gpfs/scrubbed/zixianma/checkpoints/web/slime-qwen35-292972/`.
Startup attempts are preserved separately; creation of a W&B run is not evidence
of an optimizer update. Current candidate `pilot09` enables deterministic SGLang
sampling so fixed per-scenario/attempt/turn validation seeds are actually honored.
The earlier `pilot` through `pilot06` attempts failed before any optimizer updates.
`pilot05` completed baseline validation (25/48 successes, 60% task-macro success)
and the first training batch (33/48 successes). `pilot09` reuses those exact
pre-update collections, then collects a fresh second batch. A local CP=1 odd-length
vision-index fix unblocks learner execution. cuDNN FusedAttention produced NaN
gradients in `pilot07`, localized by anomaly detection in `anomaly08`; neither
updated weights. The new candidate forces official FlashAttention 2.8.3 and is
bounded to two cycles plus full validation to fit the same allocation.
Active run: https://wandb.ai/zixianma/interact-slime-rl/runs/6neavc52.
That pilot subsequently completed both updates, validation, parameter/frozen-vision
checks, and restore-only verification. Validation changed from 25/48 to 27/48
successes and from 60.0% to 61.6667% task-macro success; this is not a statistically
established gain. The allocation ended after 1h47m39s. Its aggregate Slurm FAILED
status preserves earlier setup failures; stages 30, 31, and 32 all exited zero.

Approved continuation **293612** uses `screensim_qwen35_continue.sbatch`: 2 H200,
16 CPUs, 240 GiB, at most 3 hours / 6 GPU-hours (submission estimate $5.40).
It resumes `pilot09/checkpoints/iter_0000001`, including optimizer/scheduler and
the saved sampler cursor (offset 12, group index 12, sample index 96), and performs
seven additional updates for nine total. The next batch covers the six scenarios
not visited by the pilot. Reward, decoding, LR, frozen vision, and split stay fixed.
All later batches are fresh; the pilot's job-specific replay hook is not used.
Evaluation runs only at completed updates 6 and 9, with four attempts per validation
scenario. The official epoch-boundary callback at update 3 is explicitly skipped
without logging a zero-success evaluation; the empty debug dump is not an eval.
The new W&B run records parent run `6neavc52` in its config and remains in the
`interact-slime-rl` project. New artifacts never overwrite the parent checkpoint.
The continuation controller owns and awaits preflight, training/evaluation, and
final checkpoint/metric checks, exits on failure, and reserves cleanup time.

User-approved merged scalar history (both original W&B runs preserved):
https://wandb.ai/zixianma/interact-slime-rl/runs/d0df345ba8f8.
`merge_qwen35_wandb.py` exported the complete unsampled histories from `6neavc52`
and `6kv7biu2`, validated their step coverage/configuration, and copied them into
one explicitly labeled history-merge run. Cloud read-back verified every copied
value across 88 scalar metrics. Training indices remain 0–8; validation completed
update coordinates remain 0, 2, 6, 9. Merge timestamps and system metadata are not
training timings; original checkpoints, episodes, and source runs were unchanged.
The local export and receipt are in `interact-runs/qwen35-merged-9updates/`.

### Reusing one W&B curve on future continuations

Qwen3.5 training now verifies W&B history before starting Ray when loading a
numeric Slime checkpoint. The destination is, in priority order, an explicit
`INTERACT_WANDB_RUN_PATH` (or `--wandb-run-id`), the parent checkpoint directory's
`wandb_run.json`, or this experiment's canonical merged run:
`zixianma/interact-slime-rl/d0df345ba8f8`. For this existing experiment, explicitly
select that canonical path when preparing each continuation. A different model,
split, or experiment should start a new run, not append to this one.

The primary logger now honors `--wandb-run-id` using `resume="must"`, retains the
existing run name/group, and passes the same ID to distributed workers. New runs
record their identity beside their checkpoints. Ordinary `WANDB_RUN_ID` variables
are still scrubbed to prevent accidental reuse of an unrelated shell/dotenv run;
the explicit `INTERACT_WANDB_RUN_PATH` survives the credential wrapper.

The read-only guard checks model/split/training settings, rejects a running target
run, and requires complete optimizer and episode-success history up to the loaded
checkpoint. Ahead-of-checkpoint history is rejected too. Training/rollout axes
remain zero-based update indices; evaluation axes remain completed-update counts.
No counters are reset and no historical points are rewritten. A history mismatch
requires reconciliation, not a silent new run or duplicate update indices.

Job **294135** was already running before this change and is intentionally not
restarted or redirected. Its updates 10–15 are logged to `h6izegu2`. Those points
must be appended and verified in the canonical run before continuing there from
its final checkpoint; otherwise the new guard will refuse launch. The merged run
has not yet been changed by this resume-support implementation.

The job's `train-round-robin.jsonl` interleaves tasks (6 different tasks per batch),
without altering membership or the frozen split. Input-order SHA-256:
`31a41ba735df38727fbcc8488c843be734a78ceb6ad411b066cfea47146d5fde`.

The new isolated runtime is `openwebrl-runtime/screensim-qwen35-rl`: Torch
2.9.1+cu129 is reused, SGLang 0.5.9 / sgl-kernel 0.3.21 / FlashInfer 0.6.3 / FLA
0.4.2 are isolated, Transformers 5.3.0 comes from the comparison overlay, and
Megatron is pinned to `1dcf0dafa884ad52ffb243625717a3471643e087` with Slime's
reference Megatron patch. The memory saver is built from Slime's referenced fork
commit `8d30c59ca12a68d9deccbc9c6599076a1218cbc5`. Optional Apex gradient
accumulation fusion is disabled. This is a site-specific integration, not a claim
that the entire dependency set matches Slime's reference container.
This profile sends unexpanded text plus images to SGLang and retains the independently
encoded learner token IDs, with a strict server/learner token-count check. It uses
the supported Triton batch-invariant matmul path rather than optional DeepGEMM.

### Offline comparison runtime

All three use a separate **Transformers 5.3.0 / HF batched SDPA inference** overlay,
BF16, four concurrent episodes, temperature 0.8, top-p 1, top-k disabled,
repetition penalty 1, 512 response tokens, context cap 16384, and non-thinking
chat templates. There is no constrained JSON or silent repair. Sampling settings
override model-specific generation defaults; exact token sequences are retained.
This is a fresh matched comparison, not a numerical reproduction of the previous
SGLang inference. CPU processor checks do not prove GPU compatibility or throughput.
The shared Slime/runtime installation is not upgraded.

`comparison_metrics.py` logs native outcomes, per-task macro means, parsing/schema
validity, truncation and complete groups-of-eight with reward contrast. The
success-plus-0.25-F1 metric is a **hypothetical diagnostic only**, not a training
reward applied to an optimizer. Infrastructure failures are recorded separately,
never silently assigned reward zero. Attempt order cycles across all scenarios
to expose task coverage early; sample completion order is asynchronous. A global
fixed Torch seed is recorded, but scheduling-dependent batching is not claimed
bitwise reproducible. Workers preserve completed episodes at deadline; incomplete
coverage and model failures must be reported, never silently extrapolated.

### Training integration status

The installed runtime has SGLang **0.5.6.post2** and Transformer Engine **2.10.0**;
the pinned official `docker/Dockerfile` references SGLang **0.5.15.post1-cu129**,
Transformer Engine **2.16.1** on CUDA 12, and Megatron commit
`1dcf0dafa884ad52ffb243625717a3471643e087`. These are reference versions, not an
assertion that all older versions are incompatible. Real SGLang inference works;
the narrow Qwen2.5-VL trainer profile below passes the GPU smoke. Do not upgrade the active
baseline runtime in place. The opt-in `slime_plugins/models/qwen25_vl.py` uses NVIDIA
Megatron Bridge's Qwen2.5-VL provider and registers model-specific HF load/export
mappings per process; upstream trainer/loss files remain unchanged. Initial limits:
frozen vision tower/projector, TP2, CP1, PP1, one sample per microbatch, no dynamic
multi-sample packing. These restrictions are not the final all-models contract.

1. Validate the Qwen2.5-VL load/export round trip and image-conditioned numerical
   logprob agreement between the learner and rollout server before trusting updates.
2. Run complete ScreenSim collection, optimizer steps, checkpoint save/restore and
   updated inference weights in official Slime. Distinguish a zero-advantage smoke
   update from a nonzero-gradient learning test; never invent reward variation.
3. Add invalid/false-alarm controls and multiple tasks/seeds. Make episode truncation
   explicit (the current native ScreenSim report does not expose budget exhaustion).
   Keep human/sampling/latency profiles versioned and benchmark comparisons matched.
4. Lock a reproducible isolated GPU dependency set after compatibility is established;
   the current overlay reuses shared packages without modifying the active baseline.
5. Extend the tested path to cooking and VH using the **same** runtime, with native parity and
   browser/Unity lifecycle tests, then validate homogeneous per-task GRPO groups
   inside a mixed-engine training batch and episode-weighted per-engine evaluation.

Production retries, preemption recovery, resource pools, human-provider budgets and
cross-engine reward calibration remain future work. The protocol makes room for
them; it does not claim they are already implemented.
