# Cooking RL in the shared Slime environment

> Historical scripted-user planning and debugging log. The supported CookSim
> setup is maintained in
> [COOKING_GEMINI_HANDOFF.md](COOKING_GEMINI_HANDOFF.md),
> [RUNBOOK.md](RUNBOOK.md), and [PROGRESS.md](PROGRESS.md).

## Full training first segment: 295856

User replaced the proposed eight-hour allocation with **2 H200 x 4 hours**,
16 CPUs, 240 GiB, maximum 8 GPU-hours. Submitted **295856** using
`cooking_full.sbatch`. Target 15 actual updates, not a promise to finish in 4h.
Baseline and validation use all 10 held-out scenarios x 4 attempts at completed
updates 0, 5, 10, 15. Train batches use 6 groups x 8 complete native episodes.
The 40/10 frozen split is unchanged. Native binary reward and no-variance stop
remain active. CPU tests cover the 15-update loop, evaluation schedule, checkpoint
ordering and identity guards; full learner/restore remains GPU-unverified.

The placeholder-based GPU layout is the one verified in debug 295840. Learner
and sampler save after every update; only then is `resume-state.json` committed.
Startup refuses mismatched checkpoint pointers, sampler counts or experiment
metadata. Continuations read the original W&B ID and require that run to exist.
If killed mid-rollout, the incomplete batch is not a training checkpoint.
If killed mid-save, manual inspection is required rather than silent rollback.

Run root: `/gpfs/scrubbed/zixianma/checkpoints/web/cooking-full-295856/`.
Logs: `interact-runs/cooking-full-295856.out`; progress and metrics: run root
`audit/`; W&B receipt: `checkpoints/wandb_run.json`.
To continue after separate compute approval, submit with `COOKING_RUN_DIR`
explicitly set to this same root. No automatic resubmission.
Soft cutoff stops starting new training batches after 3.5h; supervisor cutoff
at 3h55m leaves cleanup time within the four-hour Slurm allocation.

## Weight-transfer fix validated

Acceptance 295811 failed in initial weight transfer, before baseline or updates.
The ad-hoc placement slice put SGLang on physical GPU 1 but reported logical
engine offset 0 to the colocated CUDA IPC updater. Replaced that slice with
official SGLang placeholder GPU 0 + regular GPU 1 configuration in
`configs/cooking_gpu_layout.yaml`. Both placement and weight-routing metadata
now retain offset 1. Two rollout *slots* include the empty placeholder; there
is still only one actual TP1 inference engine.

Approved debug job **295840** completed on the same node g007 in **1m51s**,
exit 0, inside a 30-minute / 1 GPU-hour ceiling. Three full weight transfers,
offload/onload cycles and generated responses passed with CUDA synchronous
error reporting enabled. Runtime assertion confirmed counts=[1], offsets=[1].
No optimizer updates, cooking episodes, or W&B logging in this test.
Evidence: `interact-runs/cooking-q35-295840/{weight-routing.json,transfer-debug.json,progress.json}`.
This fixes the observed startup crash; complete cooking reward variance,
learner updates and checkpoint restore remain to be tested. No full acceptance
retry or 15-update run has been submitted after this debugging job.

## Latest approved acceptance (supersedes allocation proposals below)

Job **295811** submitted with `cooking_hardware_acceptance.sbatch`: 2 H200,
16 CPUs, 240 GiB, two-hour ceiling / 4 GPU-hours. No automatic retries.
This tests a real learner, unlike the completed 16-turn throughput profiles.
The full target remains 15 updates, 6 groups x 8 complete episodes/update,
40 train / 10 validation, full validation (40 episodes) at completed updates
0, 5, 10, 15 and checkpoints every update. See `cooking_training_plan.py`.

Acceptance uses existing calibration membership (4 validation x 2 attempts,
then 2 training groups x 4 episodes), binary native reward, and the original
base model. Zero within-group reward variance stops before training. No
post-update full evaluation: update 1 checks parameter changes, frozen vision,
weight synchronization and independent checkpoint/optimizer/sampler restore.
This is not a claim of policy improvement or full-validation performance.

The acceptance mode assigns one TP1 SGLang engine to the second GPU, and checks
Chromium's actual GPU-process placement against the first GPU UUID. Vulkan
hardware mode fails closed on CPU fallback or context loss. After episode
workers close, no graphics processes may remain when the TP2 learner starts.
The original calibration mode and frozen split files remain unchanged.

Controller reserves separate preflight, kernels, training, restore, verification
and cleanup budgets inside two hours. Artifacts:
`interact-runs/cooking-q35-295811/` and
`/gpfs/scrubbed/zixianma/checkpoints/web/slime-cooking-295811/calibration01/`.
W&B uses a new cooking-only identity saved beside the checkpoint, never the
ScreenSim curve. Local verification: 15 targeted tests passed; GPU integration
acceptance is pending this job's result.

Prepared 2026-09-14. The first approved calibration, job **295354**, failed during
baseline evaluation; see its result below. A budget-v2 rerun is prepared but NOT
submitted or compute-approved. The current batch script requests seven hours.

## Experiment

Use the same official Slime checkout, isolated Qwen3.5 runtime, native adapter
protocol, multimodal generation hook and episode-normalized GRPO loss as ScreenSim.
Start from the original local Qwen3.5-4B checkpoint, not the ScreenSim-trained weights;
cross-domain transfer would be a separate experiment. Freeze vision, use LR 5e-7,
temperature 0.8, top-p 1, no additional KL or entropy coefficient, TP2 learner and
two TP1 SGLang engines. Preserve raw completions and full episode audits.

Cooking uses native v5 frames, baseline persona, deterministic environment seed 0,
and zero simulator-tick assistant latency. Its programmatic cook replans from live
kitchen state and accepts engine-credited flags. No paid human-model calls.
Acceptance is oracle-assisted and does not establish natural-language persuasion.

Use `native_outcome_v1`: reward 1 for native `won`, otherwise 0 for completed
`wrong_serve`, `burned`, or `timeout` outcomes. Infrastructure aborts are failures of
the run, not zero-reward episodes. This is native cooking success, not ScreenSim's
reference-duration deadline formula. Log detection and false flags separately;
do not silently switch to the existing experimental shaped reward.

## Data

`prepare_cooking_rl.py` freezes all 50 shipped composite cases: 40 training, 10
validation, with one SHA256-selected held-out variant in each of 10 recipe/layout
cells. Training order interleaves cells. Recipe and layout overlap is intentional;
the split measures scenario-variant generalization, not unseen-recipe performance.
Selection uses no model scores. Four fixed validation scenarios are also selected
for the calibration only. Record manifest and native source hashes with every run.

## First bounded allocation proposal

2 H200 GPUs, 16 CPUs, 240 GiB RAM, maximum 3 hours (6 GPU-hours).
This allocation is submitted as job 295354; no additional allocation is authorized.

1. Verify model/runtime, cooking screenshots and Qwen tokenization/context lengths.
2. Evaluate the base checkpoint on 4 calibration validation scenarios × 2 samples.
3. Collect 2 training scenario groups × 4 rollouts = global batch 8 episodes.
4. If reward groups have usable variance and inputs fit, run one real optimizer
   update; check finite gradients and actor-to-rollout weight refresh.
5. Evaluate the same 4 × 2 validation attempts after that update, save and verify
   weights, frozen vision and optimizer/sampler state. Retain all artifacts.

Total planned model episodes: 24. The controller must own and await workers,
reserve cleanup/checkpoint time, and stop within the allocation. If all rewards
are equal, record that diagnostic and stop rather than claiming a useful update.
Do not promise all stages finish before measuring cooking throughput. Existing
native silent tests take 76 assistant decisions in one case, so ScreenSim wall-time
estimates do not transfer. Context and response limits require actual cooking
prompt checks; malformed/truncated output rates must be logged.

## Full experiment after calibration

Target the comparable 6 groups × 8 episodes = global batch 48 and 15 total updates,
with full 10-scenario validation at 0, 2, 6, 9, 12, 15 (4 samples/scenario = 40).
Request the full compute budget based on observed episode lengths and throughput.
Use fixed eval sampling seeds and persist explicit scenario/attempt/update IDs.

Log one persistent cooking run in `zixianma/interact-slime-rl`; reuse its saved ID
for all cooking continuations. Never append cooking metrics to the ScreenSim curve.
The cooking calibration now has its own thin entry point and metric callbacks,
reusing the official Slime actors, rollout manager, generation and reward hooks.
It saves a W&B identity receipt beside its checkpoint. The calibration entry point
intentionally starts only from the base model; future cooking continuations must
use this receipt with verified cooking-specific checkpoint/history guards.
The restore-only stage disables logging and performs no additional updates.

Implementation: `run_cooking_calibration.sh`, `train_cooking_calibration.py`,
`cooking_calibration_{controller,preflight,metrics,actor,verify}.py`.
Local checks: six selected native-parity, reward, split and metric tests passed;
Python compilation and shell syntax passed. Full argument validation requires a
GPU (Megatron probes device architecture), so CPU-only parsing was not a GPU test.
All owned stage logs and status files go to `interact-runs/cooking-q35-295354/`.

Main plots: episode-weighted train/eval native success, cell-macro success,
mixed-reward group fraction, detected/planned errors, false flags, invalid responses,
episode turns, native ticks, outcome counts, gradient norm, and throughput.
Persist native task, observations, responses, injections, actions, terminal outcomes
and sampling identity for a paired cooking replay at 0 versus 12 completed updates.

## Calibration 295354 result (2026-09-14)

Monitored through termination: Slurm FAILED (exit 1), elapsed 01:07:18 on
g011 with 2 H200s (2.243 allocated GPU-hours). Environment/processor preflight
and GPU kernel checks passed. Baseline inference ran, but no training rollout
batch, optimizer update, checkpoint verification or restore stage was reached.

Six baseline episodes completed normally (easy, medium and expert cases, two
attempts each); all ended in wrong_serve. Both master-case attempts aborted on
the native wall-clock limit at ticks 309 and 313. Composite tasks double the
configured 1800-second native budget to 3600 seconds. The reward guard correctly
rejected these aborted reports, causing evaluation and the controller to fail
closed. This is an incomplete baseline, not an 0/8 valid success measurement;
training-group reward variance remains unmeasured.

W&B identity: https://wandb.ai/zixianma/interact-slime-rl/runs/4110irup
Artifacts and controller status remain in the paths above. No replacement job
was submitted. Before another allocation, address the measured long-case rollout
runtime/budget and retain the rejection of infrastructure-aborted episodes.

## Prepared budget-v2 rerun (not submitted)

The master reports show continued native progress through ticks 309/313, not a
deadlocked inference server. Increase the native base wall budget from 1800 to
3600 seconds (composite effective limit: 7200 seconds) and the bridge decision
limit from 300 to 600. Keep failure-closed handling of aborted reports. Do not
change reward shaping, model, human, simulator latency, sampling temperature,
scenario membership/order, calibration subset, or episode counts.

New immutable dataset profile: `interact-runs/cooking-rl-v1-20260914-budget-v2/`.
The original profile and failed-run artifacts are preserved. All row and source
hashes remain checked; the changed timing config intentionally creates new group
identities/sampling seeds. Comparisons must pair baseline and updated policies
within the new run, not treat the old run's attempts as identical seeded samples.
The shell, driver and preflight share `COOKING_SPLIT_DIR`; stale profiles or old
decision limits fail validation before model startup.

Proposed ceiling: 2 H200s, 16 CPUs, 240 GiB RAM, 7 hours / 14 GPU-hours. This is a
conservative reservation, not a predicted runtime: allow up to two hours for each
of the three rollout phases, then startup/update and restore/verification time.
The controller stops failed stages immediately and reserves downstream stage
budgets. Its global cutoff is 3 minutes before Slurm's limit. No automatic retry.
Training still stops before an optimizer update if both groups have no reward
variation; longer budgets do not guarantee a useful sparse-reward RL signal.

Local validation: 11 selected tests passed, including an accelerated-clock native
master-case regression (old budget aborts, new budget completes), unchanged split
membership, stale-profile rejection, native parity, and aborted-reward rejection.
The accelerated test uses a silent policy and no rendering/model calls; it is not
evidence of Qwen performance or a GPU-throughput guarantee. GPU training and
restore remain to be tested in the next approved allocation.

Readiness check: `../.venv/bin/python examples/interact/cooking_calibration_budget.py`
from the Slime directory. Once compute is explicitly approved, submit
`sbatch examples/interact/cooking_qwen35_calibration.sbatch`.
The new run writes `progress.json` / `COOKING_PROGRESS` markers for startup,
baseline evaluation, training rollouts, optimizer update, checkpoint save,
post-update evaluation, and restore, including the last failed phase.
