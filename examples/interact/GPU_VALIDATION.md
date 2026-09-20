# ScreenSim-first GPU validation

## Current CookSim Gemini-user RL path — September 19, 2026

The supported configuration is documented in
[COOKING_GEMINI_HANDOFF.md](COOKING_GEMINI_HANDOFF.md). A four-H200 run passed
renderer and shared renderer/policy preflights, operated four TP1 policy
servers, completed TP2 x DP2 optimizer updates on all four GPUs, checkpointed
every update, and completed a 20-episode validation at update 3. Validation
success changed from 13/20 at baseline to 16/20 at update 3; reward changed from
0.606 to 0.772. This is an active development result, not a final benchmark or
significance claim. The W&B run is
<https://wandb.ai/zixianma/interact-slime-rl/runs/kv1kjbup>.

The run also exercised disk-backed visual tensors, zero-loss static-DP padding
for variable multi-turn sample counts, first-four-of-five speculative rollout
groups, native Gemini API calls, decision-cap rewards, and exact checkpoint/W&B
resume guards. Historical sections below describe earlier ScreenSim and cooking
bring-up and should not be used as the current CookSim launcher.

Run: Slurm **289525**, node **g014**, September 11–12, 2026 (America/Los_Angeles).
Approved ceiling: **2 H200 GPUs, 16 CPUs, 240 GiB RAM, one hour / two GPU-hours**.
No additional allocation, paid model API, GitHub publishing, or baseline-job changes.
Final Slurm state: **COMPLETED, exit 0:0**, elapsed **33m27s** (**1.115 GPU-hours**).
The allocation was released early. The unrelated baseline job 288861 remained running.

## Result

The official Slime path completed two on-policy ScreenSim rollout/train cycles,
saved both distributed model/optimizer checkpoints, and pushed weights back to
SGLang. This is an integration smoke test, not evidence of benchmark improvement.

| Training rollout | Episode F1 rewards | Gradient norm | Mean learner/rollout logprob difference |
| --- | --- | --- | --- |
| 0 | 0, 0 | 0 | 0.00904 |
| 1 | 0, 0.667 | 3.68827 | 0.01208 |

The zero-gradient first batch is expected. The second batch has a genuine native
reward difference: no fabricated reward variation, oracle substitution, or entropy
bonus was used. Language weights actually changed: the first saved QKV shard has
7,938 changed BF16 elements, maximum absolute change 4.76837e-7. All **390** saved
vision-tower/projector tensor shards were compared and remain exactly unchanged.

An independent Hugging Face replay of 126 sampled response tokens has mean absolute
logprob difference **0.01856**, maximum **0.46385**, against SGLang (temperature 0.8).
It passes the explicitly chosen mean tolerance of 0.1; this is not bitwise parity.
Zeroing image pixels changes response logprobs by **0.13136** on average.

A separate clean collection of four episodes per task found:

| Task | Native F1 rewards |
| --- | --- |
| `lost_phone_lockdown` | 0, 0.8, 0, 0 |
| `app_privacy_quarantine` | 0, 0.667, 0, 0 |
| `parent_phone_readability` | 0, 0, 0, 0 |

These are untrained-checkpoint diversity probes, not post-training evaluation.
Native silent/oracle controls on the first task score **0/1**. **17 CPU tests pass**,
including direct/IPC native prompt parity, image-token order, QKV conversion,
trajectory reward accounting, cancellation, isolation, and fail-closed audit errors.

Checkpoint restore also passed: a fresh Slime/Ray process loaded iteration 1,
including optimizer and saved scheduler state, resumed at rollout 2, saved iteration
2, and refreshed inference weights. That resumed batch had equal zero rewards and
gradient norm 0; its mean learner/rollout logprob difference was **0.01069**.

## Exact implementation and runtime

- Official `THUDM/slime`: `4c193f1f37509cca70f0e88807a9305b70f63f4e`, local branch
  `unified-assistant-env`, remote `upstream`. No GitHub-hosted fork is published.
- Qwen checkpoint: `Qwen/Qwen2.5-VL-3B-Instruct`, revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- ScreenSim: `9451adb8aef3504649ec90373cd5d5e9e9416010`.
  Cooking: `f6ea8d2f677ca2f92759e13affd23208cd2d33b9`.
  VH: `5c44fe9354aac53a6d9b66bd49a05bf5dad22cfc`. Engines were not edited.
- Megatron-LM: `3714d81d418c9f1bca4594fc35f9e8289f652862`.
  Megatron Bridge 0.3.0rc0; Torch 2.9.1+cu129; torchvision 0.24.1+cu129;
  Transformers 4.57.1; SGLang 0.5.6.post2; router 0.3.2; Ray 2.58.0;
  Transformer Engine 2.10.0; Playwright 1.58.0.
- Local `.venv` overlays shared heavy dependencies. The shared runtime was not
  upgraded. These tested versions differ from Slime's reference Dockerfile; this
  is a site-specific working profile, not a portable environment lockfile.
- BF16, TP2/SP, PP1, CP1; one sample per microbatch; two episodes per GRPO batch;
  frozen vision tower/projector; Adam LR 5e-7; zero KL/entropy/weight-decay terms;
  response limit 512, context limit 16384, temperature 0.8, top-p 1, top-k disabled.
- Two TP1 SGLang engines colocate with the TP2 learner, with Slime-owned offload/
  reload and weight synchronization. No upstream trainer/loss files were changed.
  The opt-in Qwen2.5-VL plugin registers model-specific HF loading/export mappings
  and uses NVIDIA's model provider. Multi-sample packing is deliberately rejected.

The ScreenSim profile uses native scripted hands and a deterministic dialogue
human, `scripted_human_v1`, baseline persona, two screenshots, and native F1.
It is **not** the previous Gemini-human/vLLM JSON-mode evaluation: generation here
is unconstrained, and inference wall-clock latency is not scored as simulator delay.

## Artifacts and reproduction

Logs and small summaries:
`/gpfs/home/zixianma/interact/slime/interact-runs/screensim-289525/`.

Large artifacts:
`/gpfs/scrubbed/zixianma/checkpoints/web/slime-screensim-289525/`.

- `scrubbed/checkpoints/iter_0000000` and `iter_0000001`: completed distributed
  model/optimizer checkpoints; `scrubbed/rollout-*.pt`, `train-*.pt`, and `episodes/`.
- `diversity/diversity.pt` and `diversity/episodes/`: clean 12-episode collection.
- Small summaries: `hf_replay.json`, `checkpoint_update.json`, `diversity_summary.json`.
- Successful training log: `12-slime-scrubbed.log`, with a zero exit status.
- Restore log: `15-checkpoint-resume-scheduler.log`; resumed artifacts are under
  `resume-checked/`, including `checkpoints/iter_0000002`.
- `final-provenance/manifest.json` records the final local source hashes/package
  versions; `final-provenance/tasks.jsonl` demonstrates shared ScreenSim/cooking
  task preparation. It is a final-source snapshot, not a claim that failed earlier
  attempts ran identical code.

CPU validation:

```bash
source examples/interact/env.sh
python -m pytest -q tests/interact
```

GPU entry point is `examples/interact/screensim_smoke.sbatch`. A **new submission
requires new compute approval**. The controller defaults to a bounded preflight,
real rollout, independent logprob replay, and training sequence, then exits.
Interactive queued debugging is opt-in, with job-scoped queues; it must not replay
another allocation's commands. The current run used the original interactive
controller while bringing up the stack.

To extend a saved constant-LR run, pass the checkpoint directory with `--load`,
increase `--num-rollout`, and use `--use-checkpoint-opt-param-scheduler` to restore
the saved scheduler rather than silently changing its settings. Always use a new
`INTERACT_ATTEMPT` for new output artifacts.

## Failures retained, not counted as successful runs

- Initial serving retries needed the shared `ninja` executable on PATH and a local
  `libcudart.so` linker symlink. Shared CUDA files were left untouched.
- Multimodal task placeholders must be message lists, not strings. The generator
  replaces these with each native decision's actual prompt and images.
- Qwen provider initialization required variable-length configuration alignment
  and TP-compatible vocabulary handling (64-way divisibility, no extra vocabulary).
- Home-path checkpoint/rollout writes failed. Partial checkpoints were **moved,
  preserved** under `failed-artifacts/home-checkpoints`; large new artifacts use
  scrubbed storage. Failed `.pt` files must not be treated as valid checkpoints.
- The failed home-path diversity run is excluded: a native exception-to-silence
  handler could swallow an audit-write error. Shared transport now raises a
  non-swallowable infrastructure failure, with a regression test.
- First restore attempt rejected the changed scheduler horizon; it did not silently
  reinitialize the optimizer or count as a successful restore.

## Online in-time-success test — job 290990

Explicitly approved and completed on September 12, 2026, node **g004**:
**2 H200, 16 CPUs, 240 GiB, 30-minute ceiling**. Slurm state **COMPLETED**, exit
**0:0**, elapsed **7m25s** (**0.2472 GPU-hours**, approximately **$0.22** using the
submission estimate of $0.90 per GPU-hour). Both GPUs were released. The unrelated
OpenWebRL baseline job 290926 remained running on g022 and was not modified.

Fresh Qwen2.5-VL-3B-Instruct checkpoint; native `screensim_intime_success_v1` reward;
three GRPO batches of four episodes on `lost_phone_lockdown`, seed 0, composite 0,
`scripted_human_v1`. Other model/runtime settings match the narrow profile above.
**23 preflight tests passed.**

| Training rollout | In-time successes | Mean native F1 (diagnostic only) | Gradient norm | Mean learner/rollout logprob difference |
| --- | --- | --- | --- | --- |
| 0 | 0/4 | 0.33350 | 0 | 0.01113 |
| 1 | 0/4 | 0.00000 | 0 | 0.01665 |
| 2 | 0/4 | 0.16675 | 0 | 0.01212 |

All 12 native reports have `success=0` and `goal_ok=false`; three episodes have
F1=0.667 despite failing the task. Success-only reward therefore produced no GRPO
advantage or learning signal. This validates rollout, optimizer execution,
checkpoint saving, inference weight synchronization and online tracking—not an
effective policy update or benchmark improvement. No reward shaping, extra batch,
or second allocation was introduced to manufacture a nonzero gradient.

Live W&B run:
[screensim-qwen25vl3b-290990-intime-small](https://wandb.ai/zixianma/interact-slime-rl/runs/ujzg3ikd)
in the separate **`zixianma/interact-slime-rl`** project. W&B reports **finished**.
Read-only cloud-history verification matched episode counts, episode-weighted
in-time success, F1 and gradient norms against native audits/trainer logs for all
three steps. This is not the historical offline backfill.

Artifacts:

- Small logs, successful stage statuses and source-hash provenance:
  `/gpfs/home/zixianma/interact/slime/interact-runs/screensim-290990/`.
- Episodes, rollout/train dumps and W&B local records:
  `/gpfs/scrubbed/zixianma/checkpoints/web/slime-screensim-290990/intime-small/`.
- Final distributed model/optimizer checkpoint: `checkpoints/iter_0000002/`
  under that directory, with four shard files, `.metadata`, `common.pt` and
  `metadata.json`; the latest-iteration marker is 2. A new restore was not run.

Before longer success-only training, identify a task/profile or curriculum that
produces successful trajectories under this model, or explicitly evaluate a
versioned shaping reward while retaining native in-time success as the primary
outcome metric. Twelve samples on one scenario cannot establish a general success
rate, but this run supplies no evidence that merely extending it will learn.

## Next gates

1. Matched-profile held-out evaluation over tasks, composite indices, seeds and
   personas. No success or learning-gain claim from this smoke run alone.
2. Invalid/false-alarm controls, explicit native budget-exhaustion reporting, and
   failure/preemption recovery before longer runs. The native ScreenSim report
   currently lacks an explicit truncation reason.
3. Portable dependency lock/container and numerical parity across supported model
   and parallelism settings. The current provider intentionally has narrow limits.
4. Cooking through the same tested GPU bridge; its native adapter is CPU tested.
   VH still needs its native adapter and exclusive Unity leasing/lifecycle tests.
   Engine-neutral API support does not mean all three GPU integrations are done.

## Qwen3.5-4B pilot bring-up — September 12–13, 2026

User-approved allocation **292972**, g001: **2 H200, 16 CPUs, 240 GiB, at most
2 hours / 4 GPU-hours** (Slurm estimate $3.60). No new budget extension, paid
model APIs or baseline-runtime modification. The 18-train / 12-validation split
is separately frozen; `hearing_setup` moved whole to validation. Six training
tasks are interleaved, one scenario per task in each batch of six groups.

Current candidate: `pilot09`, W&B
https://wandb.ai/zixianma/interact-slime-rl/runs/6neavc52.
Baseline validation completed: **25/48 in-time successes (52.0833%)**, with
**60.0% task-macro success**, at `eval/step=0`. All 12 validation scenarios have
four completed attempts. The first training collection completed with **33/48
successes** and mixed-success groups. A finite optimizer update is now verified
in `pilot09`, but **no policy-improvement claim yet**. `pilot05` failed at its
first learner forward: the upstream vision
helper incorrectly required even sequence lengths at CP=1. The local fix maps
CP=1 tokens directly, retaining the two-chunk validation for CP>1; all nine native
Qwen3.5 tests, including the odd-length regression, pass. `pilot07` reuses only
the pre-update baseline and first batch from `pilot05`, with the original HF
weights and no saved optimizer checkpoint. Later batches are freshly generated;
the recovery hook advances the training data cursor and validates membership.
`pilot06` then failed because forced TE FlashAttention has no installed backend
in the new runtime. `pilot07` selects cuDNN FusedAttention with `--attention-backend
auto`. All 373 learner microbatches finished, but the gradient-norm safety check
found a NaN before the optimizer update. No checkpoint or successful policy update
was produced. Stage `27-anomaly` localized the first nonfinite backward operation
to `FusedAttnFuncBackward` after 18 microbatches. Stage `28-baseline-parity` passed
independent HF replay for three validation tasks (mean absolute logprob differences
0.03361, 0.04249, 0.04212), checking the initial learner-exported inference weights.
Stage `29-install-flash` installed official FlashAttention 2.8.3 for CUDA12,
Torch2.9, CPython3.12, ABI TRUE, checksum
`4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810`.
`pilot09` forces that backend and targets **two** complete RL cycles plus full
pre/post validation, reduced from three to fit the original allocation. That is
12/18 training scenarios seen in this pilot; split membership is unchanged.
The first optimizer update in `pilot09` completed successfully: gradient norm
**1.2947688**, loss **0.00150865**, and mean learner/rollout absolute logprob
difference **0.0184826**. FlashAttention resolved the observed fused-attention
backward NaN on this complete batch. Distributed checkpoint `iter_0000000` and
the latest-iteration marker are saved, and updated inference weights were sent
back to SGLang. The second batch is freshly generated. Full post-training
validation, parameter/frozen-vision comparison, and restore checks remain pending.
The replay
collection's throughput metrics measure loading saved data, not fresh inference.
The current target is two 6×8 GRPO cycles,
native success-only reward, frozen vision, LR 5e-7, and four attempts on each of
12 validation scenarios before/after training. `eval/task_macro_success` uses
completed-update coordinates 0 and 2, and never weights episodes by turn count.

Verified prerequisites:

- 31 environment/bridge tests and 8 upstream native Qwen3.5 tests passed.
- Qwen3.5 SGLang loading, learner construction/HF loading, offload, and the initial
  learner-to-inference weight synchronization completed.
- H200 linear-attention forward/backward, finite gradients, and nested memory
  offload/restore passed with a clean exit using the learner's preload mode.
- Two real scripted-human ScreenSim episodes completed through the multimodal
  bridge. Their native rewards were both 0; this probe was not training.
- Independent HF replay of 128 response tokens from each probe episode: mean
  absolute logprob differences **0.01513 / 0.04340**, below the diagnostic 0.1
  tolerance. Image ablation changed logprobs by **0.06162 / 0.37753** on average.

Startup failures are preserved, not relabeled as reward-zero episodes: CUDA
library lookup names; missing optional Apex gradient-accumulation fusion; the
old memory-saver's lack of nested regions; a standalone torch-mode allocator
teardown failure (preload-mode retest passed); optional DeepGEMM compilation;
and re-expansion of image tokens in the newer multimodal processor. Final runtime
details and launch files are in README.md; source/package snapshots are in the
job directory. The shared runtime still reports Torch 2.9.1+cu129, Transformers
4.57.1, SGLang 0.5.6.post2 and memory-saver 0.0.9, unchanged.

The controller owns and awaits training, then queues saved-checkpoint comparison,
independent baseline replay, and restore-only verification within the same
ceiling. These post-training checks remain pending. Its aggregate exit status
retains historical failed setup stages, so inspect individual stage statuses
when assessing the eventual pilot outcome.
