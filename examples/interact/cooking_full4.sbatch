#!/usr/bin/env bash
# Four-GPU continuation: one Vulkan renderer, three TP=1 policy servers,
# and a TP=1 x DP=4 learner that reuses all four GPUs after rollout offload.
#SBATCH --job-name=cooking-q35-to12-4g
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=05:00:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/cooking-full-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/qwen35_env.sh
export MALLOC_ARENA_MAX=2
export COOKING_OMP_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export COOKING_FULL_TRAINING=1 COOKING_HARDWARE_ACCEPTANCE=1 INTERACT_PROFILE=1
export COOKING_DEFER_BASELINE=1
export COOKING_EPISODE_CACHE=1
unset COOKING_TRANSFER_DEBUG COOKING_RESTORE_ONLY
: "${COOKING_RUN_DIR:?Select the existing cooking run directory explicitly}"
export COOKING_RUN_DIR
export INTERACT_JOB_DIR="$COOKING_RUN_DIR/audit"
mkdir -p "$INTERACT_JOB_DIR"
COOKING_ALLOCATION_END=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | sed -n 's/^EndTime=//p')
COOKING_ALLOCATION_END_EPOCH=$(date -d "$COOKING_ALLOCATION_END" +%s)
export COOKING_SOFT_DEADLINE=$((COOKING_ALLOCATION_END_EPOCH-1800))
export COOKING_HARD_DEADLINE=$((COOKING_ALLOCATION_END_EPOCH-300))
export COOKING_BIND_RENDER_WORKERS=1
export COOKING_EXPECTED_ACTOR_GPUS=4 COOKING_EXPECTED_TP=1 COOKING_EXPECTED_DP=4
# Run the controller in the allocation shell. Wrapping it in an outer srun
# restricts it to an alternating CPU mask on 4-GPU Tillicum nodes; the nested
# renderer srun steps then cannot satisfy their own CPU binding.
timeout --signal=TERM --kill-after=30s 420s python examples/interact/cooking_render_preflight.py
timeout --signal=TERM --kill-after=15s 120s python examples/interact/check_visual_microbatch.py
if [[ -n "${COOKING_RECOVERY_MANIFEST:-}" ]]; then
  timeout --signal=TERM --kill-after=30s 600s python examples/interact/check_cooking_episode_recovery.py
fi
export COOKING_RENDER_GPU
COOKING_RENDER_GPU=$(python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["COOKING_RUN_DIR"])/("preflight-"+os.environ["SLURM_JOB_ID"])/"selected.json"; print(json.loads(p.read_text())["renderer_gpu"])')
COOKING_LAYOUT=examples/interact/configs/cooking_gpu_layout_4gpu_render0.yaml
if [[ "$COOKING_RENDER_GPU" == 1 ]]; then
  COOKING_LAYOUT=examples/interact/configs/cooking_gpu_layout_4gpu_render1.yaml
fi
COOKING_REMAINING=$(python -c 'import os,time; print(max(1,int(float(os.environ["COOKING_HARD_DEADLINE"])-time.time())))')
timeout --signal=TERM --kill-after=60s "$COOKING_REMAINING" \
  python examples/interact/with_wandb_env.py \
  --env-file /gpfs/projects/krishna/zixianma/OpenWebRL/.env -- \
  bash examples/interact/run_cooking_calibration.sh \
  --num-rollout 12 --num-gpus-per-node 4 --actor-num-gpus-per-node 4 \
  --tensor-model-parallel-size 1 \
  --rollout-batch-size 6 --n-samples-per-prompt 8 --global-batch-size 48 \
  --rollout-num-gpus 4 --sglang-config "$COOKING_LAYOUT" \
  --custom-config-path examples/interact/configs/cooking_hardware_acceptance.yaml \
  --custom-rollout-log-function-path examples.interact.cooking_full_metrics.log_rollout \
  --custom-eval-rollout-log-function-path examples.interact.cooking_full_metrics.log_eval \
  --eval-prompt-data cooking interact-runs/cooking-rl-v1-20260914-budget-v2/validation.jsonl \
  --n-samples-per-eval-prompt 4 --eval-interval 3 \
  --wandb-group cooking-qwen35-to12
