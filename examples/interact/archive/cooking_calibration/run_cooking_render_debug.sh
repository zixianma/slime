#!/usr/bin/env bash
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/models/qwen35/qwen35_env.sh
export COOKING_TRANSFER_DEBUG=1 COOKING_HARDWARE_ACCEPTANCE=1 COOKING_RENDER_SMOKE=vulkan
export COOKING_BIND_RENDER_WORKERS=1 COOKING_RENDER_GPU=1
export CUDA_LAUNCH_BLOCKING=1
export INTERACT_JOB_DIR="/gpfs/scrubbed/zixianma/checkpoints/web/render-debug-$SLURM_JOB_ID-live-${1:-1}"
mkdir -p "$INTERACT_JOB_DIR"
bash examples/interact/archive/cooking_calibration/run_cooking_calibration.sh --rollout-num-gpus 2 \
  --sglang-config examples/interact/configs/cooking_gpu_layout_render1.yaml \
  --custom-config-path examples/interact/configs/cooking_hardware_acceptance.yaml
