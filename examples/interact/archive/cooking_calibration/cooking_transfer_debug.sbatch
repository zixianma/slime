#!/usr/bin/env bash
# User-approved dedicated debugging allocation, not an RL retry.
#SBATCH --job-name=cooking-weight-transfer-debug
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/cooking-q35-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/models/qwen35/qwen35_env.sh
export COOKING_HARDWARE_ACCEPTANCE=1 COOKING_TRANSFER_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1
export INTERACT_JOB_DIR="$PWD/interact-runs/cooking-q35-$SLURM_JOB_ID"
mkdir -p "$INTERACT_JOB_DIR"
srun --kill-on-bad-exit=1 timeout --signal=TERM --kill-after=30s 1620s \
  bash examples/interact/archive/cooking_calibration/run_cooking_calibration.sh --rollout-num-gpus 2 \
  --sglang-config examples/interact/configs/cooking_gpu_layout.yaml \
  --custom-config-path examples/interact/configs/cooking_hardware_acceptance.yaml
