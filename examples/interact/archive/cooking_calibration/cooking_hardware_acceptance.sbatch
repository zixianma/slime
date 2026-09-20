#!/usr/bin/env bash
# Approved acceptance: 2 H200, 16 CPUs, 240 GiB, at most 2 hours / 4 GPU-hours.
#SBATCH --job-name=cooking-hardware-rl-acceptance
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/cooking-q35-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/models/qwen35/qwen35_env.sh
export COOKING_HARDWARE_ACCEPTANCE=1 INTERACT_PROFILE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export INTERACT_JOB_DIR="$PWD/interact-runs/cooking-q35-$SLURM_JOB_ID"
mkdir -p "$INTERACT_JOB_DIR"
srun --kill-on-bad-exit=1 python examples/interact/tracking/with_wandb_env.py \
  --env-file /gpfs/projects/krishna/zixianma/OpenWebRL/.env \
  -- python examples/interact/archive/cooking_calibration/cooking_acceptance_controller.py
