#!/usr/bin/env bash
# User approved: 2 H200, 16 CPUs, 240 GiB, 2 hours / 4 GPU-hours.
#SBATCH --job-name=screensim-qwen35-rl
#SBATCH --partition=gpu-h200
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/screensim-qwen35-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/runtime/gpu_env.sh
export INTERACT_JOB_DIR="$PWD/interact-runs/qwen35-rl-$SLURM_JOB_ID"
export INTERACT_WANDB_MODE=online INTERACT_WANDB_ENTITY=zixianma INTERACT_WANDB_PROJECT=interact-slime-rl
mkdir -p "$INTERACT_JOB_DIR"
srun --kill-on-bad-exit=1 python examples/interact/tracking/with_wandb_env.py \
  --env-file /gpfs/projects/krishna/zixianma/OpenWebRL/.env \
  -- python examples/interact/models/qwen35/qwen35_controller.py
