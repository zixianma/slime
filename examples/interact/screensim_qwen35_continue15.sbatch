#!/usr/bin/env bash
# Approved: six additional updates; 2 H200, 16 CPUs, 240 GiB, 3 hours / 6 GPU-hours.
#SBATCH --job-name=screensim-q35-to15
#SBATCH --partition=gpu-h200
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/screensim-qwen35-continue-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/qwen35_env.sh
export INTERACT_JOB_DIR="$PWD/interact-runs/qwen35-continue-$SLURM_JOB_ID"
export Q35_CONTINUATION_PROFILE=continue15
export INTERACT_WANDB_MODE=online INTERACT_ATTEMPT=continue15
export INTERACT_PARENT_WANDB_RUN=6kv7biu2
mkdir -p "$INTERACT_JOB_DIR"
srun --kill-on-bad-exit=1 python examples/interact/with_wandb_env.py \
  --env-file /gpfs/projects/krishna/zixianma/OpenWebRL/.env \
  -- python examples/interact/qwen35_continue_controller.py
