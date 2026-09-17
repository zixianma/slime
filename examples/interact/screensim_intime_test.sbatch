#!/usr/bin/env bash
# Submit only after approval of these exact resources and the 1 GPU-hour ceiling.
#SBATCH --job-name=screensim-intime-wandb
#SBATCH --partition=gpu-h200
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/screensim-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/gpu_env.sh
export INTERACT_JOB_DIR="$PWD/interact-runs/screensim-$SLURM_JOB_ID"
export INTERACT_ATTEMPT=intime-small
export INTERACT_WANDB_MODE=online INTERACT_WANDB_ENTITY=zixianma INTERACT_WANDB_PROJECT=interact-slime-rl
export INTERACT_SMALL_INTIME_TEST=1 INTERACT_CONTROLLER_SECONDS=1700
mkdir -p "$INTERACT_JOB_DIR"
# Credentials load inside the owned allocation, never into sbatch arguments.
srun --kill-on-bad-exit=1 python examples/interact/with_wandb_env.py \
  --env-file /gpfs/projects/krishna/zixianma/OpenWebRL/.env \
  -- python examples/interact/gpu_controller.py
