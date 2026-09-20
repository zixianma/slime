#!/usr/bin/env bash
# PREPARED ONLY: a new submission requires approval for 7 hours / 14 GPU-hours.
#SBATCH --job-name=cooking-q35-calibration
#SBATCH --partition=gpu-h200
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=07:00:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/cooking-q35-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/models/qwen35/qwen35_env.sh
export INTERACT_JOB_DIR="$PWD/interact-runs/cooking-q35-$SLURM_JOB_ID"
mkdir -p "$INTERACT_JOB_DIR"
srun --kill-on-bad-exit=1 python examples/interact/tracking/with_wandb_env.py \
  --env-file /gpfs/projects/krishna/zixianma/OpenWebRL/.env \
  -- python examples/interact/archive/cooking_calibration/cooking_calibration_controller.py
