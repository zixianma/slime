#!/usr/bin/env bash
# Exact approved ceiling: 3 H200 / 24 CPU / 240 GiB / 2 hours = 6 GPU-hours.
#SBATCH --job-name=screensim-qwen-compare
#SBATCH --partition=gpu-h200
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/screensim-compare-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/runtime/env.sh
export HF_HUB_OFFLINE=1 WANDB_MODE=online WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
export HF_HOME=/gpfs/scrubbed/zixianma/openwebrl-runtime/screensim-qwen-compare/hf-cache
export XDG_CACHE_HOME=/gpfs/scrubbed/zixianma/openwebrl-runtime/screensim-qwen-compare/cache
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton-$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1
srun --kill-on-bad-exit=1 python examples/interact/tracking/with_wandb_env.py \
  --env-file /gpfs/projects/krishna/zixianma/OpenWebRL/.env \
  -- python examples/interact/tools/comparison/comparison_controller.py
