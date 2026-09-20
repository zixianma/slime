#!/usr/bin/env bash
# Shape tested in approved job 289525. A new submission requires new compute approval.
#SBATCH --job-name=screensim-vl3b-smoke
#SBATCH --partition=gpu-h200
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/screensim-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/runtime/gpu_env.sh
export INTERACT_JOB_DIR="$PWD/interact-runs/screensim-$SLURM_JOB_ID"
export INTERACT_DATA_DIR="/gpfs/scrubbed/zixianma/checkpoints/web/slime-screensim-$SLURM_JOB_ID/probe"
mkdir -p "$INTERACT_JOB_DIR"
# The controller owns and awaits every worker; no detached external srun steps.
srun --kill-on-bad-exit=1 python examples/interact/screensim/gpu_controller.py
