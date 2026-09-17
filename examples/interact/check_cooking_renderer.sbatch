#!/usr/bin/env bash
#SBATCH --job-name=cooking-render-correctness
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/render-check-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/gpu_env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec srun --kill-on-bad-exit=1 timeout --signal=TERM --kill-after=15s 540s \
  ../.venv/bin/python examples/interact/check_cooking_renderer.py \
  "/gpfs/scrubbed/zixianma/checkpoints/web/render-check-$SLURM_JOB_ID"
