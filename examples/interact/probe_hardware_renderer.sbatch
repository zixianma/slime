#!/usr/bin/env bash
# Capability gate before the requested 1-renderer + 1-policy GPU benchmark.
#SBATCH --job-name=cooking-hardware-render-probe
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/render-probe-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/gpu_env.sh
exec srun --kill-on-bad-exit=1 timeout --signal=TERM --kill-after=15s 480s \
  ../.venv/bin/python examples/interact/probe_hardware_renderer.py \
  --output "/gpfs/scrubbed/zixianma/checkpoints/web/render-probe-$SLURM_JOB_ID"
