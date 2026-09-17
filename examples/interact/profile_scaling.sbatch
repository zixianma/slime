#!/usr/bin/env bash
# Approved pair: 2 GPUs/16 CPUs/1h and 4 GPUs/32 CPUs/30min. Set resources at submission.
#SBATCH --job-name=engine-scaling-profile
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=240G
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/scaling-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/qwen35_env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PROFILE_GPUS="$1"
PROFILE_SECONDS="$2"
PROFILE_OUTPUT="/gpfs/scrubbed/zixianma/checkpoints/web/engine-scaling-$SLURM_JOB_ID"
exec srun --kill-on-bad-exit=1 python examples/interact/profile_scaling.py \
  --gpus "$PROFILE_GPUS" --seconds "$PROFILE_SECONDS" --output "$PROFILE_OUTPUT" \
  --manifest examples/interact/configs/scaling_profile_v1.json
