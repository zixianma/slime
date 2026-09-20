#!/usr/bin/env bash
#SBATCH --job-name=cooking-render1-policy1
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=00:28:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/render-policy-%j.out
set -euo pipefail
cd /gpfs/home/zixianma/interact/slime
source examples/interact/models/qwen35/qwen35_env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PROFILE_OUTPUT="/gpfs/scrubbed/zixianma/checkpoints/web/render-policy-$SLURM_JOB_ID"
PROFILE_PORT=$((30000 + SLURM_JOB_ID % 20000))
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=1 --gpus-per-task=1 --mem=160G \
  python examples/interact/tools/profiling/profile_dedicated_renderer.py --role policy \
  --output "$PROFILE_OUTPUT" --port "$PROFILE_PORT" &
POLICY_STEP_PID=$!
cleanup() {
  kill -TERM "$POLICY_STEP_PID" 2>/dev/null || true
  wait "$POLICY_STEP_PID" 2>/dev/null || true
}
trap cleanup EXIT
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=1 --gpus-per-task=1 --mem=80G \
  python examples/interact/tools/profiling/profile_dedicated_renderer.py --role renderer \
  --output "$PROFILE_OUTPUT" --port "$PROFILE_PORT" --seconds 1680
