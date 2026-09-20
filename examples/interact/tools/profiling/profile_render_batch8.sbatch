#!/usr/bin/env bash
# One-hour two-GPU profile. Optional first argument selects request limit 4 or 8.
#SBATCH --job-name=cooking-render1-policy1-batch8
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --account=zixianma
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/home/zixianma/interact/slime/interact-runs/render-policy-%j.out
set -euo pipefail
PROFILE_REQUEST_LIMIT="${1:-8}"
case "$PROFILE_REQUEST_LIMIT" in
  4|8) ;;
  *) echo "Request limit must be 4 or 8" >&2; exit 2 ;;
esac
cd /gpfs/home/zixianma/interact/slime
source examples/interact/models/qwen35/qwen35_env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PROFILE_OUTPUT="/gpfs/scrubbed/zixianma/checkpoints/web/render-policy-$SLURM_JOB_ID"
PROFILE_PORT=$((30000 + SLURM_JOB_ID % 20000))
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=1 --gpus-per-task=1 --mem=160G \
  python examples/interact/tools/profiling/profile_dedicated_renderer.py --role policy \
  --output "$PROFILE_OUTPUT" --port "$PROFILE_PORT" --max-running-requests "$PROFILE_REQUEST_LIMIT" &
POLICY_STEP_PID=$!
cleanup() {
  kill -TERM "$POLICY_STEP_PID" 2>/dev/null || true
  wait "$POLICY_STEP_PID" 2>/dev/null || true
}
trap cleanup EXIT
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=1 --gpus-per-task=1 --mem=80G \
  python examples/interact/tools/profiling/profile_dedicated_renderer.py --role renderer \
  --output "$PROFILE_OUTPUT" --port "$PROFILE_PORT" --seconds 3600 --max-running-requests "$PROFILE_REQUEST_LIMIT"
