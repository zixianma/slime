#!/usr/bin/env bash
# Evaluate one ScreenSim checkpoint over all 30 frozen scenarios.
#SBATCH --job-name=screensim-full-eval
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=screensim-full-eval-%A_%a.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the Slime checkout}"
source examples/interact/models/qwen35/qwen35_env.sh
: "${FULL_SUITE_ROOT:?}"
: "${FULL_SUITE_CHECKPOINTS:?set to a versioned checkpoint specification}"
: "${WANDB_ENV_FILE:?}"
: "${PROVIDER_ENV_FILE:?}"
: "${Q35_MODEL:?}"

readarray -t checkpoint < <(python examples/interact/evaluation/checkpoints.py \
  --spec "$FULL_SUITE_CHECKPOINTS" --index "${SLURM_ARRAY_TASK_ID:-0}")
if [[ ${#checkpoint[@]} -ne 4 ]]; then
  echo "FULL_SUITE_LAUNCH_ERROR invalid checkpoint specification" >&2
  exit 1
fi
label=${checkpoint[0]} update=${checkpoint[1]} index=${checkpoint[2]} load=${checkpoint[3]}

export FULL_SUITE_ENGINE=screensim FULL_SUITE_LABEL=$label FULL_SUITE_UPDATE=$update
export FULL_SUITE_CHECKPOINT_INDEX=$index FULL_SUITE_LOAD=$load
export FULL_SUITE_DATASET=$FULL_SUITE_ROOT/data/screensim-all.jsonl
export FULL_SUITE_MANIFEST=$FULL_SUITE_ROOT/data/manifest.json FULL_SUITE_EXPECTED=30
export FULL_SUITE_RUN_DIR=$FULL_SUITE_ROOT/screensim/$label
export FULL_SUITE_GPUS=2 FULL_SUITE_CPUS=16 FULL_SUITE_WANDB_TEAM=${FULL_SUITE_WANDB_TEAM:-zixianma}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

if [[ -e "$FULL_SUITE_RUN_DIR/audit/result.json" ]]; then
  echo "FULL_SUITE_ALREADY_COMPLETE $FULL_SUITE_RUN_DIR/audit/result.json"
  exit 0
fi
mkdir -p "$FULL_SUITE_RUN_DIR"
python examples/interact/tracking/with_wandb_env.py \
  --env-file "$WANDB_ENV_FILE" --provider-env-file "$PROVIDER_ENV_FILE" -- \
  bash examples/interact/evaluation/run.sh
