#!/usr/bin/env bash
# Evaluate one CookSim checkpoint over all 150 frozen cases.
#SBATCH --job-name=cooking-full-eval
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=cooking-full-eval-%A_%a.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the Slime checkout}"
source examples/interact/models/qwen35/qwen35_env.sh
: "${FULL_SUITE_ROOT:?}"
: "${FULL_SUITE_CHECKPOINTS:?set to a versioned checkpoint specification}"
: "${COOKING_SPLIT:?}"
: "${COOKING_WORKER_PYTHON:?}"
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

export FULL_SUITE_ENGINE=cooking FULL_SUITE_LABEL=$label FULL_SUITE_UPDATE=$update
export FULL_SUITE_CHECKPOINT_INDEX=$index FULL_SUITE_LOAD=$load
export FULL_SUITE_DATASET=$FULL_SUITE_ROOT/data/cooking-all.jsonl
export FULL_SUITE_MANIFEST=$FULL_SUITE_ROOT/data/manifest.json FULL_SUITE_EXPECTED=150
export FULL_SUITE_RUN_DIR=$FULL_SUITE_ROOT/cooking/$label
export FULL_SUITE_GPUS=4 FULL_SUITE_CPUS=32 FULL_SUITE_WANDB_TEAM=${FULL_SUITE_WANDB_TEAM:-zixianma}
export COOKING_RUN_DIR=$FULL_SUITE_RUN_DIR COOKING_BIND_RENDER_WORKERS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

if [[ -e "$FULL_SUITE_RUN_DIR/audit/result.json" ]]; then
  echo "FULL_SUITE_ALREADY_COMPLETE $FULL_SUITE_RUN_DIR/audit/result.json"
  exit 0
fi
mkdir -p "$FULL_SUITE_RUN_DIR"
timeout --signal=TERM --kill-after=30s 420s python examples/interact/cooking_gemini/cooking_render_preflight.py
export COOKING_RENDER_GPU
COOKING_RENDER_GPU=$(python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["COOKING_RUN_DIR"])/("preflight-"+os.environ["SLURM_JOB_ID"])/"selected.json"; print(json.loads(p.read_text())["renderer_gpu"])')
export INTERACT_RENDER_GPU_UUID
INTERACT_RENDER_GPU_UUID=$(python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["COOKING_RUN_DIR"])/("preflight-"+os.environ["SLURM_JOB_ID"])/"selected.json"; print(json.loads(p.read_text())["renderer_uuid"])')
export COOKING_SHARED_RENDERER=1 COOKING_SERVER_CONCURRENCY=6
export FULL_SUITE_SGLANG_CONFIG=examples/interact/configs/cooking_gpu_layout_4gpu_shared.yaml

python examples/interact/tracking/with_wandb_env.py \
  --env-file "$WANDB_ENV_FILE" --provider-env-file "$PROVIDER_ENV_FILE" -- \
  bash examples/interact/evaluation/run.sh
