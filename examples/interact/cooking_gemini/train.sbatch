#!/usr/bin/env bash
# Four-H200 CookSim RL launcher for fresh and checkpoint-resumed Gemini-user runs.
# Required variables are documented in COOKING_GEMINI_HANDOFF.md.
#SBATCH --job-name=cooking-gemini-rl
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=cooking-gemini-%j.out
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo"
source examples/interact/models/qwen35/qwen35_env.sh

: "${COOKING_GEMINI_SPLIT:?set COOKING_GEMINI_SPLIT to the generated split directory}"
: "${COOKING_GEMINI_RUN_DIR:?set COOKING_GEMINI_RUN_DIR to a new durable run directory}"
: "${Q35_MODEL:?set Q35_MODEL to the local Qwen3.5-4B checkpoint}"
: "${COOKING_WANDB_TEAM:?set COOKING_WANDB_TEAM}"
: "${COOKING_WORKER_PYTHON:?set COOKING_WORKER_PYTHON to the CookBench Python interpreter}"
: "${WANDB_ENV_FILE:?set WANDB_ENV_FILE to an env file containing WANDB_API_KEY}"
: "${PROVIDER_ENV_FILE:?set PROVIDER_ENV_FILE to an env file containing GOOGLE_API_KEY}"
export Q35_MODEL
export COOKING_WORKER_PYTHON
for required in \
  "$COOKING_GEMINI_SPLIT/manifest.json" \
  "$COOKING_GEMINI_SPLIT/train.jsonl" \
  "$COOKING_GEMINI_SPLIT/validation.jsonl" \
  "$Q35_MODEL" "$COOKING_WORKER_PYTHON" "$WANDB_ENV_FILE" "$PROVIDER_ENV_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "COOKING_LAUNCH_ERROR missing required path: $required" >&2
    exit 1
  fi
done
if [[ -e "$COOKING_GEMINI_RUN_DIR" && ! -d "$COOKING_GEMINI_RUN_DIR" ]]; then
  echo "COOKING_LAUNCH_ERROR run path exists and is not a directory" >&2
  exit 1
fi
if [[ -d "$COOKING_GEMINI_RUN_DIR" && -n "$(ls -A "$COOKING_GEMINI_RUN_DIR")" ]]; then
  echo "COOKING_LAUNCH_ERROR run directory must be new or empty" >&2
  exit 1
fi

export COOKING_RUN_DIR=$COOKING_GEMINI_RUN_DIR
export COOKING_WANDB_GROUP=${COOKING_WANDB_GROUP:-cooking-gemini37-${SLURM_JOB_ID}}
export GEMINI_MODEL=gemini-3.7-flash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false COOKING_BIND_RENDER_WORKERS=1
unset WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID
mkdir -p "$COOKING_GEMINI_RUN_DIR/audit" interact-runs

resume_args=()
if [[ -n "${COOKING_GEMINI_LOAD:-}" ]]; then
  : "${INTERACT_WANDB_RUN_PATH:?resumption requires entity/project/run-id in INTERACT_WANDB_RUN_PATH}"
  if [[ ! -f "$COOKING_GEMINI_LOAD/latest_checkpointed_iteration.txt" ]]; then
    echo "COOKING_LAUNCH_ERROR resume checkpoint marker is missing" >&2
    exit 1
  fi
  resume_args=(--load "$COOKING_GEMINI_LOAD" --use-checkpoint-opt-param-scheduler)
else
  unset INTERACT_WANDB_RUN_PATH
fi

timeout --signal=TERM --kill-after=30s 420s python examples/interact/cooking_gemini/cooking_render_preflight.py
export COOKING_RENDER_GPU
COOKING_RENDER_GPU=$(python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["COOKING_RUN_DIR"])/("preflight-"+os.environ["SLURM_JOB_ID"])/"selected.json"; print(json.loads(p.read_text())["renderer_gpu"])')
export INTERACT_RENDER_GPU_UUID
INTERACT_RENDER_GPU_UUID=$(python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["COOKING_RUN_DIR"])/("preflight-"+os.environ["SLURM_JOB_ID"])/"selected.json"; print(json.loads(p.read_text())["renderer_uuid"])')

layout=examples/interact/configs/cooking_gpu_layout_4gpu_render0.yaml
if [[ "$COOKING_RENDER_GPU" == 1 ]]; then
  layout=examples/interact/configs/cooking_gpu_layout_4gpu_render1.yaml
fi
export COOKING_SHARED_RENDERER=0 COOKING_SERVER_CONCURRENCY=4
if timeout --signal=TERM --kill-after=30s 720s \
    python examples/interact/cooking_gemini/render_policy_preflight.py; then
  layout=examples/interact/configs/cooking_gpu_layout_4gpu_shared.yaml
  export COOKING_SHARED_RENDERER=1 COOKING_SERVER_CONCURRENCY=6
  echo "COOKING_TOPOLOGY shared-renderer-policy layout=$layout"
else
  echo "COOKING_TOPOLOGY dedicated-renderer-fallback layout=$layout"
fi

python examples/interact/tracking/with_wandb_env.py \
  --env-file "$WANDB_ENV_FILE" --provider-env-file "$PROVIDER_ENV_FILE" -- \
  bash examples/interact/cooking_gemini/run.sh \
  --rollout-num-gpus 4 --sglang-config "$layout" \
  --num-rollout "${COOKING_NUM_ROLLOUT:-9}" --save-interval 1 \
  "${resume_args[@]}"
