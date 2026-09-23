#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../models/qwen35/qwen35_env.sh"
cd "$INTERACT_REPO"
source scripts/models/qwen3.5-4B.sh
MODEL_ARGS[1]=slime_plugins.models.qwen3_5_vl
MODEL_ARGS[2]=get_qwen3_5_vl_model_provider

: "${FULL_SUITE_ENGINE:?}"
: "${FULL_SUITE_DATASET:?}"
: "${FULL_SUITE_MANIFEST:?}"
: "${FULL_SUITE_RUN_DIR:?}"
: "${FULL_SUITE_LOAD:?}"
: "${FULL_SUITE_CHECKPOINT_INDEX:?}"
: "${FULL_SUITE_UPDATE:?}"
: "${FULL_SUITE_LABEL:?}"
: "${FULL_SUITE_EXPECTED:?}"
: "${FULL_SUITE_GPUS:?}"
: "${FULL_SUITE_CPUS:?}"
: "${FULL_SUITE_WANDB_TEAM:?}"
: "${Q35_MODEL:?}"

mkdir -p "$FULL_SUITE_RUN_DIR/audit" "$FULL_SUITE_RUN_DIR/wandb"
export INTERACT_OUTPUT_ROOT="$FULL_SUITE_RUN_DIR/episodes"
export INTERACT_JOB_DIR="$FULL_SUITE_RUN_DIR/audit"
export WANDB_MODE=online WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
unset WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID INTERACT_PARENT_WANDB_RUN
export RAY_TMPDIR="$(mktemp -d /tmp/full-suite-ray.XXXXXX)"

config=examples/interact/configs/qwen35_rollout.yaml
concurrency=4
data_parallel_size=$((FULL_SUITE_GPUS / 2))
extra_args=()
if [[ "$FULL_SUITE_ENGINE" == cooking ]]; then
  config=examples/interact/configs/cooking_qwen35_rollout.yaml
  concurrency=${COOKING_SERVER_CONCURRENCY:-6}
  extra_args=(--sglang-config "${FULL_SUITE_SGLANG_CONFIG:?cooking requires a GPU layout}")
elif [[ "$FULL_SUITE_ENGINE" != screensim ]]; then
  echo "FULL_SUITE_LAUNCH_ERROR unknown engine: $FULL_SUITE_ENGINE" >&2
  exit 1
fi

load_args=(--load "$FULL_SUITE_LOAD")
if (( FULL_SUITE_CHECKPOINT_INDEX >= 0 )); then
  load_args+=(--ckpt-step "$FULL_SUITE_CHECKPOINT_INDEX")
fi

python examples/interact/evaluation/eval.py "${MODEL_ARGS[@]}" \
  --use-wandb --wandb-mode online \
  --wandb-project "${FULL_SUITE_WANDB_PROJECT:-interact-full-suite-eval}" \
  --wandb-team "$FULL_SUITE_WANDB_TEAM" \
  --wandb-group "full-suite-${FULL_SUITE_ENGINE}-${FULL_SUITE_LABEL}-${SLURM_JOB_ID:-local}" \
  --disable-wandb-random-suffix --wandb-dir "$FULL_SUITE_RUN_DIR/wandb" \
  --hf-checkpoint "$Q35_MODEL" "${load_args[@]}" --no-load-optim --no-load-rng \
  --make-vocab-size-divisible-by 64 \
  --num-rollout 0 --num-gpus-per-node "$FULL_SUITE_GPUS" \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$FULL_SUITE_GPUS" --colocate \
  --tensor-model-parallel-size 2 --sequence-parallel \
  --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --moe-token-dispatcher-type alltoall --freeze-params-name-list 'model\.visual\.' \
  --prompt-data "$FULL_SUITE_DATASET" --input-key prompt \
  --custom-generate-function-path examples.interact.evaluation.generate.generate \
  --custom-eval-rollout-log-function-path examples.interact.evaluation.metrics.log_eval \
  --custom-config-path "$config" \
  --rollout-batch-size 1 --n-samples-per-prompt 1 --micro-batch-size 1 \
  --global-batch-size "$data_parallel_size" \
  --rollout-max-response-len 512 --rollout-max-context-len 16384 \
  --rollout-temperature 0.8 --rollout-top-p 1 --rollout-top-k -1 \
  --eval-interval 1 --eval-prompt-data full_suite "$FULL_SUITE_DATASET" \
  --n-samples-per-eval-prompt 1 --eval-max-response-len 512 \
  --eval-temperature 0.8 --eval-top-p 1 --eval-top-k -1 \
  --advantage-estimator grpo --use-rollout-logprobs --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0 \
  --optimizer adam --lr 5e-7 --lr-decay-style constant --weight-decay 0 \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --no-gradient-accumulation-fusion \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.35 \
  --sglang-server-concurrency "$concurrency" --sglang-max-running-requests "$concurrency" \
  --sglang-chunked-prefill-size 2048 --sglang-disable-cuda-graph \
  --sglang-enable-deterministic-inference "${extra_args[@]}" "$@"
