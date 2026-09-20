#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../runtime/gpu_env.sh"
cd "$INTERACT_REPO"
source scripts/models/qwen2.5-3B.sh
CKPT=/gpfs/scrubbed/zixianma/checkpoints/web/Qwen2.5-VL-3B-Instruct
INTERACT_TRAIN_DIR="/gpfs/scrubbed/zixianma/checkpoints/web/slime-screensim-$SLURM_JOB_ID/${INTERACT_ATTEMPT:-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$INTERACT_TRAIN_DIR"
export INTERACT_OUTPUT_ROOT="$INTERACT_TRAIN_DIR/episodes"
export WANDB_MODE="${INTERACT_WANDB_MODE:-offline}"
export WANDB_PROJECT="${INTERACT_WANDB_PROJECT:-interact-slime-rl}"
export WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
unset WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID
INTERACT_WANDB_ARGS=(--use-wandb --wandb-mode "$WANDB_MODE" --wandb-project "$WANDB_PROJECT"
  --wandb-group "screensim-qwen25vl3b-$SLURM_JOB_ID-$(basename "$INTERACT_TRAIN_DIR")"
  --disable-wandb-random-suffix --wandb-dir "$INTERACT_TRAIN_DIR/wandb")
if [[ -n "${INTERACT_WANDB_ENTITY:-}" ]]; then
  INTERACT_WANDB_ARGS+=(--wandb-team "$INTERACT_WANDB_ENTITY")
fi
echo "INTERACT_TRAIN_DIR=$INTERACT_TRAIN_DIR"
export RAY_TMPDIR="$(mktemp -d /tmp/screensim-ray.XXXXXX)"
python examples/interact/models/qwen25/train_qwen25_vl.py "${MODEL_ARGS[@]}" \
  "${INTERACT_WANDB_ARGS[@]}" \
  --hf-checkpoint "$CKPT" --load "$CKPT" \
  --make-vocab-size-divisible-by 64 \
  --save "$INTERACT_TRAIN_DIR/checkpoints" --save-interval 1 \
  --num-rollout 2 --num-gpus-per-node 2 --actor-num-nodes 1 --actor-num-gpus-per-node 2 --colocate \
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --moe-token-dispatcher-type alltoall \
  --custom-model-provider-path slime_plugins.models.qwen25_vl.model_provider \
  --custom-megatron-init-path slime_plugins.models.qwen25_vl.register \
  --prompt-data examples/interact/configs/screensim_tasks.jsonl --input-key prompt \
  --custom-generate-function-path interact_env.slime_bridge.generate.generate \
  --custom-reward-post-process-path interact_env.slime_bridge.rewards.normalize \
  --custom-rollout-log-function-path interact_env.slime_bridge.metrics.log_rollout \
  --custom-config-path examples/interact/configs/rollout.yaml \
  --rollout-batch-size 1 --n-samples-per-prompt 2 --micro-batch-size 1 --global-batch-size 2 \
  --rollout-max-response-len 512 --rollout-max-context-len 16384 --rollout-temperature 0.8 --rollout-top-p 1 \
  --advantage-estimator grpo --use-rollout-logprobs --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0 \
  --optimizer adam --lr 5e-7 --lr-decay-style constant --weight-decay 0 \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.35 \
  --sglang-server-concurrency 2 --sglang-max-running-requests 2 --sglang-chunked-prefill-size 2048 \
  --sglang-disable-cuda-graph \
  --save-debug-rollout-data "$INTERACT_TRAIN_DIR/rollout-{rollout_id}.pt" \
  --save-debug-train-data "$INTERACT_TRAIN_DIR/train-{rollout_id}.pt" "$@"
