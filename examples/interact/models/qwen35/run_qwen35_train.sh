#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/qwen35_env.sh"
cd "$INTERACT_REPO"
source scripts/models/qwen3.5-4B.sh
MODEL_ARGS[1]=slime_plugins.models.qwen3_5_vl
MODEL_ARGS[2]=get_qwen3_5_vl_model_provider
Q35_MODEL="${Q35_MODEL:-/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison/models/Qwen3.5-4B}"
Q35_SPLIT="${Q35_SPLIT:-/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-v2-18train-12val}"
Q35_TRAIN_DIR="${Q35_TRAIN_DIR:-/gpfs/scrubbed/zixianma/checkpoints/web/slime-qwen35-$SLURM_JOB_ID/${INTERACT_ATTEMPT:-pilot}}"
Q35_NUM_GPUS="${Q35_NUM_GPUS:-2}"
Q35_WANDB_TEAM="${Q35_WANDB_TEAM:-zixianma}"
Q35_WANDB_PROJECT="${Q35_WANDB_PROJECT:-interact-slime-rl}"
Q35_NUM_ROLLOUT="${Q35_NUM_ROLLOUT:-3}"
export Q35_MODEL Q35_SPLIT Q35_TRAIN_DIR Q35_NUM_GPUS Q35_NUM_ROLLOUT
mkdir -p "$Q35_TRAIN_DIR"
export INTERACT_OUTPUT_ROOT="$Q35_TRAIN_DIR/episodes"
export WANDB_MODE="${INTERACT_WANDB_MODE:-offline}" WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
unset WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID
export RAY_TMPDIR="$(mktemp -d /tmp/screensim-q35-ray.XXXXXX)"
echo "Q35_TRAIN_DIR=$Q35_TRAIN_DIR"
debug_args=()
if [[ "${Q35_SAVE_DEBUG_DATA:-0}" == 1 ]]; then
  debug_args=(
    --save-debug-rollout-data "$Q35_TRAIN_DIR/rollout-{rollout_id}.pt"
    --save-debug-train-data "$Q35_TRAIN_DIR/train-{rollout_id}.pt"
  )
fi
python examples/interact/models/qwen35/train_qwen35_vl.py "${MODEL_ARGS[@]}" \
  --use-wandb --wandb-mode "$WANDB_MODE" --wandb-project "$Q35_WANDB_PROJECT" --wandb-team "$Q35_WANDB_TEAM" \
  --wandb-group "screensim-qwen35-$SLURM_JOB_ID-${INTERACT_ATTEMPT:-pilot}" \
  --disable-wandb-random-suffix --wandb-dir "$Q35_TRAIN_DIR/wandb" \
  --hf-checkpoint "$Q35_MODEL" --load "$Q35_MODEL" --make-vocab-size-divisible-by 64 \
  --save "$Q35_TRAIN_DIR/checkpoints" --save-interval 1 \
  --num-rollout "$Q35_NUM_ROLLOUT" --num-gpus-per-node "$Q35_NUM_GPUS" --actor-num-nodes 1 --actor-num-gpus-per-node "$Q35_NUM_GPUS" --colocate \
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --moe-token-dispatcher-type alltoall --freeze-params-name-list 'model\.visual\.' \
  --prompt-data "$Q35_SPLIT/train.jsonl" --input-key prompt \
  --custom-generate-function-path interact_env.slime_bridge.generate.generate \
  --custom-reward-post-process-path interact_env.slime_bridge.rewards.normalize \
  --custom-rollout-log-function-path examples.interact.models.qwen35.qwen35_metrics.log_rollout \
  --custom-eval-rollout-log-function-path examples.interact.models.qwen35.qwen35_metrics.log_eval \
  --custom-config-path examples/interact/configs/qwen35_rollout.yaml \
  --rollout-batch-size 6 --n-samples-per-prompt 8 --micro-batch-size 1 --global-batch-size 48 \
  --rollout-max-response-len 512 --rollout-max-context-len 16384 --rollout-temperature 0.8 --rollout-top-p 1 --rollout-top-k -1 \
  --eval-interval 3 --eval-prompt-data screensim "$Q35_SPLIT/validation.jsonl" \
  --n-samples-per-eval-prompt 4 --eval-max-response-len 512 --eval-temperature 0.8 --eval-top-p 1 --eval-top-k -1 \
  --advantage-estimator grpo --use-rollout-logprobs --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0 \
  --optimizer adam --lr 5e-7 --lr-decay-style constant --weight-decay 0 \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash \
  --no-gradient-accumulation-fusion \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.35 \
  --sglang-server-concurrency 4 --sglang-max-running-requests 4 --sglang-chunked-prefill-size 2048 \
  --sglang-disable-cuda-graph --sglang-enable-deterministic-inference \
  "${debug_args[@]}" "$@"
