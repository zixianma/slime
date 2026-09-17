#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/qwen35_env.sh"
export OMP_NUM_THREADS="${COOKING_OMP_THREADS:-4}"
cd "$INTERACT_REPO"
source scripts/models/qwen3.5-4B.sh
MODEL_ARGS[1]=slime_plugins.models.qwen3_5_vl
MODEL_ARGS[2]=get_qwen3_5_vl_model_provider
COOK_MODEL=/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison/models/Qwen3.5-4B
export COOKING_SPLIT_DIR="${COOKING_SPLIT_DIR:-$PWD/interact-runs/cooking-rl-v1-20260914-budget-v2}"
COOK_SPLIT="$COOKING_SPLIT_DIR"
COOK_RUN="${COOKING_RUN_DIR:-/gpfs/scrubbed/zixianma/checkpoints/web/slime-cooking-$SLURM_JOB_ID/calibration01}"
mkdir -p "$COOK_RUN"
export INTERACT_OUTPUT_ROOT="$COOK_RUN/episodes"
export WANDB_MODE=online WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
unset WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID INTERACT_PARENT_WANDB_RUN INTERACT_WANDB_RUN_PATH
export RAY_TMPDIR="$(mktemp -d /tmp/cooking-q35-ray.XXXXXX)"
python examples/interact/train_cooking_calibration.py "${MODEL_ARGS[@]}" \
  --use-wandb --wandb-mode online --wandb-project interact-slime-rl --wandb-team zixianma \
  --wandb-group "cooking-qwen35-$SLURM_JOB_ID-calibration01" \
  --disable-wandb-random-suffix --wandb-dir "$COOK_RUN/wandb" \
  --hf-checkpoint "$COOK_MODEL" --load "$COOK_MODEL" --make-vocab-size-divisible-by 64 \
  --save "$COOK_RUN/checkpoints" --save-interval 1 \
  --num-rollout 1 --num-gpus-per-node 2 --actor-num-nodes 1 --actor-num-gpus-per-node 2 --colocate \
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --moe-token-dispatcher-type alltoall --freeze-params-name-list 'model\.visual\.' \
  --prompt-data "$COOK_SPLIT/train.jsonl" --input-key prompt \
  --custom-generate-function-path interact_env.slime_bridge.generate.generate \
  --custom-reward-post-process-path interact_env.slime_bridge.rewards.normalize \
  --custom-rollout-log-function-path examples.interact.cooking_calibration_metrics.log_rollout \
  --custom-eval-rollout-log-function-path examples.interact.cooking_calibration_metrics.log_eval \
  --custom-config-path examples/interact/configs/cooking_qwen35_rollout.yaml \
  --rollout-batch-size 2 --n-samples-per-prompt 4 --micro-batch-size 1 --global-batch-size 8 \
  --rollout-max-response-len 512 --rollout-max-context-len 16384 --rollout-temperature 0.8 --rollout-top-p 1 --rollout-top-k -1 \
  --eval-interval 1 --eval-prompt-data cooking "$COOK_SPLIT/calibration_validation.jsonl" \
  --n-samples-per-eval-prompt 2 --eval-max-response-len 512 --eval-temperature 0.8 --eval-top-p 1 --eval-top-k -1 \
  --advantage-estimator grpo --use-rollout-logprobs --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0 \
  --optimizer adam --lr 5e-7 --lr-decay-style constant --weight-decay 0 \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash --no-gradient-accumulation-fusion \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.35 \
  --sglang-server-concurrency 4 --sglang-max-running-requests 4 --sglang-chunked-prefill-size 2048 \
  --sglang-disable-cuda-graph --sglang-enable-deterministic-inference \
  --save-debug-rollout-data "$COOK_RUN/rollout-{rollout_id}.pt" \
  --save-debug-train-data "$COOK_RUN/train-{rollout_id}.pt" "$@"
