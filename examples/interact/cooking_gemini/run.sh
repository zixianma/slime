#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../qwen35_env.sh"
cd "$INTERACT_REPO"
source scripts/models/qwen3.5-4B.sh
MODEL_ARGS[1]=slime_plugins.models.qwen3_5_vl
MODEL_ARGS[2]=get_qwen3_5_vl_model_provider
model=${Q35_MODEL:?Q35_MODEL must point to the local Qwen3.5-4B checkpoint}
split=${COOKING_GEMINI_SPLIT:?COOKING_GEMINI_SPLIT is required}
run=${COOKING_GEMINI_RUN_DIR:?COOKING_GEMINI_RUN_DIR is required}
server_concurrency=${COOKING_SERVER_CONCURRENCY:-6}
wandb_group=${COOKING_WANDB_GROUP:-cooking-gemini37-single-error-${SLURM_JOB_ID}}
wandb_project=${COOKING_WANDB_PROJECT:-interact-slime-rl}
wandb_team=${COOKING_WANDB_TEAM:?COOKING_WANDB_TEAM is required}
mkdir -p "$run/checkpoints" "$run/audit" "$run/wandb"
export INTERACT_OUTPUT_ROOT="$run/episodes" INTERACT_JOB_DIR="$run/audit"
export WANDB_MODE=online WANDB_CONSOLE=off WANDB_DISABLE_CODE=true
unset WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID INTERACT_PARENT_WANDB_RUN
export RAY_TMPDIR="$(mktemp -d /tmp/cooking-gemini-ray.XXXXXX)"
python examples/interact/cooking_gemini/train.py "${MODEL_ARGS[@]}" \
  --use-wandb --wandb-mode online --wandb-project "$wandb_project" --wandb-team "$wandb_team" \
  --wandb-group "$wandb_group" \
  --disable-wandb-random-suffix --wandb-dir "$run/wandb" \
  --hf-checkpoint "$model" --load "$model" --make-vocab-size-divisible-by 64 \
  --save "$run/checkpoints" --save-interval 1 \
  --num-rollout 3 --num-gpus-per-node 4 --actor-num-nodes 1 --actor-num-gpus-per-node 4 --colocate \
  --tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --moe-token-dispatcher-type alltoall --freeze-params-name-list 'model\.visual\.' \
  --prompt-data "$split/train.jsonl" --input-key prompt \
  --custom-generate-function-path interact_env.slime_bridge.generate.generate \
  --rollout-function-path examples.interact.cooking_gemini.speculative_rollout.generate_rollout \
  --eval-function-path slime.rollout.sglang_rollout.generate_rollout \
  --custom-reward-post-process-path interact_env.slime_bridge.rewards.normalize \
  --custom-rollout-log-function-path examples.interact.cooking_gemini.metrics.log_rollout \
  --custom-eval-rollout-log-function-path examples.interact.cooking_gemini.metrics.log_eval \
  --custom-config-path examples/interact/configs/cooking_qwen35_rollout.yaml \
  --rollout-batch-size 6 --n-samples-per-prompt 4 --micro-batch-size 1 --global-batch-size 24 \
  --rollout-max-response-len 512 --rollout-max-context-len 16384 --rollout-temperature 0.8 --rollout-top-p 1 --rollout-top-k -1 \
  --eval-interval 3 --eval-prompt-data cooking "$split/validation.jsonl" \
  --n-samples-per-eval-prompt 2 --eval-max-response-len 512 --eval-temperature 0.8 --eval-top-p 1 --eval-top-k -1 \
  --advantage-estimator grpo --use-rollout-logprobs --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0 \
  --optimizer adam --lr 5e-7 --lr-decay-style constant --weight-decay 0 \
  --attention-dropout 0 --hidden-dropout 0 --attention-backend flash --no-gradient-accumulation-fusion \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.35 \
  --sglang-server-concurrency "$server_concurrency" --sglang-max-running-requests "$server_concurrency" --sglang-chunked-prefill-size 2048 \
  --sglang-disable-cuda-graph --sglang-enable-deterministic-inference "$@"
