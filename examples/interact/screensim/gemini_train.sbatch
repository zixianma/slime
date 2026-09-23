#!/usr/bin/env bash
# Four-H200 ScreenSim RL launcher for fresh and checkpoint-resumed Gemini-user runs.
# Required variables are documented in SCREENSIM_GEMINI_REPRO.md.
#SBATCH --job-name=screensim-gemini-rl
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=screensim-gemini-%j.out
set -euo pipefail

# Slurm stages batch scripts under /var/spool, so BASH_SOURCE no longer points
# into the checkout once the job starts. Prefer the submission directory and
# retain the source-relative fallback for direct/local execution.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/examples/interact/models/qwen35/qwen35_env.sh" ]]; then
  repo=$SLURM_SUBMIT_DIR
else
  repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
fi
cd "$repo"
source examples/interact/models/qwen35/qwen35_env.sh

: "${SCREENSIM_GEMINI_SPLIT:?set SCREENSIM_GEMINI_SPLIT to the generated split directory}"
: "${SCREENSIM_GEMINI_RUN_DIR:?set SCREENSIM_GEMINI_RUN_DIR to a new durable run directory}"
: "${SCREENSIM_ENGINE_ROOT:?set SCREENSIM_ENGINE_ROOT to the pinned engine checkout}"
: "${Q35_MODEL:?set Q35_MODEL to the local Qwen3.5-4B checkpoint}"
: "${SCREENSIM_WANDB_TEAM:?set SCREENSIM_WANDB_TEAM}"
: "${WANDB_ENV_FILE:?set WANDB_ENV_FILE to an env file containing WANDB_API_KEY}"
: "${PROVIDER_ENV_FILE:?set PROVIDER_ENV_FILE to an env file containing GOOGLE_API_KEY}"

for required in \
  "$SCREENSIM_GEMINI_SPLIT/manifest.json" \
  "$SCREENSIM_GEMINI_SPLIT/train.jsonl" \
  "$SCREENSIM_GEMINI_SPLIT/validation.jsonl" \
  "$SCREENSIM_ENGINE_ROOT/.git" "$Q35_MODEL" "$WANDB_ENV_FILE" "$PROVIDER_ENV_FILE"; do
  if [[ ! -e "$required" ]]; then
    echo "SCREENSIM_LAUNCH_ERROR missing required path: $required" >&2
    exit 1
  fi
done
if [[ -e "$SCREENSIM_GEMINI_RUN_DIR" && ! -d "$SCREENSIM_GEMINI_RUN_DIR" ]]; then
  echo "SCREENSIM_LAUNCH_ERROR run path exists and is not a directory" >&2
  exit 1
fi
if [[ -d "$SCREENSIM_GEMINI_RUN_DIR" && -n "$(ls -A "$SCREENSIM_GEMINI_RUN_DIR")" ]]; then
  echo "SCREENSIM_LAUNCH_ERROR run directory must be new or empty" >&2
  exit 1
fi

readarray -t split_metadata < <(python - "$SCREENSIM_GEMINI_SPLIT/manifest.json" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
m = json.loads(p.read_text())
assert m["counts"]["train_scenarios"] == 18
assert m["counts"]["validation_scenarios"] == 12
assert m["human_mode"] == "free" and m["grader"] == "person"
assert m["profile"] == "plan_human_v3:gemini-3.7-flash"
persona = m.get("persona", "baseline")
for scenario in m["scenarios"]:
    c = scenario["spec"]["config"]
    assert c["human"] == "gemini" and c["human_model"] == "gemini-3.7-flash"
    assert c["persona"] == persona
print(m["engine_revision"])
print(hashlib.sha256(p.read_bytes()).hexdigest())
print(persona)
print(m.get("engine_worktree_diff_sha256") or "")
PY
)
engine_revision=$(git -C "$SCREENSIM_ENGINE_ROOT" rev-parse HEAD)
if [[ "${split_metadata[0]}" != "$engine_revision" ]]; then
  echo "SCREENSIM_LAUNCH_ERROR engine revision differs from frozen split" >&2
  exit 1
fi
engine_diff_sha256=$(git -C "$SCREENSIM_ENGINE_ROOT" diff --binary HEAD | sha256sum | cut -d' ' -f1)
if [[ -n "${split_metadata[3]}" && "${split_metadata[3]}" != "$engine_diff_sha256" ]]; then
  echo "SCREENSIM_LAUNCH_ERROR engine worktree patch differs from frozen split" >&2
  exit 1
fi
export SCREENSIM_PERSONA=${split_metadata[2]}

export Q35_SPLIT=$SCREENSIM_GEMINI_SPLIT
export Q35_TRAIN_DIR=$SCREENSIM_GEMINI_RUN_DIR
export Q35_EXPECTED_SPLIT_SHA256=${split_metadata[1]}
export Q35_NUM_GPUS=4 Q35_CPUS=32 Q35_NUM_ROLLOUT=${SCREENSIM_NUM_ROLLOUT:-12}
export Q35_WANDB_TEAM=$SCREENSIM_WANDB_TEAM
export Q35_WANDB_PROJECT=${SCREENSIM_WANDB_PROJECT:-interact-slime-rl}
export INTERACT_WANDB_MODE=online INTERACT_ATTEMPT=${INTERACT_ATTEMPT:-gemini37-full}
export Q35_WORKER_TIMEOUT=${SCREENSIM_WORKER_TIMEOUT:-600}
export SCREENSIM_EPISODE_ATTEMPTS=${SCREENSIM_EPISODE_ATTEMPTS:-2}
export INTERACT_JOB_DIR=$SCREENSIM_GEMINI_RUN_DIR/audit
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID
mkdir -p "$SCREENSIM_GEMINI_RUN_DIR" "$INTERACT_JOB_DIR" interact-runs

if [[ "${SCREENSIM_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "SCREENSIM_PREFLIGHT_OK engine_revision=$engine_revision split_sha256=${split_metadata[1]}"
  exit 0
fi

resume_args=()
if [[ -n "${SCREENSIM_GEMINI_LOAD:-}" ]]; then
  if [[ ! -f "$SCREENSIM_GEMINI_LOAD/latest_checkpointed_iteration.txt" ]]; then
    echo "SCREENSIM_LAUNCH_ERROR resume checkpoint marker is missing" >&2
    exit 1
  fi
  checkpoint_index=$(<"$SCREENSIM_GEMINI_LOAD/latest_checkpointed_iteration.txt")
  if [[ ! "$checkpoint_index" =~ ^[0-9]+$ ]]; then
    echo "SCREENSIM_LAUNCH_ERROR invalid checkpoint marker" >&2
    exit 1
  fi
  if (( checkpoint_index + 1 >= Q35_NUM_ROLLOUT )); then
    echo "SCREENSIM_TARGET_COMPLETE completed_updates=$((checkpoint_index + 1)) target=$Q35_NUM_ROLLOUT"
    exit 0
  fi
  resume_args=(--load "$SCREENSIM_GEMINI_LOAD" --use-checkpoint-opt-param-scheduler --skip-eval-before-train)
else
  unset INTERACT_WANDB_RUN_PATH
fi

smoke_args=()
if [[ "${SCREENSIM_GEMINI_SMOKE:-0}" == 1 ]]; then
  export Q35_NUM_ROLLOUT=1
  smoke_args=(--rollout-batch-size 1 --n-samples-per-prompt 2 --global-batch-size 2
              --eval-interval 100 --skip-eval-before-train)
fi

python examples/interact/tracking/with_wandb_env.py \
  --env-file "$WANDB_ENV_FILE" --provider-env-file "$PROVIDER_ENV_FILE" -- \
  bash examples/interact/models/qwen35/run_qwen35_train.sh \
  "${resume_args[@]}" "${smoke_args[@]}" \
  --custom-generate-function-path examples.interact.screensim.retry_generate.generate
