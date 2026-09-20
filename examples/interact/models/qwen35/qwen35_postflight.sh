#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/qwen35_env.sh"
python -c 'import json,os,pathlib; p=pathlib.Path(os.environ["INTERACT_JOB_DIR"])/(os.environ.get("Q35_PILOT_STAGE","14-pilot")+".status.json"); assert json.loads(p.read_text())["returncode"] == 0, "pilot did not finish successfully"'
Q35_RESULT="/gpfs/scrubbed/zixianma/checkpoints/web/slime-qwen35-$SLURM_JOB_ID/${Q35_PILOT_ATTEMPT:-pilot05}"
python examples/interact/models/qwen35/verify_qwen35_run.py --run-dir "$Q35_RESULT"
# Restore-only: require the expected last checkpoint and add no updates.
Q35_LAST_ITERATION="${Q35_EXPECTED_LAST_ITERATION:-2}"
test "$(<"$Q35_RESULT/checkpoints/latest_checkpointed_iteration.txt")" = "$Q35_LAST_ITERATION"
INTERACT_ATTEMPT="restore-${Q35_PILOT_ATTEMPT:-pilot05}" bash examples/interact/models/qwen35/run_qwen35_checked.sh \
  --load "$Q35_RESULT/checkpoints" --use-checkpoint-opt-param-scheduler \
  --num-rollout "$((Q35_LAST_ITERATION + 1))" --prompt-data "$INTERACT_JOB_DIR/train-round-robin.jsonl"
python -c 'import json,os,pathlib; (pathlib.Path(os.environ["INTERACT_JOB_DIR"])/"restore_verified.json").write_text(json.dumps({"checkpoint_iteration":int(os.environ.get("Q35_EXPECTED_LAST_ITERATION", "2")),"additional_updates":0})+"\n")'
