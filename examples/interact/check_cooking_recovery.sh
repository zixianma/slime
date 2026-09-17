#!/usr/bin/env bash
# Read-only status; the timestamp separates retries sharing one run directory.
set -euo pipefail
job_id=${1:?job ID required}
since=${2:?retry start timestamp required}
run_root=${3:-/gpfs/scrubbed/zixianma/checkpoints/web/cooking-full-295856}
squeue -h -j "$job_id" -o '%i %T %M %L'
find "$run_root/episodes" -maxdepth 2 -name report.json -newermt "$since" -print0 |
  xargs -0 -r jq -cs '{episodes:length,wins:map(select(.outcome=="won"))|length,
    mixed_groups:(group_by(.scenario)|map(select((map(.outcome=="won")|unique|length)>1))|length)}'
find "$run_root/episodes" -maxdepth 2 -name rollout_timing.jsonl -newermt "$since" -print0 |
  xargs -0 -r jq -cs '[.[]|select(.phase=="turn")]|{turns:length,max_http:(map(.http_roundtrip_s)|max)}'
jq -c '{phase,status,completed_updates,error}' "$run_root/audit/progress.json"
