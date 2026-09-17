#!/usr/bin/env bash
# Local CPU/dependency overlay only. GPU compatibility with this upstream pin is unverified.
INTERACT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTERACT_RUNTIME="${INTERACT_RUNTIME:-/gpfs/scrubbed/zixianma/openwebrl-runtime}"
export PATH="$INTERACT_REPO/../.venv/bin:$PATH"
export PYTHONPATH="$INTERACT_REPO"
export PLAYWRIGHT_BROWSERS_PATH="$INTERACT_RUNTIME/browsers"
export XDG_CACHE_HOME="$INTERACT_REPO/interact-runs/cache"
export HF_HOME="$INTERACT_REPO/interact-runs/cache/hf"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4
