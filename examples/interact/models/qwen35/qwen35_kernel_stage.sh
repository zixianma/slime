#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/qwen35_env.sh"
export LD_PRELOAD="$Q35_RUNTIME/venv/lib/python3.12/site-packages/torch_memory_saver_hook_mode_preload_cu12.abi3.so"
export TMS_INIT_ENABLE=1 TMS_INIT_ENABLE_CPU_BACKUP=1
exec python examples/interact/models/qwen35/qwen35_gpu_preflight.py
