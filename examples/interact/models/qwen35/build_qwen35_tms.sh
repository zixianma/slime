#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/qwen35_env.sh"
export TMS_CUDA_MAJOR=12
python -m pip install --no-compile --no-deps --no-build-isolation \
  "$Q35_RUNTIME/torch_memory_saver"
python -c 'import importlib.metadata; print("TMS_VERSION", importlib.metadata.version("torch_memory_saver"))'
# The separate qwen35_kernel_stage.sh tests the real preload-mode GPU lifecycle.
