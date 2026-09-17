#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export PATH="$INTERACT_REPO/../.venv/bin:$INTERACT_RUNTIME/venv/bin:$INTERACT_RUNTIME/cuda/bin:$PATH"
export CUDA_HOME="$INTERACT_RUNTIME/cuda"
# Supply the linker name locally; leave the shared CUDA installation untouched.
mkdir -p "$INTERACT_REPO/interact-runs/cuda-link"
if [[ ! -e "$INTERACT_REPO/interact-runs/cuda-link/libcudart.so" ]]; then
    ln -s "$CUDA_HOME/lib64/libcudart.so.12" "$INTERACT_REPO/interact-runs/cuda-link/libcudart.so"
fi
export LIBRARY_PATH="$INTERACT_REPO/interact-runs/cuda-link${LIBRARY_PATH:+:$LIBRARY_PATH}"
export PYTHONPATH="$INTERACT_REPO:$INTERACT_RUNTIME/src/Megatron-LM"
export CPATH="$INTERACT_RUNTIME/src/python-headers/Include:$INTERACT_RUNTIME/src/python-headers"
export LD_LIBRARY_PATH="$INTERACT_RUNTIME/venv/lib/python3.12/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for interact_include in "$INTERACT_RUNTIME"/venv/lib/python3.12/site-packages/nvidia/*/include; do
    export CPATH="$CPATH:$interact_include"
done
for interact_library in "$INTERACT_RUNTIME"/venv/lib/python3.12/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="$interact_library:$LD_LIBRARY_PATH"
done
export FLASHINFER_WORKSPACE_BASE="$INTERACT_REPO/interact-runs/cache/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="$INTERACT_REPO/interact-runs/cache/torchinductor"
export TRITON_CACHE_DIR="$INTERACT_REPO/interact-runs/cache/triton"
export CUDA_CACHE_PATH="$INTERACT_REPO/interact-runs/cache/cuda"
export HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=4 MAX_JOBS=8 CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_ADDRESS=local
