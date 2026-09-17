#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/gpu_env.sh"
export Q35_RUNTIME=/gpfs/scrubbed/zixianma/openwebrl-runtime/screensim-qwen35-rl
export PATH="$Q35_RUNTIME/venv/bin:$PATH"
export PYTHONPATH="$INTERACT_REPO:$Q35_RUNTIME/Megatron-LM"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$Q35_RUNTIME/cache/triton"
export FLASHINFER_WORKSPACE_BASE="$Q35_RUNTIME/cache/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="$Q35_RUNTIME/cache/torchinductor"
export CUDA_CACHE_PATH="$Q35_RUNTIME/cache/cuda"
export MAX_JOBS=8
export CUDNN_HOME="$INTERACT_RUNTIME/venv/lib/python3.12/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="$Q35_RUNTIME/lib:$LD_LIBRARY_PATH"
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
