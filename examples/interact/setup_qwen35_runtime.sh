#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/gpu_env.sh"
Q35_RUNTIME=/gpfs/scrubbed/zixianma/openwebrl-runtime/screensim-qwen35-rl
python -m venv --without-pip "$Q35_RUNTIME/venv"
# Install into this new overlay only; heavy baseline packages are read-only fallbacks.
python -m pip --python "$Q35_RUNTIME/venv/bin/python" install --no-compile --no-deps \
  sglang==0.5.9 sgl-kernel==0.3.21 flash-linear-attention==0.4.2 fla-core==0.4.2 \
  flashinfer-python==0.6.3 flashinfer-cubin==0.6.3 quack-kernels==0.2.4
python -m pip --python "$Q35_RUNTIME/venv/bin/python" install --no-compile --no-deps \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810'
if [[ ! -d "$Q35_RUNTIME/Megatron-LM/.git" ]]; then
  git clone --no-checkout --filter=blob:none https://github.com/NVIDIA/Megatron-LM.git "$Q35_RUNTIME/Megatron-LM"
  git -C "$Q35_RUNTIME/Megatron-LM" fetch --depth 1 origin 1dcf0dafa884ad52ffb243625717a3471643e087
  git -C "$Q35_RUNTIME/Megatron-LM" checkout --detach FETCH_HEAD
fi
test "$(git -C "$Q35_RUNTIME/Megatron-LM" rev-parse HEAD)" = 1dcf0dafa884ad52ffb243625717a3471643e087
if ! git -C "$Q35_RUNTIME/Megatron-LM" apply --reverse --check "$INTERACT_REPO/docker/patch/latest/megatron.patch"; then
  git -C "$Q35_RUNTIME/Megatron-LM" apply --check "$INTERACT_REPO/docker/patch/latest/megatron.patch"
  git -C "$Q35_RUNTIME/Megatron-LM" apply "$INTERACT_REPO/docker/patch/latest/megatron.patch"
fi
mkdir -p "$Q35_RUNTIME/lib"
for Q35_LIBRARY in cuda_nvrtc/lib/libnvrtc.so.12 cudnn/lib/libcudnn.so.9 curand/lib/libcurand.so.10; do
  Q35_TARGET="$INTERACT_RUNTIME/venv/lib/python3.12/site-packages/nvidia/$Q35_LIBRARY"
  Q35_LINK="$Q35_RUNTIME/lib/$(basename "${Q35_LIBRARY%.so.*}").so"
  if [[ ! -e "$Q35_LINK" ]]; then ln -s "$Q35_TARGET" "$Q35_LINK"; fi
done
source examples/interact/qwen35_env.sh
if [[ ! -d "$Q35_RUNTIME/torch_memory_saver/.git" ]]; then
  git clone --no-checkout --filter=blob:none https://github.com/zhuzilin/torch_memory_saver.git "$Q35_RUNTIME/torch_memory_saver"
  git -C "$Q35_RUNTIME/torch_memory_saver" fetch --depth 1 origin 8d30c59ca12a68d9deccbc9c6599076a1218cbc5
  git -C "$Q35_RUNTIME/torch_memory_saver" checkout --detach FETCH_HEAD
fi
test "$(git -C "$Q35_RUNTIME/torch_memory_saver" rev-parse HEAD)" = 8d30c59ca12a68d9deccbc9c6599076a1218cbc5
bash examples/interact/build_qwen35_tms.sh
"$Q35_RUNTIME/venv/bin/python" -c 'import torch, transformers, sglang; from fla.modules import ShortConvolution, FusedRMSNormGated; from slime_plugins.models.qwen3_5_vl import get_qwen3_5_vl_model_provider; print("QWEN35_IMPORTS_OK", torch.__version__, transformers.__version__, sglang.__version__, torch.cuda.get_device_name(0), flush=True)'
