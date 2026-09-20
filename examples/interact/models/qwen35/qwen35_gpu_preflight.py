"""Small GPU checks for new Qwen3.5 kernels and nested memory offload."""
import json
import gc
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig
from slime_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet

# Use the same preload mode as Slime's learner actors.
with torch_memory_saver.region(tag='outer'):
    with torch_memory_saver.region(tag='inner', enable_cpu_backup=True):
        value = torch.ones(1024, device='cuda:0')
torch_memory_saver.pause(tag='inner')
torch_memory_saver.resume(tag='inner')
assert torch.equal(value, torch.ones_like(value))

config = AutoConfig.from_pretrained('/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison/models/Qwen3.5-4B', local_files_only=True).text_config
layer = Qwen3_5GatedDeltaNet(config, layer_idx=0, args=SimpleNamespace(qwen_gdn_backend='fla')).to(device='cuda:0', dtype=torch.bfloat16)
x = torch.randn(1, 16, config.hidden_size, device='cuda:0', dtype=torch.bfloat16, requires_grad=True)
y = layer(x, cu_seqlens=torch.tensor([0, 16], device='cuda:0', dtype=torch.int32))
y.float().square().mean().backward()
assert torch.isfinite(y).all() and x.grad is not None and torch.isfinite(x.grad).all()
assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in layer.parameters())
assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)
result = dict(memory_offload_roundtrip=True, gdn_forward_backward=True,
              gpu=torch.cuda.get_device_name(0), output_shape=list(y.shape))
(Path(os.environ['INTERACT_JOB_DIR']) / 'gpu-preflight.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
del value, x, y, layer
gc.collect()
torch.cuda.empty_cache()
