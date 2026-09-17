"""Initial productive allocation stage: GPU/runtime checks and checkpoint staging."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

import torch
from huggingface_hub import snapshot_download

job = Path(os.environ["INTERACT_JOB_DIR"])
assert torch.cuda.device_count() == 2, "expected exactly the approved two visible GPUs"
print(subprocess.check_output(["nvidia-smi"], text=True), flush=True)
manifest = {"gpus": [torch.cuda.get_device_name(i) for i in range(2)],
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "sglang", "ray", "transformer-engine")}}
path = snapshot_download("Qwen/Qwen2.5-VL-3B-Instruct",
                         revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
                         local_dir="/gpfs/scrubbed/zixianma/checkpoints/web/Qwen2.5-VL-3B-Instruct")
manifest["checkpoint"] = path
manifest["checkpoint_revision"] = "66285546d2b821cf421d4f5eb2576359d3770cd3"
(job / "preflight.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2), flush=True)
try:
    import megatron.bridge
    print("Megatron Bridge import passed", flush=True)
except Exception as exc:
    print(f"Megatron Bridge import failed: {type(exc).__name__}: {exc}", flush=True)
subprocess.run(["python", "-m", "pytest", "-q", "-o", "addopts=", "tests/interact"], check=True)
