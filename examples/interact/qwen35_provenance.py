"""Record source/dependency provenance without reading credentials or model weights."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/screensim-qwen35-rl')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
options = parser.parse_args()
files = list((ROOT/'examples/interact').glob('*.py')) + list((ROOT/'examples/interact').glob('*.sh'))
files += list((ROOT/'interact_env').rglob('*.py'))
files += [ROOT/'slime_plugins/models'/name for name in ('qwen3_5.py', 'qwen3_5_vl.py', 'qwen3_5_vl_utils.py', 'hf_attention.py')]
result = dict(
    packages={name: metadata.version(name) for name in (
        'torch', 'transformers', 'sglang', 'sgl-kernel', 'flashinfer-python',
        'flash-linear-attention', 'fla-core', 'torch-memory-saver', 'transformer-engine', 'ray', 'wandb')},
    git={name: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
         for name, path in [('slime', ROOT), ('screensim', ROOT.parent/'screensim-engine'),
                            ('megatron', RUNTIME/'Megatron-LM'), ('memory_saver', RUNTIME/'torch_memory_saver')]},
    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    megatron_patch_sha256=hashlib.sha256((ROOT/'docker/patch/latest/megatron.patch').read_bytes()).hexdigest(),
    split_sha256='a29464102278d784a06a6b5a57ab6cb3dd53b2da6a74015bab54f8687a3d96f9',
    note='Site-specific isolated runtime. Successful imports do not prove numerical parity or optimizer updates.',
)
try:
    result['packages']['flash-attn'] = metadata.version('flash-attn')
except metadata.PackageNotFoundError:
    pass
with options.output.open('x') as out:
    json.dump(result, out, indent=2)
    out.write('\n')
print(json.dumps(result['packages']))
