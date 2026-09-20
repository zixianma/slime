"""Read-only resume gates and job-local provenance; never change parent artifacts."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import torch
import flash_attn
from torch.distributed.checkpoint import FileSystemReader
from examples.interact.models.qwen35.qwen35_continue_profile import (
    PARENT, ORDER, NAME, VERIFICATION, PARENT_WANDB, START, TARGET, EVAL_STEPS, SAMPLER)

job = Path(os.environ['INTERACT_JOB_DIR'])
assert json.loads(VERIFICATION.read_text())['status'] == 'passed'
assert os.environ['INTERACT_ATTEMPT'] == NAME
assert os.environ['INTERACT_PARENT_WANDB_RUN'] == PARENT_WANDB
old_job = ORDER.parent
if NAME == 'continue09':
    assert json.loads((old_job/'restore_verified.json').read_text()) == {'checkpoint_iteration': 1, 'additional_updates': 0}
assert (PARENT/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip() == str(START-1)
assert hashlib.sha256(ORDER.read_bytes()).hexdigest() == '31a41ba735df38727fbcc8488c843be734a78ceb6ad411b066cfea47146d5fde'
FileSystemReader(PARENT/f'checkpoints/iter_{START-1:07d}').read_metadata()
state = torch.load(PARENT/f'checkpoints/rollout/global_dataset_state_dict_{START-1}.pt', map_location='cpu', weights_only=False)
assert {k: state[k] for k in SAMPLER} == SAMPLER
assert flash_attn.__version__ == '2.8.3'
assert torch.cuda.device_count() == 2
assert shutil.disk_usage(PARENT).free > 500 * 1024**3, 'need 500 GiB checkpoint/artifact headroom'
manifest = dict(parent_checkpoint=str(PARENT/f'checkpoints/iter_{START-1:07d}'), parent_wandb_run=PARENT_WANDB,
    start_completed_updates=START, target_completed_updates=TARGET, additional_updates=TARGET-START,
    eval_completed_updates=EVAL_STEPS, prompt_order_sha256=hashlib.sha256(ORDER.read_bytes()).hexdigest(),
    sampler={k:state[k] for k in ('sample_offset','epoch_id','sample_group_index','sample_index')},
    gpu_names=[torch.cuda.get_device_name(i) for i in range(2)])
(job/'resume_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
subprocess.run(['python','examples/interact/models/qwen35/qwen35_provenance.py','--output',str(job/'provenance.json')], check=True)
print(json.dumps(manifest), flush=True)
