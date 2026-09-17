"""Audit the saved calibration artifacts after a real independent restore."""
import ast
import json
import math
import os
from pathlib import Path
import re
import torch
from torch.distributed.checkpoint import FileSystemReader

job = Path(os.environ['INTERACT_JOB_DIR'])
acceptance = os.environ.get('COOKING_HARDWARE_ACCEPTANCE') == '1'
root = Path('/gpfs/scrubbed/zixianma/checkpoints/web')/f"slime-cooking-{os.environ['SLURM_JOB_ID']}/calibration01"
assert json.loads((job/'training_complete.json').read_text())['completed_updates'] == 1
assert json.loads((job/'restore_verified.json').read_text())['completed_updates'] == 1
assert (root/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip() == '0'
metadata = FileSystemReader(root/'checkpoints/iter_0000000').read_metadata()
keys = list(metadata.state_dict_metadata)
assert any('optimizer' in k for k in keys), 'optimizer missing from checkpoint'
common = torch.load(root/'checkpoints/iter_0000000/common.pt',map_location='cpu',weights_only=False)
assert 'opt_param_scheduler' in common, 'scheduler missing from checkpoint'
state = torch.load(root/'checkpoints/rollout/global_dataset_state_dict_0.pt',map_location='cpu',weights_only=False)
assert state['sample_index'] == 8 and state['sample_group_index'] == 2
metrics = [json.loads(line) for line in (job/'episode_metrics.jsonl').read_text().splitlines()]
assert [r['eval/step'] for r in metrics if 'eval/step' in r] == ([0] if acceptance else [0,1])
assert [r['train/step'] for r in metrics if 'train/step' in r] == [0]
for rank in range(2):
    audit = json.loads((job/f'weights-rank{rank}.json').read_text())
    assert audit['changed_language_parameters'] > 0 and audit['frozen_visual_parameters'] > 0
records = re.findall(r"step 0: (\{[^\n]+\})", (job/'02-train.log').read_text())
train = [ast.literal_eval(r) for r in records if "'train/grad_norm'" in r]
assert len(train) == 1 and math.isfinite(train[0]['train/grad_norm']) and train[0]['train/grad_norm'] > 0
# Each eval attempt must retain scenario, seed and identical first prompt on both policies.
evaluations = {}
for path in (root/'episodes').glob('*/training_audit.json'):
    entries = json.loads(path.read_text())
    if not entries[0]['metadata']['evaluation']:
        continue
    versions = {v for e in entries for v in e['weight_versions']}
    assert len(versions) == 1
    version = next(iter(versions))
    first = entries[0]
    key = (first['metadata']['group_key'],first['rollout_id'])
    target = evaluations.setdefault(version,{})
    assert key not in target
    target[key] = first['metadata']['prompt_hash']
assert len(evaluations) == (1 if acceptance else 2) and all(len(v)==8 for v in evaluations.values())
if acceptance:
    assert json.loads((job/'handoff.json').read_text())['graphics_processes'] == []
    assert len(json.loads((job/'restored-weights.json').read_text())) == 2
    assert json.loads((job/'weight_refresh.json').read_text())['synchronized']
    for path in (root/'episodes').glob('*/training_audit.json'):
        placement = json.loads((path.parent/'renderer_placement.json').read_text())
        assert placement and all(p['uuid'] == json.loads((job/'gpu-layout.json').read_text())['renderer_uuid'] for p in placement)
else:
    assert list(evaluations.values())[0] == list(evaluations.values())[1]
result = dict(status='passed', completed_updates=1, optimizer=train[0],
              evaluation=[m for m in metrics if 'eval/step' in m],
              eval_weight_versions=list(evaluations), sampler={k:state[k] for k in ['sample_index','sample_group_index']})
(job/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
