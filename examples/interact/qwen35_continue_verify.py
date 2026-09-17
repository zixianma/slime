"""Require the approved completed updates, final checkpoint, and full evaluations."""
import json
import os
from pathlib import Path
from torch.distributed.checkpoint import FileSystemReader
from examples.interact.verify_qwen35_run import check_checkpoints
from examples.interact.qwen35_continue_profile import NAME, START, TARGET, EVAL_STEPS

job = Path(os.environ['INTERACT_JOB_DIR'])
root = Path('/gpfs/scrubbed/zixianma/checkpoints/web')/f"slime-qwen35-{os.environ['SLURM_JOB_ID']}"/NAME
assert (root/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip() == str(TARGET-1)
for i in range(START,TARGET):
    FileSystemReader(root/f'checkpoints/iter_{i:07d}').read_metadata()
metrics = [json.loads(s) for s in (job/'episode_metrics.jsonl').read_text().splitlines()]
assert [m['eval/step'] for m in metrics if 'eval/step' in m] == EVAL_STEPS
assert [m['train/step'] for m in metrics if 'train/step' in m] == list(range(START,TARGET))
result = dict(status='passed', completed_updates=TARGET, additional_updates=TARGET-START,
              checkpoints=check_checkpoints(root/'checkpoints'),
              evaluation=[m for m in metrics if 'eval/step' in m])
(job/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
