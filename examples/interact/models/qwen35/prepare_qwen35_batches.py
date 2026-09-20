"""Interleave the frozen training tasks: six different tasks in each RL batch."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from examples.interact.screensim.prepare_rl_split import prompt_rows

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
path = Path('/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-v2-18train-12val/manifest.json')
raw = path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'a29464102278d784a06a6b5a57ab6cb3dd53b2da6a74015bab54f8687a3d96f9'
rows = prompt_rows(json.loads(raw), 'train')
tasks = defaultdict(list)
for row in rows:
    tasks[row['metadata']['episode']['task_id']].append(row)
assert len(tasks) == 6 and {len(v) for v in tasks.values()} == {3}
ordered = [row for batch in zip(*tasks.values(), strict=True) for row in batch]
assert {r['metadata']['scenario_id'] for r in ordered} == {r['metadata']['scenario_id'] for r in rows}
with args.output.open('x') as out:
    for row in ordered:
        out.write(json.dumps(row)+'\n')
print(json.dumps({'scenarios': len(ordered), 'tasks_per_batch': 6,
                  'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))
