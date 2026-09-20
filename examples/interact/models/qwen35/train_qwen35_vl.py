"""Job-local Ray lifecycle around the official Slime Qwen3.5-VL trainer."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from slime.utils.arguments import parse_args

args = parse_args()
split = Path('/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-v2-18train-12val/manifest.json')
raw = split.read_bytes()
args.interact_split_version = json.loads(raw)['split_version']
args.interact_split_sha256 = hashlib.sha256(raw).hexdigest()
assert args.interact_split_sha256 == 'a29464102278d784a06a6b5a57ab6cb3dd53b2da6a74015bab54f8687a3d96f9'
args.interact_model_revision = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
if os.environ.get('INTERACT_PARENT_WANDB_RUN'):
    args.interact_parent_wandb_run = os.environ['INTERACT_PARENT_WANDB_RUN']
    args.interact_parent_checkpoint = args.load
if os.environ.get('INTERACT_PARSE_ONLY') == '1':
    print('QWEN35_ARGS_OK', args.spec, args.interact_split_version, flush=True)
else:
    from examples.interact.tracking.wandb_resume import configure_tracking
    configure_tracking(args, os.environ)
    import ray
    from train import train
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        ray.init(address='local', num_cpus=16, num_gpus=2,
                 object_store_memory=4 * 1024**3, include_dashboard=False)
        train(args)
    finally:
        ray.shutdown()
