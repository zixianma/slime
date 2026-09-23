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
if worker_timeout := os.environ.get("Q35_WORKER_TIMEOUT"):
    worker_timeout = int(worker_timeout)
    if worker_timeout < 180:
        raise ValueError("Q35_WORKER_TIMEOUT must be at least 180 seconds")
    args.interact["worker_timeout"] = worker_timeout
split_root = Path(os.environ.get(
    "Q35_SPLIT", "/gpfs/scrubbed/zixianma/checkpoints/web/screensim-rl-v2-18train-12val"
))
split = split_root / "manifest.json"
raw = split.read_bytes()
args.interact_split_version = json.loads(raw)['split_version']
args.interact_split_sha256 = hashlib.sha256(raw).hexdigest()
expected_split = os.environ.get("Q35_EXPECTED_SPLIT_SHA256")
if expected_split and args.interact_split_sha256 != expected_split:
    raise ValueError("Q35 split manifest hash mismatch")
args.interact_model_revision = os.environ.get(
    "Q35_MODEL_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
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
        ray.init(address='local', num_cpus=int(os.environ.get("Q35_CPUS", "16")),
                 num_gpus=int(os.environ.get("Q35_NUM_GPUS", "2")),
                 object_store_memory=4 * 1024**3, include_dashboard=False)
        train(args)
    finally:
        ray.shutdown()
