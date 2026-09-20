"""Load the opt-in model plugin, then invoke the unmodified official Slime trainer."""
import os
import signal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from slime_plugins.models.qwen25_vl import register
register()
from slime.utils.arguments import parse_args
args = parse_args()
if os.environ.get("INTERACT_PARSE_ONLY") == "1":
    print("OFFICIAL_SLIME_ARGS_OK", args.hf_checkpoint, args.custom_model_provider_path, flush=True)
else:
    import ray
    from train import train
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        ray.init(address="local", num_cpus=16, num_gpus=2, object_store_memory=4 * 1024**3,
                 include_dashboard=False)
        train(args)
    finally:
        ray.shutdown()
