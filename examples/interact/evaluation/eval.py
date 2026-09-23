#!/usr/bin/env python3
"""Ray lifecycle and fail-closed provenance checks for a full-suite eval."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from slime.utils.arguments import parse_args

    args = parse_args()
    engine = os.environ["FULL_SUITE_ENGINE"]
    dataset = Path(os.environ["FULL_SUITE_DATASET"])
    manifest_path = Path(os.environ["FULL_SUITE_MANIFEST"])
    manifest = json.loads(manifest_path.read_text())
    suite = manifest["suites"][engine]
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != suite["sha256"]:
        raise ValueError("full-suite dataset hash mismatch")
    if Path(suite["path"]).resolve() != dataset.resolve():
        raise ValueError("full-suite dataset path differs from its manifest")
    if int(os.environ["FULL_SUITE_EXPECTED"]) != suite["count"]:
        raise ValueError("full-suite expected count differs from its manifest")
    if args.num_rollout != 0 or args.eval_interval is None or args.n_samples_per_eval_prompt != 1:
        raise ValueError("full-suite evaluation must be eval-only with one rollout per scenario")
    if len(args.eval_datasets) != 1 or Path(args.eval_datasets[0].path).resolve() != dataset.resolve():
        raise ValueError("full-suite evaluator received the wrong eval dataset")

    checkpoint_index = int(os.environ["FULL_SUITE_CHECKPOINT_INDEX"])
    if checkpoint_index >= 0:
        checkpoint = Path(args.load)
        marker = checkpoint / "latest_checkpointed_iteration.txt"
        if not marker.is_file() or not (checkpoint / f"iter_{checkpoint_index:07d}" / ".metadata").is_file():
            raise ValueError("requested numeric checkpoint is incomplete")
        if args.ckpt_step != checkpoint_index:
            raise ValueError("loaded checkpoint index differs from requested index")
    elif not (Path(args.load) / "config.json").is_file():
        raise ValueError("update-0 evaluation must load the frozen HF checkpoint")

    if engine == "cooking":
        worker_python = os.environ.get("COOKING_WORKER_PYTHON")
        if not worker_python or not Path(worker_python).is_file():
            raise ValueError("COOKING_WORKER_PYTHON must name the CookBench worker interpreter")
        args.interact["worker_python"] = worker_python
    args.interact_engine = engine
    args.interact_eval_protocol = manifest["protocol"]
    args.interact_eval_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    args.interact_checkpoint_label = os.environ["FULL_SUITE_LABEL"]
    args.interact_checkpoint_update = int(os.environ["FULL_SUITE_UPDATE"])
    args.interact_model_revision = os.environ.get(
        "Q35_MODEL_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    if os.environ.get("INTERACT_PARSE_ONLY") == "1":
        print(
            f"FULL_SUITE_ARGS_OK engine={engine} count={suite['count']} "
            f"update={args.interact_checkpoint_update} checkpoint_index={checkpoint_index}",
            flush=True,
        )
        return

    import ray
    from train import train

    signal.signal(signal.SIGTERM, lambda signum, _frame: (_ for _ in ()).throw(SystemExit(128 + signum)))
    try:
        ray.init(
            address="local",
            num_cpus=int(os.environ["FULL_SUITE_CPUS"]),
            num_gpus=int(os.environ["FULL_SUITE_GPUS"]),
            object_store_memory=4 * 1024**3,
            include_dashboard=False,
        )
        train(args)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
