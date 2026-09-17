"""Backfill scalar RL metrics from completed stages; offline unless explicitly online.

Never uploads prompts, images, transcripts, checkpoints, credentials or raw logs.
Native audit records supply episode metrics; trainer logs supply actual step metrics.
"""
import argparse
import ast
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys
import uuid
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from interact_env.slime_bridge.metrics import episode_metrics


def stage_records(log_path, data_dir, samples_per_rollout):
    status = json.loads(log_path.with_suffix(".status.json").read_text())
    if status.get("returncode") != 0:
        raise ValueError(f"refusing failed stage: {log_path}")
    steps = {}
    for line in log_path.read_text().splitlines():
        match = re.search(r"model\.py:\d+ - step (\d+): (\{.*\})", line)
        if match:
            step = int(match[1])
            metrics = ast.literal_eval(match[2])
            if not all(isinstance(value, (float, int)) and math.isfinite(value) for value in metrics.values()):
                raise ValueError("non-finite/non-scalar training metric")
            if step in steps and steps[step] != metrics:
                raise ValueError("conflicting training steps")
            steps[step] = metrics
    by_rollout = defaultdict(list)
    for path in sorted((data_dir / "episodes").glob("*/training_audit.json")):
        audit = json.loads(path.read_text())
        if not audit:
            raise ValueError("empty training audit")
        step = audit[0]["rollout_id"] // samples_per_rollout
        if step not in steps:
            raise ValueError("episode has no completed training step")
        native = json.loads((path.parent / "report.json").read_text())
        for turn in audit:
            meta = turn["metadata"]
            components = dict(meta["reward_components"])
            if meta["engine"] == "screensim":
                components.update(in_time_success=native["success"], goal_ok=native["goal_ok"],
                                  final_tick=native["final_tick"])
            by_rollout[step].append(SimpleNamespace(reward=turn["reward"],
                metadata={**meta, "reward_components": components}))
    if steps.keys() != by_rollout.keys():
        raise ValueError("incomplete rollout/step coverage")
    result = {}
    for step, samples in by_rollout.items():
        metrics = episode_metrics(samples)
        if metrics["rollout/episodes/count"] != samples_per_rollout:
            raise ValueError("unexpected episode count")
        result[step] = {**steps[step], **metrics, "rollout/step": step, "train/step": step}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs=2, action="append", metavar=("LOG", "DATA_DIR"), required=True)
    parser.add_argument("--samples-per-rollout", type=int, default=2)
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--project", default="interact-slime-rl")
    parser.add_argument("--entity")
    parser.add_argument("--name", default="screensim-qwen25vl3b-job289525-backfill")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "online" and not args.entity:
        parser.error("online backfill requires an explicitly approved --entity and --project")
    records = {}
    for log, directory in args.stage:
        more = stage_records(Path(log), Path(directory), args.samples_per_rollout)
        if records.keys() & more.keys():
            raise ValueError("overlapping training steps")
        records.update(more)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    import wandb
    # A generated ID + resume='never' cannot attach to an inherited baseline run ID.
    with wandb.init(project=args.project, entity=args.entity, name=args.name,
                    id=uuid.uuid4().hex, resume="never", mode=args.mode,
                    dir=str(args.output), tags=["screensim", "slime", "historical-backfill", "smoke"],
                    config={"historical_backfill": True, "model": "Qwen2.5-VL-3B-Instruct",
                            "reward_version": "screensim_f1_v1", "human": "scripted_human_v1",
                            "samples_per_rollout": args.samples_per_rollout,
                            "source_stages": [Path(log).stem for log, _ in args.stage]},
                    settings=wandb.Settings(console="off", disable_code=True)) as run:
        run.define_metric("train/step")
        run.define_metric("train/*", step_metric="train/step")
        run.define_metric("rollout/step")
        run.define_metric("rollout/*", step_metric="rollout/step")
        for step, metrics in sorted(records.items()):
            run.log(metrics, step=step)
        run.summary["completed_training_steps"] = len(records)
        run.summary["historical_backfill"] = True
        location = {"id": run.id, "mode": args.mode, "project": args.project,
                    "entity": args.entity, "directory": run.dir,
                    "url": run.url if args.mode == "online" else None}
    (args.output / "wandb_run.json").write_text(json.dumps(location, indent=2) + "\n")
    print(json.dumps(location))


if __name__ == "__main__":
    main()
