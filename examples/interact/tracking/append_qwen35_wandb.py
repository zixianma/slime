"""Append a finished continuation's scalar history to the canonical W&B run.

The target run is modified only after validating source/target coordinates. The
operation is idempotent: a second invocation refuses if the source points are
already present, and it never overwrites a conflicting point.
"""
import argparse
import json
import math
from pathlib import Path

import wandb

from examples.interact.tracking.merge_qwen35_wandb import PREFIXES, scalars, points


def rows_for(run):
    rows = [scalars(row) for row in run.scan_history(page_size=200)]
    return [row for row in rows if row]


def coordinate(row):
    for axis in ("train/step", "rollout/step", "eval/step"):
        if axis in row:
            return axis, row[axis]
    return None


def validate(source, target):
    if not source or not target:
        raise ValueError("source and target histories must not be empty")
    # All overlapping metric/axis points must agree exactly.
    for metric in sorted(set().union(*(r.keys() for r in source))):
        for axis in ("train/step", "rollout/step", "eval/step"):
            if not any(metric in row and axis in row for row in source):
                continue
            source_points = points([row for row in source if metric in row and axis in row], metric, axis)
            target_points = points([row for row in target if metric in row and axis in row], metric, axis) if any(
                metric in row and axis in row for row in target) else {}
            overlap = set(source_points) & set(target_points)
            for step in overlap:
                if source_points[step] != target_points[step]:
                    raise ValueError(f"conflict for {metric} at {axis}={step}")
    source_steps = {coordinate(row) for row in source if coordinate(row)}
    target_steps = {coordinate(row) for row in target if coordinate(row)}
    if source_steps & target_steps:
        # A source row can share a step with an existing row if it contains a
        # different metric; this is safe, but exact duplicate rows are not.
        if any(row in target for row in source):
            raise ValueError("source history is already appended")
    for row in source:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("source contains a non-finite metric")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="h6izegu2")
    parser.add_argument("--target", default="d0df345ba8f8")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    api = wandb.Api(timeout=60)
    project = "zixianma/interact-slime-rl"
    source_run = api.run(f"{project}/{args.source}")
    target_run = api.run(f"{project}/{args.target}")
    if source_run.state != "finished":
        raise ValueError(f"source run is {source_run.state}, not finished")
    source = rows_for(source_run)
    target_before = rows_for(target_run)
    validate(source, target_before)
    source_train = points(source, "train/grad_norm", "train/step")
    source_eval = points(source, "eval/success", "eval/step")
    if set(source_train) != set(range(9, 15)) or set(source_eval) != {12, 15}:
        raise ValueError(f"unexpected source coverage: train={source_train.keys()} eval={source_eval.keys()}")

    with wandb.init(entity="zixianma", project="interact-slime-rl", id=args.target,
                    resume="allow", mode="online", settings=wandb.Settings(console="off",
                    disable_code=True, x_disable_stats=True, x_disable_meta=True)) as run:
        for axis in ("train/step", "rollout/step", "eval/step"):
            run.define_metric(axis)
        for prefix, axis in [("train/*", "train/step"), ("rollout/*", "rollout/step"),
                             ("eval/*", "eval/step"), ("perf/*", "rollout/step")]:
            run.define_metric(prefix, step_metric=axis, step_sync=False)
        for row in source:
            run.log({**row, "merge/source_run_id": args.source})
        run.summary.update({"completed_updates": 15, "historical_merge": True,
                            "continuation_source_run": args.source,
                            "last_eval_completed_updates": 15,
                            "last_eval_success": source_eval[15]})

    api.flush()
    target_after = rows_for(api.run(f"{project}/{args.target}"))
    # Validate the newly visible history and ensure every source scalar survived.
    validate(source, target_after)
    target_train = points(target_after, "train/grad_norm", "train/step")
    target_eval = points(target_after, "eval/success", "eval/step")
    if not set(range(15)).issubset(target_train) or not {0, 2, 6, 9, 12, 15}.issubset(target_eval):
        raise ValueError("read-back history is incomplete")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps({"status": "verified", "target": args.target,
        "source": args.source, "target_rows": len(target_after),
        "train_steps": sorted(target_train), "eval_steps": sorted(target_eval)}, indent=2) + "\n")
    print(json.dumps({"status": "verified", "target": args.target, "source": args.source,
                      "train_steps": sorted(target_train), "eval_steps": sorted(target_eval)}), flush=True)


if __name__ == "__main__":
    main()
