"""Build a compact Cooking full-suite replay dashboard.

The evaluator stores overlapping frame windows on every policy decision.  Exporting all of
them would make a multi-gigabyte website, so this builder keeps every dialogue turn and one
compressed, newest observation frame every ``--image-stride`` turns.  A later invocation
automatically replaces update-3/update-12 placeholders once their audit result exists.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from examples.interact.tools.visualization.replay_common import (
    cooking_response as parse_response,
    cooking_task_instruction as task_instruction,
    cooking_user_text as current_user_text,
)


DEFAULT_OUT = ROOT / "interact-runs/cooking-gemini-full-suite-replay"
CHECKPOINTS = (
    ("update-0", "Update 0", "Base policy"),
    ("update-12", "Update 12", "Trained checkpoint"),
)

def export_frame(uri: str, images: Path, width: int, quality: int) -> str:
    encoded = uri.split(",", 1)[1]
    source = base64.b64decode(encoded)
    with Image.open(BytesIO(source)) as image:
        image = image.convert("RGB")
        if image.width > width:
            height = round(image.height * width / image.width)
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, "WEBP", quality=quality, method=4)
    data = output.getvalue()
    name = hashlib.sha256(data).hexdigest() + ".webp"
    target = images / name
    if not target.exists():
        target.write_bytes(data)
    # Image URLs are resolved by the browser relative to the dashboard document, not the
    # fetched episode JSON file.
    return "images/" + name


def reward_explanation(components: dict[str, Any]) -> dict[str, float]:
    success = float(components.get("task_success", 0))
    prevented = float(components.get("prevented_errors", 0))
    false_flags = float(components.get("false_flags", 0))
    turns = float(components.get("assistant_calls", 0))
    invalid_fraction = float(components.get("invalid_response_fraction", 0))
    # cooking_prevention_turns_v1: success + prevention shaping - turn/flag/invalid costs.
    return {
        "task_success": success,
        "prevention_bonus": 0.3 * prevented,
        "false_flag_cost": -0.1 * false_flags,
        "turn_cost": -0.001 * turns,
        "invalid_cost": -0.15 * invalid_fraction,
    }


def export_episode(
    root: Path,
    row: dict[str, Any],
    destination: Path,
    images: Path,
    image_stride: int,
    image_width: int,
    image_quality: int,
) -> dict[str, Any]:
    episode = root / "episodes" / row["episode_id"]
    decisions_path = episode / "decisions.jsonl"
    lines = decisions_path.read_text().splitlines()
    turns: list[dict[str, Any]] = []
    instruction = ""
    last = len(lines) - 1
    for index, line in enumerate(lines):
        record = json.loads(line)
        decision = record["decision"]
        observation = decision["observation"]
        if not instruction:
            instruction = task_instruction(observation.get("system", ""))
        text, flag, progress = parse_response(record.get("response", ""))
        take_frame = index == 0 or index == last or index % image_stride == 0 or flag is not None
        frame = None
        if take_frame and observation.get("images"):
            frame = export_frame(observation["images"][-1], images, image_width, image_quality)
        turns.append(
            {
                "index": index,
                "tick": decision.get("tick"),
                "user": current_user_text(observation.get("user", "")),
                "assistant": text,
                "flag": flag,
                "progress": progress,
                "raw": record.get("response", ""),
                "frame": frame,
            }
        )

    report_path = episode / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    components = row.get("reward_components", {})
    payload = {
        "scenario": row["scenario_id"],
        "episode_id": row["episode_id"],
        "task_instruction": instruction,
        "split": row["split"],
        "cell": row.get("cell"),
        "parent_case": row.get("parent_case"),
        "error_label": row.get("error_label"),
        "outcome": row.get("outcome"),
        "success": bool(row.get("success")),
        "reward": row.get("reward"),
        "reward_components": components,
        "reward_terms": reward_explanation(components),
        "decision_cap_reached": (episode / "decision_timeout.json").exists(),
        "native_report_available": report_path.exists(),
        "native_summary": {
            key: report.get(key)
            for key in ("err", "error_committed", "errors_offered", "errors_planned", "false_positives", "flags", "outcome", "ticks")
            if key in report
        },
        "turns": turns,
    }
    target = destination / "episodes" / f"{row['scenario_id']}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return {
        "file": f"episodes/{row['scenario_id']}.json",
        "scenario": row["scenario_id"],
        "split": row["split"],
        "cell": row.get("cell"),
        "parent_case": row.get("parent_case"),
        "error_label": row.get("error_label"),
        "outcome": row.get("outcome"),
        "success": bool(row.get("success")),
        "reward": row.get("reward"),
        "turns": row.get("turns"),
        "false_flags": components.get("false_flags"),
        "prevented_errors": components.get("prevented_errors"),
        "decision_cap_reached": payload["decision_cap_reached"],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    cells: dict[str, dict[str, int]] = {}
    errors: dict[str, dict[str, int]] = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
        for key, bucket in (("cell", cells), ("error_label", errors)):
            label = row.get(key) or "unknown"
            item = bucket.setdefault(label, {"episodes": 0, "successes": 0})
            item["episodes"] += 1
            item["successes"] += int(row["success"])
    return {"outcomes": outcomes, "cells": cells, "errors": errors}


def partial_rows(source: Path, baseline: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover finished rows before the evaluator writes its final aggregate audit."""
    discovered: dict[str, tuple[int, dict[str, Any]]] = {}
    for audit_path in source.glob("episodes/*/training_audit.json"):
        try:
            audit = json.loads(audit_path.read_text())
            sample = audit[0]
            metadata = sample["metadata"]
            scenario = metadata["task_id"]
            original = baseline[scenario]
            row = {
                **original,
                "episode_id": metadata["episode_id"],
                "outcome": metadata.get("outcome", "unknown"),
                "success": float(metadata.get("reward_components", {}).get("task_success", 0)),
                "reward": sample.get("reward", 0),
                "reward_components": metadata.get("reward_components", {}),
                "turns": metadata.get("num_turns", len(audit)),
            }
            modified = audit_path.stat().st_mtime_ns
            if scenario not in discovered or modified > discovered[scenario][0]:
                discovered[scenario] = (modified, row)
        except (KeyError, IndexError, json.JSONDecodeError, OSError):
            # A worker may still be atomically finishing the file. It will be included on refresh.
            continue
    return [value[1] for value in discovered.values()]


def partial_metrics(rows: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    count = len(rows)
    successes = sum(float(row["success"]) for row in rows)
    rewards = sum(float(row["reward"]) for row in rows)
    caps = sum(
        (source / "episodes" / row["episode_id"] / "decision_timeout.json").exists()
        for row in rows
    )
    values: dict[str, Any] = {
        "eval/episodes": count,
        "eval/success": successes / count if count else 0,
        "eval/reward": rewards / count if count else 0,
        "eval/decision_cap_reached": caps / count if count else 0,
    }
    for split in ("train", "validation"):
        selected = [row for row in rows if row["split"] == split]
        if selected:
            values[f"eval/{split}/episodes"] = len(selected)
            values[f"eval/{split}/success"] = sum(float(row["success"]) for row in selected) / len(selected)
    return values


def build_checkpoint(
    source: Path,
    destination: Path,
    images: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    status: str,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "episodes").mkdir(exist_ok=True)
    wanted = {row["scenario_id"] for row in rows}
    existing_index = destination / "index.json"
    index: list[dict[str, Any]] = []
    if reuse_existing and existing_index.is_file():
        index = [
            item for item in json.loads(existing_index.read_text())
            if item["scenario"] in wanted and (destination / item["file"]).is_file()
        ]
    exported = {item["scenario"] for item in index}
    pending = [row for row in rows if row["scenario_id"] not in exported]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                export_episode,
                source,
                row,
                destination,
                images,
                args.image_stride,
                args.image_width,
                args.image_quality,
            )
            for row in pending
        ]
        for count, future in enumerate(as_completed(futures), 1):
            index.append(future.result())
            if count % 15 == 0:
                print(f"{source.name}: exported {count}/{len(pending)} new episodes", flush=True)
    index.sort(key=lambda item: item["scenario"])
    (destination / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    return {
        "status": status,
        "metrics": metrics,
        "episodes": len(index),
        "completed": len(index),
        "expected": 150,
        "aggregate": aggregate(index),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--image-stride", type=int, default=10)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-quality", type=int, default=48)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--refresh", action="store_true", help="Update an existing dashboard and reuse completed checkpoints")
    parser.add_argument("--html-only", action="store_true", help="Refresh only the dashboard UI without rebuilding exports")
    args = parser.parse_args()
    if args.output.exists() and not args.refresh:
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")

    if args.html_only:
        if not args.output.is_dir() or not (args.output / "metadata.json").is_file():
            raise SystemExit("--html-only requires an existing dashboard output")
        shutil.copyfile(Path(__file__).with_name("cooking_full_suite_replay.html"), args.output / "index.html")
        print(json.dumps({"output": str(args.output), "html_only": True}))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    images = args.output / "images"
    images.mkdir(exist_ok=True)
    baseline_result = json.loads((args.eval_root / "update-0/audit/result.json").read_text())
    baseline = {row["scenario_id"]: row for row in baseline_result["scenarios"]}
    checkpoints = []
    for directory, label, description in CHECKPOINTS:
        source = args.eval_root / directory
        audit = source / "audit/result.json"
        checkpoint = {"directory": directory, "label": label, "description": description}
        if audit.exists():
            result = json.loads(audit.read_text())
            destination = args.output / "checkpoints" / directory
            existing_index = destination / "index.json"
            if args.refresh and existing_index.exists() and len(json.loads(existing_index.read_text())) == len(result["scenarios"]):
                index = json.loads(existing_index.read_text())
                checkpoint.update({
                    "status": "complete", "metrics": result["metrics"], "episodes": len(index),
                    "completed": len(index), "expected": 150, "aggregate": aggregate(index),
                })
            else:
                checkpoint.update(
                    build_checkpoint(
                        source, destination, images, args, result["scenarios"], result["metrics"], "complete",
                        reuse_existing=args.refresh,
                    )
                )
        else:
            rows = partial_rows(source, baseline) if source.exists() else []
            if rows:
                checkpoint.update(
                    build_checkpoint(
                        source, args.output / "checkpoints" / directory, images, args,
                        rows, partial_metrics(rows, source), "running", reuse_existing=args.refresh,
                    )
                )
            else:
                checkpoint.update({"status": "queued", "completed": 0, "expected": 150})
        checkpoints.append(checkpoint)

    metadata = {
        "title": "Cooking policy evaluation",
        "model": "Qwen3.5-4B",
        "user": "Gemini 3.7 Flash · classic novice",
        "image_policy": f"Newest observation frame every {args.image_stride} policy turns, plus flagged/first/final turns",
        "checkpoints": checkpoints,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copyfile(Path(__file__).with_name("cooking_full_suite_replay.html"), args.output / "index.html")
    (args.output / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    print(json.dumps({"output": str(args.output), "checkpoints": checkpoints, "images": len(list(images.iterdir()))}, indent=2))


if __name__ == "__main__":
    main()
