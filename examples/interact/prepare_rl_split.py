"""Create an immutable RL split without relabeling the offline comparison."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path


SPLIT_VERSION = "screensim_rl_v2_18train_12val"
OLD_VALIDATION = {
    "lock_screen_lockdown", "siri_lock_screen_privacy",
    "rent_share_standard", "rent_landlord_instant",
}


def build_split(source, source_sha256):
    scenarios = deepcopy(source["scenarios"])
    ids = [s["id"] for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario IDs")
    by_task = {}
    for scenario in scenarios:
        task = scenario["spec"]["task_id"]
        by_task.setdefault(task, []).append(scenario)
        expected = "validation" if task in OLD_VALIDATION else "train"
        if scenario["split"] != expected:
            raise ValueError("source is not the frozen comparison split")
    if Counter(s["split"] for s in scenarios) != {"train": 21, "validation": 9}:
        raise ValueError("unexpected source scenario counts")
    if len(by_task) != 11 or len(by_task.get("hearing_setup", [])) != 3:
        raise ValueError("expected 11 tasks and three hearing_setup scenarios")
    for scenario in scenarios:
        if scenario["spec"]["task_id"] == "hearing_setup":
            scenario["split"] = "validation"
        # Offline sampling targets are not an RL training schedule.
        scenario.pop("attempts", None)
    return {
        "split_version": SPLIT_VERSION,
        "source_manifest_sha256": source_sha256,
        "engine_revision": source["engine_revision"],
        "profile": source["profile"],
        "persona": source["persona"],
        "models": deepcopy(source["models"]),
        "decoding": deepcopy(source["decoding"]),
        "scenarios": scenarios,
        "counts": {"train_tasks": 6, "train_scenarios": 18,
                   "validation_tasks": 5, "validation_scenarios": 12},
        "selection_reason": "Move hearing_setup as a whole task to add accessibility coverage; not selected by model scores.",
        "evaluation_note": "Development validation, not an untouched test: offline results informed model selection. Re-evaluate the initial checkpoint and trained checkpoints under one fixed protocol; do not relabel historical W&B runs.",
    }


def prompt_rows(manifest, split):
    return [
        {"prompt": [{"role": "user", "content": "Native assistant episode"}],
         "metadata": {"episode": s["spec"], "scenario_id": s["id"],
                      "split": split, "split_version": manifest["split_version"]}}
        for s in manifest["scenarios"] if s["split"] == split
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    result = build_split(json.loads(raw), hashlib.sha256(raw).hexdigest())
    result["source_manifest"] = str(args.source.resolve())
    # Existing experiment directories are never overwritten or partially reused.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(result, indent=2) + "\n"
    (args.output_dir / "manifest.json").write_text(payload)
    for split in ("train", "validation"):
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in prompt_rows(result, split)))
    print(json.dumps({"output_dir": str(args.output_dir), **result["counts"],
                      "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
