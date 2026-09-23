#!/usr/bin/env python3
"""Build immutable all-task evaluation datasets from the frozen RL splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from interact_env.protocol import EpisodeSpec


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def group_key(row: dict) -> str:
    return EpisodeSpec(**row["metadata"]["episode"]).group_key


def build(engine: str, sources: list[tuple[str, Path]], output: Path) -> dict:
    rows: list[dict] = []
    source_records = []
    for split, path in sources:
        source_rows = read_jsonl(path)
        for row in source_rows:
            if row["metadata"]["episode"]["engine"] != engine:
                raise ValueError(f"{path} contains a non-{engine} episode")
            if row["metadata"].get("split") != split:
                raise ValueError(f"{path} contains a row outside split={split}")
        rows.extend(source_rows)
        source_records.append(
            {"split": split, "path": str(path.resolve()), "count": len(source_rows), "sha256": sha256(path)}
        )

    keys = [group_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{engine} full suite contains duplicate environment specifications")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return {
        "engine": engine,
        "path": str(output.resolve()),
        "count": len(rows),
        "sha256": sha256(output),
        "splits": {split: sum(row["metadata"]["split"] == split for row in rows) for split, _ in sources},
        "sources": source_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cooking-split", type=Path, required=True)
    parser.add_argument("--screensim-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    suites = {
        "cooking": build(
            "cooking",
            [("train", args.cooking_split / "train.jsonl"),
             ("validation", args.cooking_split / "validation_full.jsonl")],
            args.output / "cooking-all.jsonl",
        ),
        "screensim": build(
            "screensim",
            [("train", args.screensim_split / "train.jsonl"),
             ("validation", args.screensim_split / "validation.jsonl")],
            args.output / "screensim-all.jsonl",
        ),
    }
    if suites["cooking"]["count"] != 150 or suites["cooking"]["splits"] != {"train": 120, "validation": 30}:
        raise ValueError("unexpected Cooking full-suite cardinality")
    if suites["screensim"]["count"] != 30 or suites["screensim"]["splits"] != {"train": 18, "validation": 12}:
        raise ValueError("unexpected ScreenSim full-suite cardinality")
    manifest = {"version": 1, "protocol": "one rollout per frozen scenario and checkpoint", "suites": suites}
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"FULL_SUITE_DATA_OK manifest={manifest_path} cooking=150 screensim=30", flush=True)


if __name__ == "__main__":
    main()
