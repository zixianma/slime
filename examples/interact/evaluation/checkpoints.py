#!/usr/bin/env python3
"""Resolve one checkpoint from a portable full-suite evaluation specification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re


LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def resolve(spec: dict, index: int, environ: dict[str, str]) -> dict:
    if spec.get("version") != 1 or not isinstance(spec.get("checkpoints"), list):
        raise ValueError("checkpoint specification must have version=1 and a checkpoints list")
    try:
        row = dict(spec["checkpoints"][index])
    except (IndexError, TypeError) as exc:
        raise ValueError(f"checkpoint index {index} is outside the specification") from exc

    expected = {"label", "update", "checkpoint_index"}
    if not expected <= row.keys() or bool(row.get("load")) == bool(row.get("load_env")):
        raise ValueError("each checkpoint needs label, update, checkpoint_index, and exactly one load source")
    if not isinstance(row["label"], str) or not LABEL.fullmatch(row["label"]):
        raise ValueError("checkpoint label must be safe for directory and W&B group names")
    if not isinstance(row["update"], int) or row["update"] < 0:
        raise ValueError("checkpoint update must be a non-negative integer")
    if not isinstance(row["checkpoint_index"], int) or row["checkpoint_index"] < -1:
        raise ValueError("checkpoint_index must be -1 or a non-negative integer")
    if row["checkpoint_index"] == -1 and row["update"] != 0:
        raise ValueError("only update 0 may use the frozen Hugging Face checkpoint")
    if row["checkpoint_index"] >= 0 and row["update"] != row["checkpoint_index"] + 1:
        raise ValueError("checkpoint update must equal checkpoint_index + 1")

    if row.get("load_env"):
        variable = row["load_env"]
        if not isinstance(variable, str) or variable not in environ:
            raise ValueError(f"checkpoint load environment variable is missing: {variable}")
        load = environ[variable]
    else:
        load = row["load"]
    if not isinstance(load, str) or not load or "\n" in load:
        raise ValueError("checkpoint load path must be a non-empty single line")
    return {
        "label": row["label"],
        "update": row["update"],
        "checkpoint_index": row["checkpoint_index"],
        "load": load,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    row = resolve(json.loads(args.spec.read_text()), args.index, dict(os.environ))
    for key in ("label", "update", "checkpoint_index", "load"):
        print(row[key])


if __name__ == "__main__":
    main()
