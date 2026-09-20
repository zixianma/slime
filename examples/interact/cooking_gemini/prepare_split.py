#!/usr/bin/env python3
"""Derive a parent-disjoint single-error Gemini-user split from CookSim v5."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENGINE_POOL = ROOT.parent / "cook-bench-engine/bench/v5/cases_composite_v4.json"
DEFAULT_PARENT_SPLIT = ROOT / "interact-runs/cooking-rl-v1-20260914-budget-v2"
BASELINE_VERSION = "cooking-single-error-gemini37-vulkan-v3-20260918"
NOVICE_VERSION = "cooking-single-error-gemini37-classic-novice-v1-20260919"
NOVICE_BUDGET_VERSION = "cooking-single-error-gemini37-classic-novice-budget-v2-20260919"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_names(parent_split, name):
    return [json.loads(line)["metadata"]["episode"]["task_id"]
            for line in (parent_split / f"{name}.jsonl").read_text().splitlines()]


def derive(parent):
    result = []
    pairs = list(zip(parent["subs"], parent["anchors"], strict=True))
    for index, (sub, anchor) in enumerate(pairs):
        if sub.get("family") != "error":
            continue
        name = f"s1_{sub['name']}"
        result.append({
            "name": name, "family": "composite", "cell": parent["cell"],
            "err": "composite", "inject": parent.get("inject", True),
            "subs": [sub], "n_errors": 1, "n_ruinous": int(bool(sub.get("ruinous"))),
            "order_pattern": "E", "anchors": [anchor],
            "rationale": f"Single-error curriculum case derived from {parent['name']}: {sub['err']}",
            "parent_case": parent["name"], "error_label": sub["err"],
            "source_sub_index": index,
        })
    return result


def row(case, split, pool, persona, wall_seconds):
    config = {
        "human": "gemini", "human_model": "gemini-3.7-flash", "persona": persona,
        "observation": "frames", "latency_ticks": 0, "scripted_accept": False,
        "wall_seconds": wall_seconds, "reward_version": "cooking_prevention_turns_v1",
        "renderer": "vulkan", "case_pool": str(pool.resolve()),
    }
    return {"prompt": [{"role": "user", "content": "Native assistant episode"}],
            "metadata": {"episode": {"engine": "cooking", "task_id": case["name"],
                                      "seed": 0, "config": config},
                         "split": split, "cell": case["cell"],
                         "parent_case": case["parent_case"], "error_label": case["error_label"]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--persona", choices=("baseline", "classic_novice"), default="baseline")
    parser.add_argument("--wall-seconds", type=int, choices=(1800, 3600), default=1800)
    parser.add_argument("--engine-pool", type=Path, default=DEFAULT_ENGINE_POOL)
    parser.add_argument("--parent-split", type=Path, default=DEFAULT_PARENT_SPLIT)
    args = parser.parse_args()
    if args.persona == "classic_novice" and args.wall_seconds == 3600:
        version = NOVICE_BUDGET_VERSION
    else:
        version = NOVICE_VERSION if args.persona == "classic_novice" else BASELINE_VERSION
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")

    engine_pool = args.engine_pool.resolve()
    parent_split = args.parent_split.resolve()
    parents = {x["name"]: x for x in json.loads(engine_pool.read_text())["cases"]}
    train_parents = parent_names(parent_split, "train")
    val_parents = parent_names(parent_split, "validation")
    assert len(train_parents) == 40 and len(val_parents) == 10
    assert set(train_parents).isdisjoint(val_parents)
    train = [case for name in train_parents for case in derive(parents[name])]
    val_full = [case for name in val_parents for case in derive(parents[name])]
    assert len(train) == 120 and len(val_full) == 30

    # One validation error per held-out parent/cell. Greedy balancing maximizes label
    # coverage; SHA ranking makes ties immutable and independent of model scores.
    ordered_val_parents = sorted(val_parents, key=lambda name: parents[name]["cell"])
    choices = [derive(parents[parent]) for parent in ordered_val_parents]
    best = None
    for candidate in itertools.product(*choices):
        counts = Counter(x["error_label"] for x in candidate)
        # Keep validation membership identical across persona experiments.
        tie = hashlib.sha256((BASELINE_VERSION + ":" + ":".join(x["name"] for x in candidate)).encode()).hexdigest()
        score = (len(counts), -sum(value * value for value in counts.values()), tie)
        if best is None or score > best[0]:
            best = (score, candidate, counts)
    _, val, labels = best
    val = list(val)
    assert len(val) == 10 and len({x["cell"] for x in val}) == 10

    pool = args.output / "cases_single_error.json"
    all_cases = train + val_full
    assert len({x["name"] for x in all_cases}) == len(all_cases)
    pool.write_text(json.dumps({"schema": "single-error-v1", "version": version,
                                "cases": all_cases}, indent=2) + "\n")

    files = {}
    for name, cases in (("train", train), ("validation", val), ("validation_full", val_full)):
        target = args.output / f"{name}.jsonl"
        target.write_text("".join(json.dumps(row(case, "validation" if name.startswith("validation") else "train", pool, args.persona, args.wall_seconds)) + "\n"
                                  for case in cases))
        files[name] = {"count": len(cases), "sha256": digest(target),
                       "parents": sorted({x["parent_case"] for x in cases}),
                       "labels": dict(sorted(Counter(x["error_label"] for x in cases).items()))}
    manifest = {
        "version": version,
        "derivation": "Split composite parents first, then isolate each native error sub-case",
        "generalization": "Held-out parent scenario variants; recipes/layouts overlap",
        "human": {"kind": "engine-native Gemini", "model": "gemini-3.7-flash",
                  "persona": args.persona, "stochastic": True, "paid_api_calls": True},
        "timing": {"base_wall_seconds": args.wall_seconds,
                   "composite_wall_seconds": 2 * args.wall_seconds,
                   "policy_decision_limit": 200},
        "reward": {"version": "cooking_prevention_turns_v1",
                   "formula": "success + .30*prevented - .02*min(false_flags,15) - .05*min(turns/200,1)"},
        "source_hashes": {"cases_composite_v4.json": digest(engine_pool),
                          "parent_manifest.json": digest(parent_split / "manifest.json")},
        "case_pool": {"path": str(pool.resolve()), "sha256": digest(pool), "count": len(all_cases)},
        "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "train": len(train),
                      "validation": len(val), "validation_full": len(val_full),
                      "validation_labels": dict(labels)}, sort_keys=True))


if __name__ == "__main__":
    main()
