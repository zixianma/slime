#!/usr/bin/env python3
"""Retrospectively stress-test candidate CookSim rewards on archived eval episodes."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def load_rows(root: Path, weights: set[str]) -> list[dict]:
    rows = []
    for audit_path in sorted((root / "episodes").glob("*/training_audit.json")):
        # Evaluation files are large because they include token arrays. A cheap text check
        # avoids parsing training episodes, while JSON parsing keeps field extraction exact.
        if '"evaluation": true' not in audit_path.read_text(errors="ignore"):
            continue
        audit = json.loads(audit_path.read_text())
        if not audit:
            continue
        first = audit[0]
        weight = str((first.get("weight_versions") or [""])[0])
        if weight not in weights:
            continue
        meta = first["metadata"]
        report_path = Path(meta["artifacts"]["native_report"])
        report = json.loads(report_path.read_text())
        error_injections = [x for x in report.get("injections", [])
                            if x.get("family") == "error"]
        rows.append({
            "episode_id": meta["episode_id"],
            "weight": weight,
            "task": meta["task_id"],
            "group": meta["group_key"],
            "turns": int(meta["num_turns"]),
            "success": int(report.get("outcome") == "won"),
            "false_flags": len(report.get("false_positives", [])),
            "invalid_fraction": float(meta["reward_components"]["invalid_response_fraction"]),
            "offered": int(report.get("errors_offered", 0)),
            "detected": sum(bool(x.get("flagged_valid")) for x in error_injections),
            "convinced": sum(bool(x.get("convinced")) for x in error_injections),
            "prevented": sum(bool(x.get("prevented")) for x in error_injections),
        })
    return rows


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(rows: list[dict]) -> dict:
    # The mode is the least assumption-heavy retrospective proxy for the nominal length.
    # A new experiment should instead freeze references from no-intervention controls.
    references = {}
    for task in sorted({x["task"] for x in rows}):
        counts = Counter(x["turns"] for x in rows if x["task"] == task)
        references[task] = min(k for k, v in counts.items() if v == max(counts.values()))

    for x in rows:
        ref = references[x["task"]]
        x["excess_turn_ratio"] = min(max(x["turns"] - ref, 0) / max(ref, 1), 1)
        prevented = int(x["prevented"] > 0)
        x["reward_count"] = (x["success"] + .30 * prevented
                             - .10 * min(x["false_flags"], 3)
                             - .05 * x["excess_turn_ratio"])
        # A smoother comparator: reaching a 5% false-flag rate spends the full 0.10 budget.
        fp_rate = x["false_flags"] / max(x["turns"], 1)
        x["reward_rate"] = (x["success"] + .30 * prevented
                            - .10 * min(fp_rate / .05, 1)
                            - .05 * x["excess_turn_ratio"])
        # Simplest candidate for same-scenario GRPO: unavoidable scenario length is
        # common within a group and cancels during advantage normalization.
        x["reward_simple"] = (x["success"] + .30 * prevented
                              - .02 * min(x["false_flags"], 15)
                              - .05 * min(x["turns"] / 200, 1))

    def one(part: list[dict]) -> dict:
        groups = defaultdict(list)
        for x in part:
            groups[(x["weight"], x["group"])].append(x)
        prevention_pairs = []
        for group in groups.values():
            for i, left in enumerate(group):
                for right in group[i + 1:]:
                    if bool(left["prevented"]) == bool(right["prevented"]):
                        continue
                    better, worse = ((left, right) if left["prevented"] else (right, left))
                    prevention_pairs.append(better["reward_simple"] - worse["reward_simple"])
        rewards = [x["reward_count"] for x in part]
        rate_rewards = [x["reward_rate"] for x in part]
        simple_rewards = [x["reward_simple"] for x in part]
        success_rewards = [x["reward_count"] for x in part if x["success"]]
        failure_rewards = [x["reward_count"] for x in part if not x["success"]]
        return {
            "episodes": len(part),
            "successes": sum(x["success"] for x in part),
            "episodes_any_prevention": sum(x["prevented"] > 0 for x in part),
            "prevention_mixed_group_fraction": (
                sum(len({bool(x["prevented"]) for x in group}) > 1 for group in groups.values())
                / len(groups)),
            "offered_errors": sum(x["offered"] for x in part),
            "detected_errors": sum(x["detected"] for x in part),
            "convinced_errors": sum(x["convinced"] for x in part),
            "prevented_errors": sum(x["prevented"] for x in part),
            "false_flags_mean": statistics.fmean(x["false_flags"] for x in part),
            "false_flag_cap_fraction": sum(x["false_flags"] >= 3 for x in part) / len(part),
            "excess_turn_nonzero_fraction": sum(x["excess_turn_ratio"] > 0 for x in part) / len(part),
            "count_reward": {
                "mean": statistics.fmean(rewards),
                "min": min(rewards),
                "p25": quantile(rewards, .25),
                "median": statistics.median(rewards),
                "p75": quantile(rewards, .75),
                "max": max(rewards),
                "unique_rounded": len({round(v, 6) for v in rewards}),
                "mixed_group_fraction": (sum(len({round(x["reward_count"], 8) for x in group}) > 1
                                             for group in groups.values()) / len(groups)),
                "max_failure": max(failure_rewards) if failure_rewards else None,
                "min_success": min(success_rewards) if success_rewards else None,
            },
            "rate_reward": {
                "mean": statistics.fmean(rate_rewards),
                "min": min(rate_rewards),
                "median": statistics.median(rate_rewards),
                "max": max(rate_rewards),
                "unique_rounded": len({round(v, 6) for v in rate_rewards}),
                "mixed_group_fraction": (sum(len({round(x["reward_rate"], 8) for x in group}) > 1
                                             for group in groups.values()) / len(groups)),
            },
            "simple_reward": {
                "formula": "success + .30*any_prevented - .02*min(false_flags,15) - .05*min(turns/200,1)",
                "mean": statistics.fmean(simple_rewards),
                "min": min(simple_rewards),
                "p25": quantile(simple_rewards, .25),
                "median": statistics.median(simple_rewards),
                "p75": quantile(simple_rewards, .75),
                "max": max(simple_rewards),
                "unique_rounded": len({round(v, 6) for v in simple_rewards}),
                "mixed_group_fraction": (sum(len({round(x["reward_simple"], 8) for x in group}) > 1
                                             for group in groups.values()) / len(groups)),
                "false_flag_cap_fraction": sum(x["false_flags"] >= 15 for x in part) / len(part),
                "prevention_pairs": len(prevention_pairs),
                "prevention_preferred_fraction": (
                    sum(delta > 0 for delta in prevention_pairs) / len(prevention_pairs)
                    if prevention_pairs else None),
                "prevention_margin_median": (
                    statistics.median(prevention_pairs) if prevention_pairs else None),
            },
        }

    return {
        "scope": "archived scripted-user composite validation; not Gemini/single-error data",
        "reference_turns_by_task": references,
        "all": one(rows),
        "by_weight": {weight: one([x for x in rows if x["weight"] == weight])
                      for weight in sorted({x["weight"] for x in rows})},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--weights", nargs="+", default=["4", "7"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(load_rows(args.root, set(args.weights)))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
