"""Strict episode-weighted metrics for all-task checkpoint evaluations."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path

from interact_env.protocol import EpisodeSpec
from interact_env.slime_bridge.metrics import episode_metrics


def _catalog(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row["metadata"]
        spec = EpisodeSpec(**meta["episode"])
        result[spec.group_key] = {
            "split": meta["split"],
            "task_id": spec.task_id,
            "scenario_id": meta.get("scenario_id", spec.task_id),
            "cell": meta.get("cell"),
            "error_label": meta.get("error_label"),
            "parent_case": meta.get("parent_case"),
        }
    return result


def summarize(samples: list, *, engine: str, dataset: Path, expected: int) -> tuple[dict, list[dict]]:
    episode_metrics(samples)  # Reject incomplete or internally inconsistent trajectories.
    episodes = {sample.metadata["episode_id"]: sample for sample in samples}
    catalog = _catalog(dataset)
    if len(catalog) != expected or len(episodes) != expected:
        raise ValueError(f"incomplete full-suite pass: catalog={len(catalog)} episodes={len(episodes)} expected={expected}")

    by_group: dict[str, list] = defaultdict(list)
    for sample in episodes.values():
        if sample.metadata["engine"] != engine:
            raise ValueError("mixed engines in full-suite result")
        by_group[sample.metadata["group_key"]].append(sample)
    if set(by_group) != set(catalog) or any(len(group) != 1 for group in by_group.values()):
        raise ValueError("full-suite pass does not contain exactly one episode per frozen scenario")

    success_key = "task_success" if engine == "cooking" else "success"
    records = []
    for key, group in by_group.items():
        sample = group[0]
        components = dict(sample.metadata["reward_components"])
        records.append({
            **catalog[key],
            "group_key": key,
            "episode_id": sample.metadata["episode_id"],
            "reward": float(sample.reward),
            "success": float(components[success_key]),
            "turns": int(sample.metadata["num_turns"]),
            "outcome": sample.metadata.get("outcome"),
            "reward_components": components,
        })
    records.sort(key=lambda row: (row["split"], row["scenario_id"]))

    values: dict[str, float | int | str] = {
        "eval/episodes": len(records),
        "eval/tasks": len({row["task_id"] for row in records}),
        "eval/reward": sum(row["reward"] for row in records) / len(records),
        "eval/success": sum(row["success"] for row in records) / len(records),
        "eval/turns": sum(row["turns"] for row in records) / len(records),
    }
    for split in ("train", "validation"):
        rows = [row for row in records if row["split"] == split]
        values[f"eval/{split}/episodes"] = len(rows)
        values[f"eval/{split}/reward"] = sum(row["reward"] for row in rows) / len(rows)
        values[f"eval/{split}/success"] = sum(row["success"] for row in rows) / len(rows)
        task_values = defaultdict(list)
        for row in rows:
            task_values[row["task_id"]].append(row["success"])
        values[f"eval/{split}/task_macro_success"] = sum(
            sum(group) / len(group) for group in task_values.values()
        ) / len(task_values)

    task_values = defaultdict(list)
    for row in records:
        task_values[row["task_id"]].append(row["success"])
    values["eval/task_macro_success"] = sum(
        sum(group) / len(group) for group in task_values.values()
    ) / len(task_values)

    if engine == "cooking":
        for name in ("prevented_errors", "false_flags", "invalid_response_fraction", "decision_cap_reached"):
            values[f"eval/{name}"] = sum(
                float(row["reward_components"].get(name, 0)) for row in records
            ) / len(records)
        outcomes = {str(row["outcome"]) for row in records}
        for outcome in outcomes:
            values[f"eval/outcome/{outcome}"] = sum(row["outcome"] == outcome for row in records) / len(records)
    return values, records


def log_eval(_rollout_id, args, data, _extra_metrics):
    from slime.observability.logging_utils import log

    samples = [sample for dataset in data.values() for sample in dataset["samples"]]
    engine = os.environ["FULL_SUITE_ENGINE"]
    update = int(os.environ["FULL_SUITE_UPDATE"])
    label = os.environ["FULL_SUITE_LABEL"]
    dataset = Path(os.environ["FULL_SUITE_DATASET"])
    expected = int(os.environ["FULL_SUITE_EXPECTED"])
    values, records = summarize(samples, engine=engine, dataset=dataset, expected=expected)
    values.update({"eval/step": update, "eval/checkpoint_update": update, "eval/checkpoint_label": label})
    result = {
        "engine": engine,
        "checkpoint_label": label,
        "checkpoint_update": update,
        "checkpoint_load": str(args.load),
        "checkpoint_index": os.environ.get("FULL_SUITE_CHECKPOINT_INDEX"),
        "dataset": str(dataset.resolve()),
        "metrics": values,
        "scenarios": records,
    }
    job_dir = Path(os.environ["INTERACT_JOB_DIR"])
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("FULL_SUITE_RESULT " + json.dumps(values, sort_keys=True), flush=True)
    log(args, values, step_key="eval/step")
    return True
