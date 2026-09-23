"""Export paired ScreenSim Gemini-user update-0/update-12 full-suite evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = ROOT / "interact-runs/screensim-gemini-full-suite-replay"

CURATED = {
    "stop_profiling__c3": {
        "title": "Clearest positive change",
        "summary": (
            "Update 12 catches the second extra-write at injection time and gets it repaired; "
            "update 0 notices it only later and fails to prevent it. The task changes from failure "
            "to success with no false flags in either rollout."
        ),
        "rank": 1,
    },
    "standby_bedside__c3": {
        "title": "Faster recovery from the second mistake",
        "summary": (
            "Update 12 catches the wrong target immediately and finishes five ticks earlier. "
            "Update 0 gets stuck arguing about the visible setting. The trained rollout succeeds, "
            "although it also emits one false flag."
        ),
        "rank": 2,
    },
    "app_privacy_quarantine__c3": {
        "title": "Outcome improves, dialogue remains imperfect",
        "summary": (
            "The trained rollout recovers the wrong-target mistake and completes the requested state. "
            "Its conversation still contains contradictory instructions, so this is a useful warning "
            "that engine success is not the same as uniformly better language."
        ),
        "rank": 3,
    },
    "lock_screen_lockdown__c1": {
        "title": "Better detection without task completion",
        "summary": (
            "F1 rises from 0.33 to 1.00 and false flags fall from three to zero, but both policies fail "
            "the end-to-end task. This separates monitoring quality from successful guidance."
        ),
        "rank": 4,
    },
    "hearing_setup__c3": {
        "title": "Important regression",
        "summary": (
            "Update 0 catches and prevents both mistakes and succeeds. Update 12 misreads the user's "
            "state, prevents neither mistake, and the user exits before the goal is complete."
        ),
        "rank": 5,
    },
}


def load_result(eval_root: Path, label: str) -> tuple[Path, dict[str, dict]]:
    root = eval_root / label
    result = json.loads((root / "audit/result.json").read_text())
    rows = {row["scenario_id"]: row for row in result["scenarios"]}
    if len(rows) != 30:
        raise ValueError(f"{label} does not contain all 30 scenarios")
    return root, rows


def parse_response(raw: str) -> tuple[str, object]:
    try:
        value = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return raw, None
    return value.get("text", raw), value.get("flag")


def export_image(uri: str, output: Path) -> str:
    header, encoded = uri.split(",", 1)
    if header != "data:image/png;base64":
        raise ValueError(f"unsupported observation image: {header}")
    data = base64.b64decode(encoded, validate=True)
    name = hashlib.sha256(data).hexdigest() + ".png"
    target = output / "images" / name
    if not target.exists():
        target.write_bytes(data)
    return "images/" + name


def export_episode(root: Path, row: dict, output: Path) -> dict:
    directory = root / "episodes" / row["episode_id"]
    report = json.loads((directory / "report.json").read_text())
    spec = json.loads((directory / "spec.json").read_text())
    decisions = [json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines()]
    audit = json.loads((directory / "training_audit.json").read_text())
    if not (len(decisions) == len(audit) == row["turns"]):
        raise ValueError(f"turn count mismatch for {row['scenario_id']}")
    turns = []
    for decision, audited in zip(decisions, audit, strict=True):
        payload = decision["decision"]
        text, flag = parse_response(decision["response"])
        turns.append({
            "decision_id": payload["decision_id"],
            "tick": payload["tick"],
            "images": [export_image(uri, output) for uri in payload["observation"]["images"]],
            "assistant_text": text,
            "flag": flag,
            "raw": decision["response"],
            "prompt": payload["observation"]["user"],
            "prompt_hash": audited["metadata"]["prompt_hash"],
        })
    keep_report = {
        key: report.get(key)
        for key in (
            "task", "episode", "instruction", "persona", "success", "goal_ok", "f1",
            "precision", "recall", "cc", "dt", "false_flags", "fallback_repairs",
            "final_tick", "n_talk", "n_polls", "prevented", "beats", "timeline",
            "human_mode", "human_model", "grader",
        )
    }
    return {"spec": spec, "report": keep_report, "turns": turns}


def category(before: dict, after: dict) -> str:
    if after["success"] > before["success"]:
        return "improved"
    if after["success"] < before["success"]:
        return "regressed"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--before", default="update-0")
    parser.add_argument("--after", default="update-12-best")
    args = parser.parse_args()
    (args.output / "images").mkdir(parents=True, exist_ok=True)
    (args.output / "pairs").mkdir(parents=True, exist_ok=True)
    before_root, before_rows = load_result(args.eval_root, args.before)
    after_root, after_rows = load_result(args.eval_root, args.after)
    if before_rows.keys() != after_rows.keys():
        raise ValueError("checkpoint evaluations contain different scenarios")

    index = []
    for scenario_id in sorted(before_rows):
        before_row, after_row = before_rows[scenario_id], after_rows[scenario_id]
        if before_row["group_key"] != after_row["group_key"] or before_row["split"] != after_row["split"]:
            raise ValueError(f"mismatched pair: {scenario_id}")
        before = export_episode(before_root, before_row, args.output)
        after = export_episode(after_root, after_row, args.output)
        if before["spec"] != after["spec"]:
            raise ValueError(f"episode spec changed across checkpoints: {scenario_id}")
        if before["turns"][0]["prompt_hash"] != after["turns"][0]["prompt_hash"]:
            raise ValueError(f"initial policy observation changed: {scenario_id}")
        pair_file = f"pairs/{len(index):02d}.json"
        (args.output / pair_file).write_text(json.dumps({"before": before, "after": after}, ensure_ascii=False))
        br, ar = before["report"], after["report"]
        curated = CURATED.get(scenario_id)
        index.append({
            "file": pair_file,
            "scenario": scenario_id,
            "task": before_row["task_id"],
            "split": before_row["split"],
            "category": category(br, ar),
            "before_success": br["success"],
            "after_success": ar["success"],
            "before_f1": br["f1"],
            "after_f1": ar["f1"],
            "before_false_flags": br["false_flags"],
            "after_false_flags": ar["false_flags"],
            "before_goal_ok": br["goal_ok"],
            "after_goal_ok": ar["goal_ok"],
            "before_prevented": br["prevented"],
            "after_prevented": ar["prevented"],
            "interesting": curated is not None,
            "highlight_title": curated["title"] if curated else None,
            "highlight_summary": curated["summary"] if curated else None,
            "highlight_rank": curated["rank"] if curated else 999,
        })

    counts = {name: sum(row["category"] == name for row in index) for name in ("improved", "regressed", "neutral")}
    (args.output / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    shutil.copyfile(Path(__file__).with_name("full_suite_replay.html"), args.output / "index.html")
    audit = {
        "pairs": len(index),
        "categories": counts,
        "splits": {split: sum(row["split"] == split for row in index) for split in ("train", "validation")},
        "curated": len(CURATED),
        "source": str(args.eval_root.resolve()),
        "before": args.before,
        "after": args.after,
    }
    (args.output / "build-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **audit}))


if __name__ == "__main__":
    main()
