"""Clone a frozen ScreenSim RL split for a revision-pinned Gemini human run."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def build_manifest(source, source_raw, engine_root, human_model, persona="baseline"):
    manifest = deepcopy(source)
    revision = subprocess.check_output(
        ["git", "-C", str(engine_root), "rev-parse", "HEAD"], text=True
    ).strip()
    sys.path.insert(0, str(engine_root.resolve()))
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.personas import PERSONAS
    from screensim.tasks import by_id

    if persona not in PERSONAS:
        raise ValueError(f"unknown ScreenSim persona: {persona}")

    for scenario in manifest["scenarios"]:
        spec = scenario["spec"]
        _, episodes = design_composite_episodes(by_id(spec["task_id"]), per_task=3)
        episode_index = spec["config"]["episode_index"]
        if episode_index >= len(episodes) or not episodes[episode_index].ok:
            raise ValueError(f"scenario no longer certifies: {scenario['id']}")
        spec["config"].update(
            human="gemini", human_model=human_model, persona=persona, max_turns=160
        )
    counts = {
        "train_scenarios": sum(s["split"] == "train" for s in manifest["scenarios"]),
        "validation_scenarios": sum(
            s["split"] == "validation" for s in manifest["scenarios"]
        ),
    }
    if counts != {"train_scenarios": 18, "validation_scenarios": 12}:
        raise ValueError(f"expected frozen 18/12 RL split, got {counts}")
    persona_tag = "" if persona == "baseline" else f"_{persona}"
    engine_diff = subprocess.check_output(
        ["git", "-C", str(engine_root), "diff", "--binary", "HEAD"]
    )
    manifest.update(
        split_version=f"screensim_rl_v3_gemini37flash{persona_tag}_18train_12val_engine{revision[:7]}",
        source_manifest_sha256=hashlib.sha256(source_raw).hexdigest(),
        engine_revision=revision,
        engine_worktree_diff_sha256=(hashlib.sha256(engine_diff).hexdigest() if engine_diff else None),
        profile=f"plan_human_v3:{human_model}",
        persona=persona,
        human_mode="free",
        grader="person",
        counts=counts,
        evaluation_note=(
            "Development validation with a stochastic API-backed human. Keep separate "
            "from scripted-human curves; engine semantics and human distribution differ."
        ),
    )
    return manifest


def prompt_rows(manifest, split):
    return [
        {
            "prompt": [{"role": "user", "content": "Native assistant episode"}],
            "metadata": {
                "episode": scenario["spec"],
                "scenario_id": scenario["id"],
                "split": split,
                "split_version": manifest["split_version"],
            },
        }
        for scenario in manifest["scenarios"]
        if scenario["split"] == split
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--human-model", default="gemini-3.7-flash")
    parser.add_argument("--persona", default="baseline")
    args = parser.parse_args()
    source_raw = args.source.read_bytes()
    manifest = build_manifest(
        json.loads(source_raw), source_raw, args.engine_root, args.human_model, args.persona
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest["source_manifest"] = str(args.source.resolve())
    payload = json.dumps(manifest, indent=2) + "\n"
    (args.output_dir / "manifest.json").write_text(payload)
    for split in ("train", "validation"):
        (args.output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in prompt_rows(manifest, split))
        )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "engine_revision": manifest["engine_revision"],
                **manifest["counts"],
                "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
