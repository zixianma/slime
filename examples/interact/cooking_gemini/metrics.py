"""Episode-weighted metrics for single-error CookSim RL with a Gemini user."""

from collections import defaultdict
import json
import os
from pathlib import Path

from interact_env.slime_bridge.metrics import episode_metrics


VERSION = "cooking_prevention_turns_v1"


def summarize(samples, prefix):
    episode_metrics(samples)
    episodes = {sample.metadata["episode_id"]: sample for sample in samples}
    groups, tasks = defaultdict(list), defaultdict(list)
    for sample in episodes.values():
        meta = sample.metadata
        assert meta["engine"] == "cooking" and meta["reward_version"] == VERSION
        components = meta["reward_components"]
        capped = components.get("decision_cap_reached", 0)
        # Completed native reports must contain the frozen split's one planned
        # error. A policy-horizon timeout has no final native report and is the
        # sole valid zero-planned exception.
        assert (components["errors_planned"] == 1 or
                (capped == 1 and meta["outcome"] == "timeout" and
                 components["errors_planned"] == 0))
        groups[sample.group_index].append(sample.reward)
        tasks[meta["task_id"]].append(sample.reward)
    n = len(episodes)
    values = {
        f"{prefix}/episodes": n,
        f"{prefix}/reward": sum(x.reward for x in episodes.values()) / n,
        f"{prefix}/scenario_macro_reward": sum(sum(v) / len(v) for v in tasks.values()) / len(tasks),
        f"{prefix}/success": sum(x.metadata["reward_components"]["task_success"] for x in episodes.values()) / n,
        f"{prefix}/prevention": sum(
            x.metadata["reward_components"].get("prevented_errors", 0)
            for x in episodes.values()) / n,
        f"{prefix}/false_flags": sum(x.metadata["reward_components"]["false_flags"] for x in episodes.values()) / n,
        f"{prefix}/turns": sum(x.metadata["num_turns"] for x in episodes.values()) / n,
        f"{prefix}/invalid_response_fraction": sum(x.metadata["reward_components"]["invalid_response_fraction"] for x in episodes.values()) / n,
        f"{prefix}/decision_cap_fraction": sum(
            x.metadata["reward_components"].get("decision_cap_reached", 0)
            for x in episodes.values()) / n,
    }
    for outcome in ("won", "wrong_serve", "burned", "timeout"):
        values[f"{prefix}/outcome/{outcome}"] = sum(
            x.metadata["outcome"] == outcome for x in episodes.values()) / n
    if prefix == "train":
        values["train/groups"] = len(groups)
        values["train/reward_mixed_group_fraction"] = sum(
            len({round(v, 8) for v in group}) > 1 for group in groups.values()) / len(groups)
        values["train/prevention_mixed_group_fraction"] = sum(
            len({x.metadata["reward_components"].get("prevented_errors", 0)
                 for x in episodes.values() if x.group_index == group}) > 1
            for group in groups) / len(groups)
    return values


def record(values):
    path = Path(os.environ["INTERACT_JOB_DIR"]) / "gemini_episode_metrics.jsonl"
    with path.open("a") as out:
        out.write(json.dumps(values) + "\n")
    print("COOKING_GEMINI_METRICS " + json.dumps(values), flush=True)


def log_rollout(rollout_id, args, samples, *unused):
    from slime.observability.logging_utils import log
    values = summarize(samples, "train") | episode_metrics(samples)
    assert values["train/episodes"] == 24 and values["train/groups"] == 6
    values.update({"train/step": rollout_id + 1, "train/rollout_policy_update": rollout_id})
    if os.environ.get("COOKING_BIND_RENDER_WORKERS") == "1":
        from examples.interact.cooking_gpu_handoff import assert_no_graphics
        handoff = assert_no_graphics()
        (Path(os.environ["INTERACT_JOB_DIR"]) / f"renderer-handoff-{rollout_id + 1}.json").write_text(
            json.dumps(handoff) + "\n")
    record(values)
    log(args, values, step_key="train/step")
    if values["train/reward_mixed_group_fraction"] == 0:
        raise RuntimeError("COOKING_GEMINI_NO_REWARD_VARIANCE")
    return False


def log_eval(completed_updates, args, data, extra):
    from slime.observability.logging_utils import log
    samples = [sample for dataset in data.values() for sample in dataset["samples"]]
    values = summarize(samples, "eval")
    groups = defaultdict(set)
    for sample in samples:
        groups[sample.metadata["group_key"]].add(sample.metadata["episode_id"])
    assert len(groups) == 10 and {len(x) for x in groups.values()} == {2}
    assert values["eval/episodes"] == 20
    values["eval/step"] = completed_updates
    record(values)
    log(args, values, step_key="eval/step")
    return True
