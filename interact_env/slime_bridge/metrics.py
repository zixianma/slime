"""Scalar, episode-weighted metrics for any native adapter. No prompts/images."""
from collections import defaultdict
import math


def episode_metrics(samples):
    episodes = defaultdict(list)
    for sample in samples:
        episodes[sample.metadata["episode_id"]].append(sample)
    by_engine = defaultdict(list)
    for turns in episodes.values():
        first = turns[0]
        meta = first.metadata
        if sorted(t.metadata["turn_index"] for t in turns) != list(range(meta["num_turns"])):
            raise ValueError("cannot log an incomplete or duplicated trajectory as an episode")
        if any(t.metadata["num_turns"] != len(turns) or t.reward != first.reward
               or t.metadata["reward_components"] != meta["reward_components"] for t in turns):
            raise ValueError("inconsistent terminal episode metrics")
        components = dict(meta["reward_components"])
        values = {"reward": first.reward, "turns": len(turns), **components}
        by_engine[(meta["engine"], meta["reward_version"])].append(values)
    metrics = {"rollout/episodes/count": len(episodes)}
    for (engine, version), rows in by_engine.items():
        prefix = f"rollout/episodes/{engine}/{version}"
        metrics[f"{prefix}/count"] = len(rows)
        for key in sorted(set().union(*(row.keys() for row in rows))):
            values = [float(row[key]) for row in rows if isinstance(row.get(key), (float, int))
                      and math.isfinite(row[key])]
            if values:
                metrics[f"{prefix}/{key}"] = sum(values) / len(values)
                if len(values) != len(rows):
                    metrics[f"{prefix}/{key}_count"] = len(values)
    return metrics


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    from slime.observability import logging_utils
    from slime.observability.metric_utils import compute_rollout_step
    metrics = episode_metrics(samples)
    metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, metrics, step_key="rollout/step")
    return False  # Preserve upstream loss/performance/rollout logging as well.
