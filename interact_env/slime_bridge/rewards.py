"""Normalize once per trajectory within a homogeneous prompt group.

Use --custom-reward-post-process-path; no Slime core modifications.
Upstream rollout_mask_sums handles token-weighted loss within each rollout.
"""
from collections import defaultdict
import math


def normalize(args, samples):
    groups = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        if sample.group_index is None or sample.rollout_id is None:
            raise ValueError("group_index and rollout_id are required")
        groups[sample.group_index][sample.rollout_id].append(sample)
    raw = [s.get_reward_value(args) for s in samples]
    normalized = {}
    for trajectories in groups.values():
        keys = {(s.metadata["group_key"], s.metadata["reward_version"])
                for turns in trajectories.values() for s in turns}
        if len(keys) != 1:
            raise ValueError("mixed engine/task/human/timing/reward configurations within a GRPO group")
        rewards = []
        for turns in trajectories.values():
            values = {s.get_reward_value(args) for s in turns}
            if len(values) != 1 or not math.isfinite(next(iter(values))):
                raise ValueError("every turn must carry the same finite terminal episode reward")
            if sorted(s.metadata["turn_index"] for s in turns) != list(range(len(turns))):
                raise ValueError("missing or duplicated episode turns")
            if any(s.metadata["num_turns"] != len(turns) for s in turns):
                raise ValueError("incomplete episode reached reward normalization")
            rewards.append(next(iter(values)))
        values = rewards
        if getattr(args, "rewards_normalization", True):
            if args.advantage_estimator != "grpo":
                raise ValueError("this initial normalization hook is validated for GRPO only")
            mean = sum(rewards) / len(rewards)
            values = [r - mean for r in rewards]
            if getattr(args, "grpo_std_normalization", True) and len(values) > 1:
                std = math.sqrt(sum(v * v for v in values) / (len(values) - 1))
                values = [v / (std + 1e-6) for v in values]
        for turns, value in zip(trajectories.values(), values, strict=True):
            for s in turns:
                normalized[id(s)] = value
    return raw, [normalized[id(s)] for s in samples]
