"""Straggler-tolerant CookSim rollout: launch five, train on first four per group."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import time

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.async_utils import run
from slime.utils.types import Sample


ACCEPTED_PER_GROUP = 4
LAUNCHED_PER_GROUP = 5


def _policy_version(turns: list[Sample]) -> str:
    versions = {str(version) for turn in turns for version in turn.weight_versions}
    if len(versions) != 1:
        raise RuntimeError(f"trajectory has missing or mixed policy versions: {sorted(versions)}")
    return next(iter(versions))


async def _first_completed_per_group(tasks, *, groups, accepted):
    """Collect complete attempts and cancel one trailing attempt per group."""
    pending = set(tasks)
    cancelled = set()
    kept = {group: [] for group in range(groups)}
    rows = []
    try:
        while any(len(items) < accepted for items in kept.values()):
            if not pending:
                raise RuntimeError("speculative rollout exhausted attempts before every group completed")
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            finished = []
            for task in done:
                group, slot, started = tasks[task]
                try:
                    ended, turns = task.result()
                    if not isinstance(turns, list) or not turns or not all(isinstance(x, Sample) for x in turns):
                        raise TypeError("cooking attempt must return a non-empty list[Sample]")
                    if any(turn.status == Sample.Status.ABORTED for turn in turns):
                        raise RuntimeError("cooking attempt returned an aborted trajectory")
                    finished.append((ended, group, slot, started, turns))
                except Exception as exc:
                    rows.append({"group": group, "slot": slot, "status": "failed",
                                 "seconds": time.monotonic() - started,
                                 "error": f"{type(exc).__name__}: {exc}"})
            for ended, group, slot, started, turns in sorted(finished, key=lambda x: (x[0], x[2])):
                if len(kept[group]) < accepted:
                    kept[group].append((slot, turns))
                    rows.append({"group": group, "slot": slot, "status": "accepted",
                                 "seconds": ended - started, "turns": len(turns)})
                else:
                    rows.append({"group": group, "slot": slot, "status": "discarded_complete",
                                 "seconds": ended - started, "turns": len(turns)})
                if len(kept[group]) == accepted:
                    for task in list(pending):
                        other_group, other_slot, other_started = tasks[task]
                        if other_group == group:
                            task.cancel()
                            pending.remove(task)
                            cancelled.add(task)
                            rows.append({"group": group, "slot": other_slot,
                                         "status": "cancelled_straggler",
                                         "seconds": time.monotonic() - other_started})
            # A failed attempt is tolerable only while enough attempts remain.
            for group in range(groups):
                possible = len(kept[group]) + sum(tasks[t][0] == group for t in pending)
                if possible < accepted:
                    raise RuntimeError(f"speculative group {group} cannot reach {accepted} completions")
        return kept, rows
    finally:
        for task in pending:
            task.cancel()
        # Await every cancelled trajectory so its environment ``close`` block
        # completes before Slime hands the shared GPUs back to the learner.
        await asyncio.gather(*(pending | cancelled), return_exceptions=True)


async def _timestamped(coro):
    """Preserve actual completion order across one asyncio.wait wake-up."""
    value = await coro
    return time.monotonic(), value


async def _drain_policy_servers(args):
    """Abort any decode orphaned by an HTTP-task cancellation and await idle."""
    from slime.backends.sglang_utils.server_control import abort_servers_until_idle
    from slime.rollout.sglang_rollout import get_model_url
    from slime.utils.http_utils import get

    response = await get(get_model_url(args, "policy", "/workers"))
    urls = [worker["url"] for worker in response["workers"]]
    if not urls:
        raise RuntimeError("policy router returned no workers during speculative drain")
    await abort_servers_until_idle(urls)


async def _generate(args, rollout_id, data_source):
    from slime.rollout.sglang_rollout import GenerateState, generate_and_rm

    if args.n_samples_per_prompt != ACCEPTED_PER_GROUP:
        raise ValueError("speculative cooking rollout requires GRPO group size four")
    get_samples = data_source.get_samples if hasattr(data_source, "get_samples") else data_source
    groups = get_samples(args.rollout_batch_size)
    if len(groups) != args.rollout_batch_size or any(len(group) != ACCEPTED_PER_GROUP for group in groups):
        raise RuntimeError("data source did not return the requested four-attempt groups")

    state = GenerateState(args)
    state.reset()
    task_info = {}
    started = time.monotonic()
    try:
        synthetic_base = 1_000_000_000 + int(rollout_id) * 10_000
        for group_pos, group in enumerate(groups):
            attempts = list(group)
            spare = copy.deepcopy(group[-1])
            spare.index = synthetic_base + group_pos
            spare.rollout_id = synthetic_base + group_pos
            spare.session_id = None
            attempts.append(spare)
            for slot, sample in enumerate(attempts):
                params = state.sampling_params.copy()
                if getattr(args, "sglang_enable_deterministic_inference", False):
                    params["sampling_seed"] = args.rollout_seed + slot
                started_attempt = time.monotonic()
                task = asyncio.create_task(_timestamped(
                    generate_and_rm(args, sample, params, evaluation=False)
                ))
                task_info[task] = (group_pos, slot, started_attempt)

        kept, rows = await _first_completed_per_group(
            task_info, groups=len(groups), accepted=ACCEPTED_PER_GROUP
        )
        if any(row["status"] == "cancelled_straggler" for row in rows):
            await _drain_policy_servers(args)
        output = []
        group_versions = []
        for group_pos in range(len(groups)):
            attempts = sorted(kept[group_pos], key=lambda item: item[0])
            versions = {_policy_version(turns) for _, turns in attempts}
            if len(versions) != 1:
                raise RuntimeError(f"speculative group {group_pos} mixed policy versions: {sorted(versions)}")
            group_versions.append(next(iter(versions)))
            output.append([turns for _, turns in attempts])
        if len(set(group_versions)) != 1:
            raise RuntimeError(f"rollout mixed policy versions across groups: {group_versions}")

        elapsed = time.monotonic() - started
        receipt = {
            "version": "cooking_speculative_5_to_4_v1",
            "rollout_id": rollout_id,
            "groups": len(groups),
            "launched_per_group": LAUNCHED_PER_GROUP,
            "accepted_per_group": ACCEPTED_PER_GROUP,
            "accepted_episodes": len(groups) * ACCEPTED_PER_GROUP,
            "policy_version": group_versions[0],
            "elapsed_s": elapsed,
            "attempts": rows,
        }
        audit = Path(os.environ["INTERACT_JOB_DIR"])
        (audit / f"speculative-rollout-{rollout_id}.json").write_text(json.dumps(receipt, indent=2) + "\n")
        cancelled_count = sum(row["status"] == "cancelled_straggler" for row in rows)
        discarded = sum(row["status"] == "discarded_complete" for row in rows)
        accepted_seconds = [row["seconds"] for row in rows if row["status"] == "accepted"]
        return RolloutFnTrainOutput(samples=output, metrics={
            "rollout/speculative/launched": len(groups) * LAUNCHED_PER_GROUP,
            "rollout/speculative/accepted": len(groups) * ACCEPTED_PER_GROUP,
            "rollout/speculative/cancelled": cancelled_count,
            "rollout/speculative/discarded_complete": discarded,
            "rollout/speculative/accepted_mean_s": sum(accepted_seconds) / len(accepted_seconds),
            "rollout/speculative/accepted_max_s": max(accepted_seconds),
            "rollout/speculative/elapsed_s": elapsed,
        })
    finally:
        for task in task_info:
            if not task.done():
                task.cancel()
        await asyncio.gather(*task_info, return_exceptions=True)
        state.reset()


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        raise ValueError("speculative collector is training-only; use the exact eval function")
    return run(_generate(args, rollout_id, data_source))
