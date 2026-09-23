"""Shared retry policy for isolated interactive episode failures."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import traceback


async def generate_with_retries(
    args,
    sample,
    sampling_params,
    *,
    evaluation: bool,
    attempts_env: str,
    default_attempts: int,
    failure_file: str,
    require_evaluation: bool = False,
):
    """Retry one episode and retain an audit record for every failed attempt."""
    if require_evaluation and not evaluation:
        raise ValueError("this retry wrapper is evaluation-only")

    attempts = int(os.environ.get(attempts_env, str(default_attempts)))
    if attempts < 1:
        raise ValueError(f"{attempts_env} must be at least 1")
    job_dir_value = os.environ.get("INTERACT_JOB_DIR") or os.environ.get("Q35_TRAIN_DIR")
    if not job_dir_value:
        raise ValueError("INTERACT_JOB_DIR or Q35_TRAIN_DIR must be set for retry auditing")
    job_dir = Path(job_dir_value)
    job_dir.mkdir(parents=True, exist_ok=True)

    from interact_env.slime_bridge.generate import generate as native_generate

    for attempt in range(1, attempts + 1):
        try:
            return await native_generate(
                args, sample, sampling_params, evaluation=evaluation
            )
        except Exception as exc:
            record = {
                "sample_index": sample.index,
                "group_index": sample.group_index,
                "evaluation": evaluation,
                "attempt": attempt,
                "attempts": attempts,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            with (job_dir / failure_file).open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if attempt == attempts:
                raise
            await asyncio.sleep(5 * attempt)
