"""Full-suite entry point for the shared evaluation-only retry policy."""

from examples.interact.common.retry_generate import generate_with_retries


async def generate(args, sample, sampling_params, evaluation=False):
    return await generate_with_retries(
        args,
        sample,
        sampling_params,
        evaluation=evaluation,
        attempts_env="FULL_SUITE_EPISODE_ATTEMPTS",
        default_attempts=3,
        failure_file="episode_failures.jsonl",
        require_evaluation=True,
    )
