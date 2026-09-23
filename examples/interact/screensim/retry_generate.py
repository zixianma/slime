"""ScreenSim entry point for the shared isolated-episode retry policy."""

from examples.interact.common.retry_generate import generate_with_retries


async def generate(args, sample, sampling_params, evaluation=False):
    return await generate_with_retries(
        args,
        sample,
        sampling_params,
        evaluation=evaluation,
        attempts_env="SCREENSIM_EPISODE_ATTEMPTS",
        default_attempts=2,
        failure_file="screensim_episode_failures.jsonl",
    )
