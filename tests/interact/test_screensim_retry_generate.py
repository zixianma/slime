import asyncio
from types import SimpleNamespace

from examples.interact.common import retry_generate as retry_policy
from examples.interact.screensim import retry_generate


def test_retry_generate_retries_one_episode(monkeypatch, tmp_path):
    calls = []

    async def flaky(args, sample, sampling_params, evaluation=False):
        calls.append(evaluation)
        if len(calls) == 1:
            raise TimeoutError("transient worker timeout")
        return "complete"

    async def no_sleep(*_args):
        return None

    monkeypatch.setenv("Q35_TRAIN_DIR", str(tmp_path))
    monkeypatch.setenv("SCREENSIM_EPISODE_ATTEMPTS", "2")
    monkeypatch.setattr("interact_env.slime_bridge.generate.generate", flaky)
    monkeypatch.setattr(retry_policy.asyncio, "sleep", no_sleep)
    sample = SimpleNamespace(index=3, group_index=1)
    assert asyncio.run(retry_generate.generate(None, sample, {}, evaluation=True)) == "complete"
    assert calls == [True, True]
    assert "transient worker timeout" in (tmp_path / "screensim_episode_failures.jsonl").read_text()
