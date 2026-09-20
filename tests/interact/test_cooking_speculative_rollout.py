import asyncio
from types import SimpleNamespace

import pytest

from examples.interact.cooking_gemini.speculative_rollout import (
    _generate,
    _first_completed_per_group,
    _policy_version,
    _timestamped,
)
from slime.utils.types import Sample


def completed(version="7"):
    return [Sample(status=Sample.Status.COMPLETED, weight_versions=[version])]


def test_accepts_first_results_per_group_and_awaits_cancelled_cleanup():
    async def exercise():
        cleaned = set()

        async def attempt(name, delay):
            try:
                await asyncio.sleep(delay)
                return completed()
            finally:
                cleaned.add(name)

        task_info = {}
        delays = ((0.002, 0.010, 0.200), (0.003, 0.012, 0.200))
        loop = asyncio.get_running_loop()
        for group, group_delays in enumerate(delays):
            for slot, delay in enumerate(group_delays):
                task = asyncio.create_task(_timestamped(attempt((group, slot), delay)))
                task_info[task] = (group, slot, loop.time())

        kept, rows = await _first_completed_per_group(task_info, groups=2, accepted=2)
        assert [{slot for slot, _ in kept[group]} for group in range(2)] == [{0, 1}, {0, 1}]
        assert sum(row["status"] == "cancelled_straggler" for row in rows) == 2
        assert cleaned == {(group, slot) for group in range(2) for slot in range(3)}

    asyncio.run(exercise())


def test_one_failed_spare_is_tolerated_when_four_attempts_complete():
    async def exercise():
        async def attempt(slot):
            await asyncio.sleep(slot / 1000)
            if slot == 0:
                raise RuntimeError("probe failure")
            return completed()

        loop = asyncio.get_running_loop()
        task_info = {}
        for slot in range(5):
            task = asyncio.create_task(_timestamped(attempt(slot)))
            task_info[task] = (0, slot, loop.time())
        kept, rows = await _first_completed_per_group(task_info, groups=1, accepted=4)
        assert {slot for slot, _ in kept[0]} == {1, 2, 3, 4}
        assert sum(row["status"] == "failed" for row in rows) == 1

    asyncio.run(exercise())


def test_policy_version_requires_one_nonempty_version():
    assert _policy_version(completed("11")) == "11"
    with pytest.raises(RuntimeError, match="missing or mixed"):
        _policy_version([Sample(status=Sample.Status.COMPLETED)])
    with pytest.raises(RuntimeError, match="missing or mixed"):
        _policy_version([
            Sample(status=Sample.Status.COMPLETED, weight_versions=["11"]),
            Sample(status=Sample.Status.COMPLETED, weight_versions=["12"]),
        ])


def test_generate_uses_slime_data_source_and_replaces_straggler(monkeypatch, tmp_path):
    import examples.interact.cooking_gemini.speculative_rollout as speculative
    import slime.rollout.sglang_rollout as stock

    class DataSource:
        def __init__(self):
            self.calls = []

        def get_samples(self, count):
            self.calls.append(count)
            return [
                [Sample(group_index=group, index=group * 4 + slot,
                        rollout_id=group * 4 + slot)
                 for slot in range(4)]
                for group in range(count)
            ]

    class State:
        sampling_params = {"temperature": 0.8}

        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

    state = State()
    drains = []

    async def fake_drain(args):
        drains.append(args)

    async def fake_generate(_args, sample, params, evaluation=False):
        assert not evaluation
        slot = params["sampling_seed"] - 100
        await asyncio.sleep(0.04 if slot == 3 else slot / 1000)
        return [Sample(group_index=sample.group_index, index=sample.index,
                       rollout_id=sample.rollout_id, status=Sample.Status.COMPLETED,
                       weight_versions=["19"])]

    monkeypatch.setattr(stock, "GenerateState", lambda _args: state)
    monkeypatch.setattr(stock, "generate_and_rm", fake_generate)
    monkeypatch.setattr(speculative, "_drain_policy_servers", fake_drain)
    monkeypatch.setenv("INTERACT_JOB_DIR", str(tmp_path))
    args = SimpleNamespace(n_samples_per_prompt=4, rollout_batch_size=2,
                           sglang_enable_deterministic_inference=True,
                           rollout_seed=100)
    source = DataSource()
    output = asyncio.run(_generate(args, 3, source))

    assert source.calls == [2]
    assert [len(group) for group in output.samples] == [4, 4]
    assert all(len({turns[0].rollout_id for turns in group}) == 4
               for group in output.samples)
    # Slot 3 is slow, so the synthetic slot-4 spare must replace it.
    assert all(any(turns[0].index >= 1_000_000_000 for turns in group)
               for group in output.samples)
    assert output.metrics["rollout/speculative/launched"] == 10
    assert output.metrics["rollout/speculative/accepted"] == 8
    assert output.metrics["rollout/speculative/cancelled"] == 2
    assert state.resets == 2
    assert drains == [args]
    assert (tmp_path / "speculative-rollout-3.json").exists()
