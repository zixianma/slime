import asyncio
import base64
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from interact_env import Action, Decision, Environment, EpisodeResult, EpisodeSpec, Observation
from interact_env.registry import AdapterDefinition, ADAPTERS, resolve

ROOT = Path(__file__).resolve().parents[2]
SILENT = '{"text":"","flag":null}'
CASE = "b3_easy_nops_medium_map_2"
CHECKPOINT = os.environ.get("INTERACT_TEST_CHECKPOINT", "/gpfs/scrubbed/zixianma/checkpoints/web/Qwen2.5-VL-3B-Instruct")


def test_runtime_imports_no_trainer_or_native_engine():
    code = "import sys, interact_env; assert not any(k.startswith(('slime.', 'openwebrl', 'cooksim', 'e2e_stepin_rollout')) for k in sys.modules)"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=15)


def test_worker_timeout_identifies_episode_and_log(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setitem(ADAPTERS, 'fixture', AdapterDefinition(
        'interact_env.testing:FixtureAdapter', ('process',), 'fixture', 'test only'))
    async def stalled_read():
        await asyncio.sleep(10)
    async def run():
        env = Environment(EpisodeSpec('fixture', 'task'), tmp_path, timeout=0.001)
        env.process = SimpleNamespace(stdout=SimpleNamespace(readline=stalled_read))
        with pytest.raises(TimeoutError, match='Native worker response timed out') as error:
            await env._receive()
        assert env.episode_id in str(error.value) and 'worker.log' in str(error.value)
    asyncio.run(run())


def test_catalog_is_honest():
    assert set(ADAPTERS) == {"cooking", "screensim", "vh"}
    assert resolve("cooking").factory
    assert resolve("screensim").factory
    for engine in ("vh",):
        with pytest.raises(NotImplementedError):
            Environment(EpisodeSpec(engine, "example"), "/tmp/unused")


def test_policy_observation_is_an_allowlist():
    with pytest.raises(TypeError):
        Observation("system", "user", hidden_plan="secret")
    with pytest.raises(ValueError):
        Observation("system", "user", images=("/private/image.png",))
    with pytest.raises(ValueError):
        EpisodeResult("id", "infra_failure", False, True, True, 0.0, "bad")
    assert EpisodeSpec("vh", "task").group_key != EpisodeSpec("cooking", "task").group_key


def test_shared_runtime_is_not_cooking_specific(tmp_path, monkeypatch):
    monkeypatch.setitem(ADAPTERS, "fixture", AdapterDefinition(
        "interact_env.testing:FixtureAdapter", ("process",), "fixture", "test only"))

    async def run():
        env = Environment(EpisodeSpec("fixture", "task", config={"hidden_plan": "secret"}), tmp_path)
        try:
            msg = await env.reset()
            ticks = []
            while isinstance(msg, Decision):
                assert "secret" not in json.dumps(asdict(msg))
                assert msg.action_schema == "fixture.free_text" and msg.time_unit == "seconds"
                ticks.append(msg.tick)
                with pytest.raises(ValueError):
                    await env.step(Action("other-episode", msg.decision_id, "oops"))
                msg = await env.step(Action(env.episode_id, msg.decision_id, "arbitrary non-JSON speech"))
            assert ticks == [0, 0.25, 0.5] and msg.outcome == "fixture_finished"
            assert msg.reward == 1.0
        finally:
            await env.close()
        assert env.process.returncode == 0
    asyncio.run(run())


def test_cooking_full_native_parity(tmp_path):
    code = '''
import json, sys
from contextlib import redirect_stdout
from interact_env.adapters.cooking import CookingSpec, run
class Silent:
    def __init__(self): self.prompts=[]
    def sup_call(self, system, user, frames):
        assert not frames
        self.prompts.append([system,user])
        return {"text":"", "flag":None}
s=Silent()
with redirect_stdout(sys.stderr): report=run(CookingSpec(),s)
print(json.dumps({"report":report,"prompts":s.prompts}, default=str))
'''
    direct = tmp_path / "direct"
    direct.mkdir()
    result = subprocess.run([sys.executable, "-c", code], cwd=direct, text=True, capture_output=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=120, check=True)
    expected = json.loads(result.stdout)

    async def rollout():
        env = Environment(EpisodeSpec("cooking", CASE), tmp_path)
        prompts = []
        try:
            msg = await env.reset()
            while isinstance(msg, Decision):
                prompts.append([msg.observation.system, msg.observation.user])
                if len(prompts) == 1:
                    await asyncio.sleep(0.1)
                msg = await env.step(Action(env.episode_id, msg.decision_id, SILENT))
            assert msg.eligible and msg.terminated and not msg.truncated
            return json.loads(Path(msg.artifacts["native_report"]).read_text()), prompts
        finally:
            await env.close()
    report, prompts = asyncio.run(rollout())
    assert prompts == expected["prompts"]
    for key in ("outcome", "ticks", "actions", "flags", "injections", "errors_offered", "plate_seq"):
        assert report[key] == expected["report"][key], key
    assert len(prompts) == 76 and report["ticks"] == 151


def test_concurrent_isolation_and_cancellation(tmp_path):
    async def run():
        envs = [Environment(EpisodeSpec("cooking", CASE, seed=i), tmp_path) for i in range(2)]
        try:
            decisions = await asyncio.gather(*(e.reset() for e in envs))
            assert decisions[0].episode_id != decisions[1].episode_id
            await envs[0].close()
            msg = await envs[1].step(Action(envs[1].episode_id, decisions[1].decision_id, SILENT))
            assert msg.decision_id == 1
        finally:
            await asyncio.gather(*(e.close() for e in envs))
        assert all(e.process.returncode == 0 for e in envs)
    asyncio.run(run())


def test_native_error_is_not_reward(tmp_path):
    async def run():
        env = Environment(EpisodeSpec("cooking", "nonexistent"), tmp_path)
        with pytest.raises(RuntimeError):
            await env.reset()
        assert env.process.returncode is not None
        assert not (env.directory / "report.json").exists()
    asyncio.run(run())


def test_audit_io_failure_bypasses_native_exception_to_silence(monkeypatch):
    from interact_env.worker import Session, TransportFailure
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(asdict(Action("episode", 0, SILENT))) + "\n"))
    def failing_open(*args, **kwargs):
        raise OSError("disk quota exceeded")
    monkeypatch.setattr("builtins.open", failing_open)
    session = Session(lambda *args: None, "episode")
    with pytest.raises(TransportFailure, match="disk quota"):
        try:
            session.request(Observation("", "prompt"), tick=0, time_unit="tick", action_schema="test")
        except Exception:
            pytest.fail("a native exception-to-silence handler must not swallow audit failures")


def test_real_frames_and_official_processor(tmp_path):
    from PIL import Image, ImageStat
    from transformers import AutoProcessor, AutoTokenizer
    from interact_env.slime_bridge.generate import encode_decision

    async def frame():
        env = Environment(EpisodeSpec("cooking", CASE, config={"observation": "frames"}), tmp_path)
        try:
            msg = await env.reset()
            return await env.step(Action(env.episode_id, msg.decision_id, SILENT))
        finally:
            await env.close()
    decision = asyncio.run(frame())
    assert len(decision.observation.images) > 1
    for data in decision.observation.images:
        im = Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1]))).convert("RGB")
        assert min(im.size) > 100 and max(ImageStat.Stat(im).var) > 100
    processor = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    prompt, ids, images, inputs = encode_decision(asdict(decision.observation), tokenizer, processor)
    assert decision.observation.user in prompt and decision.observation.system in prompt
    assert prompt.index(decision.observation.user) < prompt.index("<|vision_start|>")
    grid = inputs["image_grid_thw"]
    assert grid.shape[0] == len(images)
    assert ids.count(tokenizer.convert_tokens_to_ids("<|image_pad|>")) == int(grid.prod(dim=1).sum()) // 4


def test_cooking_reward_and_parser():
    from interact_env.adapters.cooking import parse_reply
    from interact_env.adapters.cooking_rewards import score
    assert parse_reply("<think>private</think>" + SILENT) == {"text": "", "flag": None}
    assert parse_reply('{"text":3}') == {}
    report = dict(outcome="won", errors_planned=2, errors_offered=2, injections=[])
    assert score(report)[0] == 1.0
    assert score(report, "assistant_detection_v1")[0] == 0.5
    with pytest.raises(ValueError):
        score({**report, "aborted": "wall_clock"})
