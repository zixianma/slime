import asyncio
import json
from pathlib import Path

from interact_env import Action, Decision, Environment, EpisodeSpec
from interact_env.adapters.screensim import run_native

SILENT = '{"text":"","flag":null}'


def test_native_screensim_controls_and_prompt_parity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = EpisodeSpec("screensim", "lost_phone_lockdown", config={"observation": "relational"})
    assert run_native(spec, control="silent")["f1"] == 0
    assert run_native(spec, control="oracle")["f1"] == 1

    class Client:
        calls = empty = failures = 0

        def __init__(self):
            self.prompts = []

        def text(self, prompt, images=None, **kwargs):
            assert not images
            self.prompts.append(prompt)
            return SILENT

    direct = Client()
    expected = run_native(spec, direct)

    async def rollout():
        env = Environment(spec, tmp_path / "episodes")
        prompts = []
        try:
            msg = await env.reset()
            while isinstance(msg, Decision):
                assert msg.observation.system == ""
                assert msg.action_schema == "screensim_native_v2"
                prompts.append(msg.observation.user)
                msg = await env.step(Action(env.episode_id, msg.decision_id, SILENT))
            assert msg.eligible and msg.reward == expected["f1"]
            return prompts, json.loads(Path(msg.artifacts["native_report"]).read_text())
        finally:
            await env.close()

    prompts, actual = asyncio.run(rollout())
    assert prompts == direct.prompts
    for key in ("f1", "recall", "precision", "n_fired", "n_polls", "final_tick", "goal_ok"):
        assert actual[key] == expected[key], key


def test_native_intime_success_profile(tmp_path, monkeypatch):
    from interact_env.adapters.screensim_rewards import score
    monkeypatch.chdir(tmp_path)
    spec = EpisodeSpec("screensim", "lost_phone_lockdown", config={
        "observation": "relational", "reward_version": "screensim_intime_success_v1"})
    silent = run_native(spec, control="silent")
    oracle = run_native(spec, control="oracle")
    assert score(silent, spec.config["reward_version"]) == 0
    assert score(oracle, spec.config["reward_version"]) == 1


def test_native_gemini_human_uses_free_v3_person_grader_without_api(tmp_path, monkeypatch):
    import screensim.interact.agents as agents

    class FakeHumanClient:
        def text(self, *args, **kwargs):
            return json.dumps({"utterance": "", "move": "answer", "close_conversation": True,
                               "done": True, "plan_update": {"op": "drop", "actions": []},
                               "assistant_error_call": "none", "call_about": "", "think": ""})

    monkeypatch.setattr(agents, "make_client", lambda model: FakeHumanClient())
    monkeypatch.chdir(tmp_path)
    report = run_native(EpisodeSpec("screensim", "lost_phone_lockdown", config={
        "human": "gemini", "human_model": "gemini-3.7-flash",
        "observation": "relational", "max_turns": 2}), control="silent")
    assert report["adapter_profile"] == "plan_human_v3:gemini-3.7-flash"
    assert report["human_model"] == "gemini-3.7-flash"
    assert report["human_mode"] == "free"
    assert report["grader"] == "person"
