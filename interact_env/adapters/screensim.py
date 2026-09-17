"""Native ScreenSim v2 assistant adapter; initial deterministic-human profile."""
import base64
from contextlib import nullcontext
import json
from pathlib import Path
import random
import re
import sys

from ..protocol import EpisodeResult, Observation
from .screensim_rewards import VERSIONS, score


class ScriptedDialogueHuman:
    """Test human: native scripted hands; accept only native-verified corrections.

    This is NOT the Gemini human used in the published/local baseline sweep.
    Privileged engine_hint stays inside this environment human, never the policy.
    """
    def turn(self, *, engine_hint="unspotted", **kwargs):
        return {"utterance": "", "move": "accept" if engine_hint == "correct" else "ignore",
                "close_conversation": True, "plan_update": {"op": "drop", "actions": []}}


def run_native(spec, client=None, control=None):
    config = dict(spec.config)
    root = Path(config.pop("engine_root", Path(__file__).resolve().parents[3] / "screensim-engine"))
    human = config.pop("human", "scripted")
    observation = config.pop("observation", "frames")
    persona_name = config.pop("persona", "baseline")
    episode_index = config.pop("episode_index", 0)
    k_frames = config.pop("k_frames", 2)
    max_turns = config.pop("max_turns", 500)
    reward_version = config.pop("reward_version", "screensim_f1_v1")
    if config:
        raise ValueError(f"unknown ScreenSim configuration keys: {sorted(config)}")
    if human != "scripted":
        raise ValueError("initial ScreenSim adapter supports scripted test human only")
    if observation not in ("frames", "relational") or reward_version not in VERSIONS:
        raise ValueError("unsupported ScreenSim observation/reward profile")
    sys.path.insert(0, str(root.resolve()))
    random.seed(spec.seed)
    from screensim.tasks import by_id
    from screensim.interact.composite import design_composite_episodes
    from screensim.interact.composite_run_v2 import run_composite_v2, SilentV2, OracleV2
    from screensim.interact.agents import AssistantV2
    from screensim.interact.personas import PERSONAS
    from screensim.interact.manual import manual_context
    from screensim.render.capture import Capturer
    task = by_id(spec.task_id)
    ir, episodes = design_composite_episodes(task, per_task=3)
    episode = episodes[episode_index]
    if not episode.ok:
        raise ValueError("uncertified ScreenSim composite")
    persona = PERSONAS[persona_name]
    if control == "silent":
        assistant = SilentV2()
    elif control == "oracle":
        assistant = OracleV2()
    else:
        assistant = AssistantV2(persona, client, use_relational=observation == "relational",
                                task_desc=task.instruction, manual=manual_context(task.id))
    with Capturer(scale=1.0) if observation == "frames" else nullcontext() as cap:
        report = run_composite_v2(episode, ir, persona, assistant, ScriptedDialogueHuman(),
                                  frame_fn=cap.snap if cap else None, k_frames=k_frames,
                                  max_turns=max_turns, contestant=control or "external-policy", human_mode="scripted")
    report.pop("_film", None)
    report["adapter_profile"] = "scripted_human_v1"
    report["reward_version"] = reward_version
    return report


class ScreenSimAdapter:
    def run(self, spec, session):
        class Client:
            calls = 0
            empty = 0
            failures = 0

            def text(self, prompt, images=None, **kwargs):
                self.calls += 1
                match = re.search(r"CURRENT TIME \(tick (\d+)\)", prompt)
                encoded = tuple("data:image/png;base64," + base64.b64encode(png).decode() for png in (images or []))
                return session.request(Observation("", prompt, encoded, spec.config.get("observation", "frames")),
                                       tick=int(match.group(1)) if match else None,
                                       time_unit="screensim_tick", action_schema="screensim_native_v2")

        report = run_native(spec, Client())
        path = Path("report.json").resolve()
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        version = report["reward_version"]
        reward = score(report, version)
        components = {k: report[k] for k in ("f1", "recall", "precision", "dt", "success", "cc",
                                             "false_flags", "n_fired", "n_designed", "n_polls")}
        components.update(in_time_success=report["success"], goal_ok=report["goal_ok"],
                          final_tick=report["final_tick"])
        return EpisodeResult(session.episode_id, "goal_met" if report["goal_ok"] else "goal_unmet",
                             True, False, True, reward, version, components,
                             {"native_report": str(path)})
