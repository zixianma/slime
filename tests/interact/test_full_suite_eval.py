import asyncio
import json
from types import SimpleNamespace

import pytest

from examples.interact.evaluation import generate as evaluation_generate
from examples.interact.evaluation.checkpoints import resolve
from examples.interact.evaluation.metrics import summarize
from interact_env.protocol import EpisodeSpec


def sample(episode_id, spec, reward, success, turns=2):
    result = []
    for turn in range(turns):
        result.append(SimpleNamespace(
            reward=reward,
            metadata={
                "episode_id": episode_id,
                "engine": spec.engine,
                "task_id": spec.task_id,
                "group_key": spec.group_key,
                "turn_index": turn,
                "num_turns": turns,
                "reward_version": "test-v1",
                "reward_components": {"success": success},
                "outcome": "won" if success else "timeout",
            },
        ))
    return result


def test_summarize_requires_one_complete_episode_per_scenario(tmp_path):
    specs = [
        EpisodeSpec("screensim", "a", config={"episode_index": 0}),
        EpisodeSpec("screensim", "b", config={"episode_index": 0}),
    ]
    dataset = tmp_path / "all.jsonl"
    rows = [
        {"prompt": [], "metadata": {"episode": vars(spec), "split": split, "scenario_id": spec.task_id}}
        for spec, split in zip(specs, ("train", "validation"))
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    samples = sample("e1", specs[0], 1.0, 1.0) + sample("e2", specs[1], 0.0, 0.0)

    values, records = summarize(samples, engine="screensim", dataset=dataset, expected=2)
    assert values["eval/success"] == 0.5
    assert values["eval/train/success"] == 1.0
    assert values["eval/validation/success"] == 0.0
    assert len(records) == 2

    with pytest.raises(ValueError, match="incomplete full-suite pass"):
        summarize(samples[:2], engine="screensim", dataset=dataset, expected=2)


def test_full_suite_retry_is_evaluation_only():
    sample_value = SimpleNamespace(index=0, group_index=0)
    with pytest.raises(ValueError, match="evaluation-only"):
        asyncio.run(evaluation_generate.generate(None, sample_value, {}, evaluation=False))


def test_checkpoint_spec_resolves_literal_and_environment_paths():
    spec = {
        "version": 1,
        "checkpoints": [
            {"label": "update-0", "update": 0, "checkpoint_index": -1, "load_env": "MODEL"},
            {"label": "update-12", "update": 12, "checkpoint_index": 11, "load": "/checkpoints"},
        ],
    }
    assert resolve(spec, 0, {"MODEL": "/model"})["load"] == "/model"
    assert resolve(spec, 1, {}) == {
        "label": "update-12",
        "update": 12,
        "checkpoint_index": 11,
        "load": "/checkpoints",
    }

    spec["checkpoints"][1]["update"] = 11
    with pytest.raises(ValueError, match="checkpoint_index"):
        resolve(spec, 1, {})
