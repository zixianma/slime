from types import SimpleNamespace

import pytest

from interact_env.slime_bridge.metrics import episode_metrics, log_rollout
from interact_env.adapters.screensim_rewards import score


def samples():
    return [SimpleNamespace(reward=float(episode == 1), metadata={
        "episode_id": str(episode), "engine": "screensim", "reward_version": "screensim_f1_v1",
        "turn_index": turn, "num_turns": count,
        "reward_components": {"f1": float(episode == 1), "in_time_success": float(episode == 1)},
    }) for episode, count in ((0, 1), (1, 3)) for turn in range(count)]


def test_metrics_are_episode_weighted_and_do_not_mix_versions():
    data = samples()
    metrics = episode_metrics(data)
    prefix = "rollout/episodes/screensim/screensim_f1_v1"
    assert metrics["rollout/episodes/count"] == 2
    assert metrics[f"{prefix}/in_time_success"] == 0.5
    assert metrics[f"{prefix}/reward"] == 0.5
    data[0].metadata["reward_version"] = "screensim_intime_success_v1"
    assert episode_metrics(data)[f"{prefix}/count"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        episode_metrics(data[:-1])


def test_native_success_is_not_f1_or_untimed_goal_completion():
    report = {"n_fired": 2, "f1": 0.667, "success": 0, "goal_ok": True}
    assert score(report, "screensim_f1_v1") == 0.667
    assert score(report, "screensim_intime_success_v1") == 0
    assert score({**report, "n_fired": 0, "success": 1}, "screensim_intime_success_v1") == 1
    with pytest.raises(ValueError, match="binary"):
        score({**report, "success": 0.5}, "screensim_intime_success_v1")


def test_custom_hook_preserves_upstream_logging(monkeypatch):
    from slime.observability import logging_utils
    captured = []
    monkeypatch.setattr(logging_utils, "log", lambda args, metrics, step_key: captured.append(metrics))
    args = SimpleNamespace(wandb_always_use_train_step=False)
    assert log_rollout(5, args, samples(), {}, 1.0) is False
    assert captured[0]["rollout/step"] == 5
