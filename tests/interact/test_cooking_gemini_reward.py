import json

import pytest
from types import SimpleNamespace

from interact_env.adapters.cooking import CookingSpec
from interact_env.adapters.cooking_rewards import score
from examples.interact.cooking_gemini.metrics import summarize


def report(*, success=False, prevented=False, false_flags=0):
    return {
        "outcome": "won" if success else "wrong_serve",
        "errors_planned": 1,
        "errors_offered": 1,
        "injections": [{"family": "error", "flagged_valid": prevented,
                        "prevented": prevented}],
        "false_positives": [{}] * false_flags,
    }


def test_prevention_turn_reward_preferences_and_bounds():
    version = "cooking_prevention_turns_v1"
    assert score(report(), version, assistant_turns=0)[0] == 0
    assert score(report(prevented=True), version, assistant_turns=0)[0] == pytest.approx(.30)
    assert score(report(prevented=True, false_flags=1), version, assistant_turns=0)[0] == pytest.approx(.28)
    assert score(report(success=True, false_flags=15), version, assistant_turns=200)[0] == pytest.approx(.65)
    assert score(report(success=True, prevented=True), version, assistant_turns=0)[0] == pytest.approx(1.30)


def test_prevention_turn_reward_rejects_wrong_contract():
    version = "cooking_prevention_turns_v1"
    with pytest.raises(ValueError, match="exactly one"):
        score({**report(), "errors_planned": 2}, version, assistant_turns=10)
    with pytest.raises(ValueError, match="assistant_turns"):
        score(report(), version)
    with pytest.raises(ValueError, match="assistant_turns"):
        score(report(), version, assistant_turns=201)


def test_gemini_case_pool_and_model_are_explicit(tmp_path):
    pool = tmp_path / "cases.json"
    pool.write_text(json.dumps({"cases": []}))
    CookingSpec(human="gemini", human_model="gemini-3.7-flash",
                case_pool=str(pool)).validate()
    with pytest.raises(ValueError, match="pinned"):
        CookingSpec(human="gemini", human_model="gemini-flash-latest",
                    case_pool=str(pool)).validate()


def test_metrics_accept_only_explicit_horizon_timeout_without_planned_error():
    metadata = {
        "episode_id": "timeout", "engine": "cooking",
        "reward_version": "cooking_prevention_turns_v1", "task_id": "task",
        "turn_index": 0, "num_turns": 1, "outcome": "timeout",
        "reward_components": {
            "task_success": 0.0, "errors_planned": 0, "errors_offered": 0,
            "detected_errors": 0, "prevented_errors": 0, "false_flags": 0,
            "assistant_calls": 1, "invalid_responses": 0,
            "invalid_response_fraction": 0.0, "native_ticks": 2,
            "decision_cap_reached": 1,
        },
    }
    sample = SimpleNamespace(metadata=metadata, reward=-0.05, group_index=0)
    values = summarize([sample], "eval")
    assert values["eval/decision_cap_fraction"] == 1
    # Historical timeout receipts omitted this zero-valued field; metric
    # aggregation must remain backward compatible with those episodes.
    legacy = {**metadata, "reward_components": {
        key: value for key, value in metadata["reward_components"].items()
        if key != "prevented_errors"
    }}
    assert summarize([SimpleNamespace(metadata=legacy, reward=-0.05,
                                      group_index=0)], "eval")["eval/prevention"] == 0
    with pytest.raises(AssertionError):
        summarize([SimpleNamespace(metadata={
            **metadata, "reward_components": {
                **metadata["reward_components"], "decision_cap_reached": 0}},
            reward=-0.05, group_index=0)], "eval")
