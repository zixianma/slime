import pytest
from examples.interact.archive.cooking_calibration.cooking_training_plan import experiment, should_evaluate


def test_sparse_validation_uses_completed_update_count():
    assert [i for i in range(13) if should_evaluate(i)] == [0, 3, 6, 9, 12]
    with pytest.raises(ValueError):
        should_evaluate(13)


def test_full_experiment_not_calibration(tmp_path):
    from examples.interact.cooking_gemini.prepare_cooking_rl import prepare

    split = tmp_path / "split"
    prepare(split, extended_budget=True)
    plan = experiment(split)
    assert plan['train_scenarios'] == 40
    assert plan['validation_scenarios'] == 10
    assert plan['global_batch_episodes'] == 48
    assert plan['train_episodes_total'] == 576
    assert plan['eval_episodes_per_round'] == 40
    assert plan['eval_episodes_total'] == 200
    assert plan['checkpoint_every_updates'] == 1
    assert plan['max_running_requests'] == 4
