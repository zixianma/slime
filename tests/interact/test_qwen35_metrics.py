from types import SimpleNamespace

import pytest

from examples.interact.models.qwen35.qwen35_metrics import summarize


def test_macro_and_episode_metrics_do_not_weight_turns():
    samples = []
    for episode, (task, reward, turns, group) in enumerate([
        ('a', 0, 1, 0), ('a', 1, 4, 0), ('b', 1, 2, 1),
    ]):
        for turn in range(turns):
            samples.append(SimpleNamespace(reward=reward, group_index=group, metadata={
                'episode_id': str(episode), 'engine': 'screensim', 'task_id': task,
                'turn_index': turn, 'num_turns': turns,
                'reward_version': 'screensim_intime_success_v1',
                'reward_components': {'success': reward},
            }))
    result = summarize(samples, 'train')
    assert result['train/success'] == pytest.approx(2/3)
    assert result['train/task_macro_success'] == .75
    assert result['train/success_mixed_group_fraction'] == .5
    with pytest.raises(ValueError, match='incomplete'):
        summarize(samples[:-1], 'train')
