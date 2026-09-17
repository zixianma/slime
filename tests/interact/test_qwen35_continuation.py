from types import SimpleNamespace

import pytest

from examples.interact import qwen35_continue_eval as schedule


@pytest.mark.parametrize('steps,completed', [({6,9},3),({6,9},6),({6,9},9),
                                         ({12,15},9),({12,15},12),({12,15},15)])
def test_only_requested_validation_steps_collect_and_log(monkeypatch, steps, completed):
    monkeypatch.setattr(schedule, 'STEPS', steps)
    calls = []
    sentinel = object()
    def generate(*args, **kwargs):
        calls.append(('generate', args[1], kwargs['evaluation']))
        return sentinel
    def log(*args):
        calls.append(('log', args[0]))
        return True
    monkeypatch.setattr(schedule, 'generate_rollout', generate)
    monkeypatch.setattr(schedule, 'log_eval', log)
    output = schedule.generate(SimpleNamespace(), completed-1, None, evaluation=True)
    if completed not in steps:
        assert output.data == {} and output.metrics == {'scheduled_skip': True}
        assert schedule.log(completed-1, None, output.data, output.metrics)
        assert calls == []
    else:
        assert output is sentinel
        assert schedule.log(completed-1, None, {'screensim': {}}, None)
        assert calls == [('generate', completed-1, True), ('log', completed-1)]


def test_skip_does_not_hide_unexpected_evaluation_data():
    with pytest.raises(AssertionError):
        schedule.log(2, None, {'screensim': {}}, {'scheduled_skip': True})


@pytest.mark.parametrize('name,start,target,steps,cursor', [
    ('continue09',2,9,[6,9],96), ('continue15',9,15,[12,15],432)])
def test_resume_profiles_preserve_existing_and_new_plans(monkeypatch,name,start,target,steps,cursor):
    import runpy
    from pathlib import Path
    monkeypatch.setenv('Q35_CONTINUATION_PROFILE', name)
    profile = runpy.run_path(str(Path(__file__).resolve().parents[2]/'examples/interact/qwen35_continue_profile.py'))
    assert (profile['START'],profile['TARGET'],profile['EVAL_STEPS']) == (start,target,steps)
    assert profile['SAMPLER']['sample_index'] == cursor
    assert profile['SAMPLER']['sample_group_index'] == cursor//8
