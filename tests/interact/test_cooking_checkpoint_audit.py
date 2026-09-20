import json
from types import SimpleNamespace
import pytest
from examples.interact.archive.cooking_calibration.cooking_checkpoint_audit import verify


@pytest.mark.parametrize('matching', [True, False])
def test_restoration_audit_wakes_memory_before_reading_and_always_offloads(tmp_path, monkeypatch, matching):
    monkeypatch.setenv('INTERACT_JOB_DIR', str(tmp_path))
    (tmp_path/'weights-update1-rank0.json').write_text(json.dumps({'after':{'weight':'correct'}}))
    events = []
    actor = SimpleNamespace(args=SimpleNamespace(offload_train=True, finetune=False,
                            no_load_optim=False, no_load_rng=False, global_batch_size=48), model=object(),
                            optimizer=SimpleNamespace(optimizer=SimpleNamespace(
                                param_groups=[dict(params=[1],step=1)],state={})),
                            opt_param_scheduler=SimpleNamespace(num_steps=48),
                            wake_up=lambda:events.append('wake'), sleep=lambda:events.append('sleep'))
    def snapshot(model):
        assert model is actor.model and events == ['wake']
        events.append('snapshot')
        return {'weight':'correct' if matching else 'wrong'}
    if matching:
        assert verify(actor, 1, snapshot, lambda:0) == dict(rank=0, weights_match=True,
                                                         optimizer_steps=[1.], scheduler_samples=48)
        events.clear()
        actor.optimizer.optimizer.param_groups[0]['step'] = 0
        with pytest.raises(AssertionError, match='optimizer step was not restored'):
            verify(actor, 1, snapshot, lambda:0)
    else:
        with pytest.raises(AssertionError, match='restored parameters differ'):
            verify(actor, 1, snapshot, lambda:0)
    assert events == ['wake', 'snapshot', 'sleep']
