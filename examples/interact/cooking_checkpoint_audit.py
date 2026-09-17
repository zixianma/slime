"""Audit a restored learner only while its memory-saver allocation is awake."""
import json
import os
from pathlib import Path


def verify(actor, completed, snapshot, get_rank):
    offloaded = actor.args.offload_train
    if offloaded:
        actor.wake_up()
    try:
        rank = get_rank()
        name = (f'weights-update{completed}-rank{rank}.json' if completed is not None
                else f'weights-rank{rank}.json')
        audit = json.loads((Path(os.environ['INTERACT_JOB_DIR'])/name).read_text())
        assert snapshot(actor.model) == audit['after'], 'restored parameters differ from saved post-update audit'
        result = dict(rank=rank, weights_match=True)
        if completed is not None:
            assert not actor.args.finetune and not actor.args.no_load_optim and not actor.args.no_load_rng
            steps = set()
            for wrapper in getattr(actor.optimizer, 'chained_optimizers', [actor.optimizer]):
                inner = wrapper.optimizer
                for group in inner.param_groups:
                    if group.get('params') and 'step' in group:
                        steps.add(float(group['step']))
                for state in inner.state.values():
                    if 'step' in state:
                        steps.add(float(state['step']))
            assert steps == {completed}, f'optimizer step was not restored: {steps}'
            count = actor.opt_param_scheduler.num_steps
            assert count == completed*actor.args.global_batch_size, 'scheduler progress was not restored'
            result.update(optimizer_steps=sorted(steps), scheduler_samples=count)
        return result
    finally:
        if offloaded:
            actor.sleep()
