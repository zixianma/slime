"""Only evaluate at the two approved completed-update coordinates."""
from slime.rollout.base_types import RolloutFnEvalOutput
from slime.rollout.sglang_rollout import generate_rollout
from examples.interact.qwen35_metrics import log_eval

from examples.interact.qwen35_continue_profile import EVAL_STEPS

STEPS = set(EVAL_STEPS)


def generate(args, rollout_id, data_source, evaluation=False):
    assert evaluation
    if rollout_id + 1 not in STEPS:
        return RolloutFnEvalOutput(data={}, metrics={'scheduled_skip': True})
    return generate_rollout(args, rollout_id, data_source, evaluation=True)


def log(rollout_id, args, data, extra_metrics):
    if rollout_id + 1 not in STEPS:
        assert not data and extra_metrics == {'scheduled_skip': True}
        print(f'EVAL_NOT_SCHEDULED completed_updates={rollout_id+1}', flush=True)
        return True  # No zero-success point for an intentionally absent evaluation.
    return log_eval(rollout_id, args, data, extra_metrics)
