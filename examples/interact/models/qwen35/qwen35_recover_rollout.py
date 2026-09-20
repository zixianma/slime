"""Reuse only the pre-update pilot05 collection; later batches are fresh/on-policy.

This job-specific recovery requires the original untouched HF checkpoint and a
failed first learner call. It never replays stale data after an optimizer update.
"""
import json
import os
from pathlib import Path

from interact_env.protocol import EpisodeSpec
from interact_env.slime_bridge.metrics import episode_metrics
from slime.observability.rollout_data_utils import load_debug_rollout_data
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import generate_rollout as fresh_rollout


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if rollout_id != 0:
        return fresh_rollout(args, rollout_id, data_source, evaluation=evaluation)
    original = Path('/gpfs/scrubbed/zixianma/checkpoints/web/slime-qwen35-292972/pilot05')
    assert os.environ['SLURM_JOB_ID'] == '292972'
    assert Path(args.load).resolve() == Path(args.hf_checkpoint).resolve()
    assert not (original / 'checkpoints/latest_checkpointed_iteration.txt').exists()
    job = Path(os.environ['INTERACT_JOB_DIR'])
    assert json.loads((job / '14-pilot.status.json').read_text())['returncode'] != 0
    path = original / ('rollout-eval_0.pt' if evaluation else 'rollout-0.pt')
    samples = load_debug_rollout_data(str(path), rollout_id=0)
    episode_metrics(samples)
    assert all(s.metadata['evaluation'] == evaluation for s in samples)
    print(f'RECOVER_PRE_UPDATE_ROLLOUT source={path} evaluation={evaluation}', flush=True)
    if evaluation:
        return RolloutFnEvalOutput(data={'screensim': {
            'samples': samples, 'rewards': [s.reward for s in samples],
            'truncated': [False for s in samples]}})
    parents = data_source.get_samples(args.rollout_batch_size)
    expected = {EpisodeSpec(**group[0].metadata['episode']).group_key for group in parents}
    assert {s.metadata['group_key'] for s in samples} == expected
    assert len({s.rollout_id for s in samples}) == args.global_batch_size == 48
    return RolloutFnTrainOutput(samples=samples)
