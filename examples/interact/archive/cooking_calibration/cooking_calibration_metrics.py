"""Episode-weighted cooking logs and a no-signal gate before learner updates."""
from collections import defaultdict
import json
import os
from pathlib import Path

from interact_env.slime_bridge.metrics import episode_metrics


def summarize(samples, prefix):
    episode_metrics(samples)  # validates complete, unique trajectories
    episodes = {s.metadata['episode_id']: s for s in samples}
    groups, tasks = defaultdict(list), defaultdict(list)
    for sample in episodes.values():
        m = sample.metadata
        assert m['engine'] == 'cooking' and m['reward_version'] == 'native_outcome_v1'
        assert sample.reward in (0., 1.)
        groups[sample.group_index].append(sample.reward)
        tasks[m['task_id']].append(sample.reward)
    values = {f'{prefix}/episodes': len(episodes), f'{prefix}/success':
              sum(s.reward for s in episodes.values())/len(episodes),
              f'{prefix}/scenario_macro_success': sum(sum(v)/len(v) for v in tasks.values())/len(tasks)}
    for field in ('detected_errors', 'errors_planned', 'errors_offered', 'false_flags',
                  'invalid_response_fraction', 'native_ticks'):
        values[f'{prefix}/{field}'] = sum(s.metadata['reward_components'][field] for s in episodes.values())/len(episodes)
    values[f'{prefix}/turns'] = sum(s.metadata['num_turns'] for s in episodes.values())/len(episodes)
    for outcome in ('won', 'wrong_serve', 'burned', 'timeout'):
        values[f'{prefix}/outcome/{outcome}'] = sum(s.metadata['outcome'] == outcome for s in episodes.values())/len(episodes)
    for task, scores in tasks.items():
        values[f'{prefix}/scenario/{task}/success'] = sum(scores)/len(scores)
    if prefix == 'train':
        values['train/groups'] = len(groups)
        values['train/success_mixed_group_fraction'] = sum(len(set(v)) > 1 for v in groups.values())/len(groups)
    return values


def record(values):
    with (Path(os.environ['INTERACT_JOB_DIR'])/'episode_metrics.jsonl').open('a') as out:
        out.write(json.dumps(values)+'\n')
    print('COOKING_METRICS '+json.dumps(values), flush=True)


def log_rollout(rollout_id, args, samples, *unused):
    from slime.observability.logging_utils import log
    values = summarize(samples, 'train') | episode_metrics(samples)
    values.update({'train/step':rollout_id, 'rollout/step':rollout_id})
    assert values['train/episodes'] == 8 and values['train/groups'] == 2
    record(values)
    log(args, values, step_key='train/step')
    if values['train/success_mixed_group_fraction'] == 0:
        (Path(os.environ['INTERACT_JOB_DIR'])/'no_reward_variance.json').write_text(json.dumps(values)+'\n')
        raise RuntimeError('COOKING_NO_REWARD_VARIANCE: stopping before optimizer update')
    return False


def log_eval(completed_updates, args, data, extra):
    from slime.observability.logging_utils import log
    samples = [s for dataset in data.values() for s in dataset['samples']]
    values = summarize(samples, 'eval')
    scenarios = defaultdict(set)
    for s in samples:
        scenarios[s.metadata['group_key']].add(s.metadata['episode_id'])
    assert len(scenarios) == 4 and {len(v) for v in scenarios.values()} == {2}
    values['eval/step'] = completed_updates
    record(values)
    log(args, values, step_key='eval/step')
    return True
