"""Episode/task-weighted metrics for the versioned ScreenSim RL pilot."""
from collections import defaultdict
import json
import os
from pathlib import Path

from interact_env.slime_bridge.metrics import episode_metrics


def save_metrics(values):
    if os.environ.get('INTERACT_JOB_DIR'):
        with (Path(os.environ['INTERACT_JOB_DIR']) / 'episode_metrics.jsonl').open('a') as out:
            out.write(json.dumps(values) + '\n')
    print('SCREENSIM_METRICS ' + json.dumps(values), flush=True)


def summarize(samples, prefix):
    # Validate full trajectories before reducing repeated per-turn rewards.
    episode_metrics(samples)
    episodes = {s.metadata['episode_id']: s for s in samples}
    tasks, groups = defaultdict(list), defaultdict(list)
    for sample in episodes.values():
        tasks[sample.metadata['task_id']].append(float(sample.metadata['reward_components']['success']))
        groups[sample.group_index].append(sample.reward)
    means = {k: sum(v)/len(v) for k, v in tasks.items()}
    metrics = {f'{prefix}/episodes': len(episodes), f'{prefix}/tasks': len(tasks),
               f'{prefix}/success': sum(sum(v) for v in tasks.values()) / len(episodes),
               f'{prefix}/task_macro_success': sum(means.values()) / len(means)}
    metrics.update({f'{prefix}/task/{k}/success': v for k, v in means.items()})
    if prefix == 'train':
        metrics[f'{prefix}/groups'] = len(groups)
        metrics[f'{prefix}/success_mixed_group_fraction'] = sum(len(set(v)) > 1 for v in groups.values()) / len(groups)
    return metrics


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    from slime.observability import logging_utils
    values = episode_metrics(samples) | summarize(samples, 'train')
    # Each pilot collection corresponds to one optimizer batch.
    values['rollout/step'] = rollout_id
    values['train/step'] = rollout_id
    save_metrics(values)
    logging_utils.log(args, values, step_key='rollout/step')
    return False


def log_eval(rollout_id, args, data, extra_metrics):
    from slime.observability import logging_utils
    samples = [s for dataset in data.values() for s in dataset['samples']]
    episodes = {s.metadata['episode_id']: s for s in samples}
    scenarios = defaultdict(int)
    for sample in episodes.values():
        scenarios[sample.metadata['group_key']] += 1
    if len(scenarios) != 12 or set(scenarios.values()) != {args.n_samples_per_eval_prompt}:
        raise ValueError('evaluation is not a complete balanced pass over the 12 validation scenarios')
    values = summarize(samples, 'eval')
    # Upstream passes 0 before any updates and 2 after the third update.
    values['eval/step'] = 0 if rollout_id == 0 else rollout_id + 1
    save_metrics(values)
    logging_utils.log(args, values, step_key='eval/step')
    return True  # Upstream default would weight episodes by their turn counts.
