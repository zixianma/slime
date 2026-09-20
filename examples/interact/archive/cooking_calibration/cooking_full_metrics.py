"""Full experiment metrics: never accept calibration-sized batches."""
from collections import defaultdict
from pathlib import Path
import os
import json
from examples.interact.archive.cooking_calibration.cooking_calibration_metrics import summarize, record
from interact_env.slime_bridge.metrics import episode_metrics


TRAINING_REWARD_VERSION = 'native_outcome_lexicographic_v2'


def lexicographic_reward(components):
    """Keep outcome success primary; break binary ties with detector quality."""
    success = float(components['task_success'])
    planned = int(components['errors_planned'])
    detected = int(components['detected_errors'])
    false_flags = int(components['false_flags'])
    assert success in (0., 1.) and planned >= 0 and 0 <= detected <= planned and false_flags >= 0
    recall = detected / planned if planned else 0.
    reward = success + .1 * recall - .01 * min(false_flags, 10)
    # Success remains lexicographically dominant under every valid tie-break.
    assert (-.1 <= reward <= .1) if not success else (.9 <= reward <= 1.1)
    return reward


def apply_training_reward(samples):
    episodes = {}
    groups = defaultdict(list)
    for sample in samples:
        episode_id = sample.metadata['episode_id']
        reward = lexicographic_reward(sample.metadata['reward_components'])
        if episode_id in episodes:
            assert episodes[episode_id] == reward
        else:
            episodes[episode_id] = reward
            groups[sample.group_index].append(reward)
        sample.metadata['raw_outcome_reward'] = sample.reward
        sample.metadata['training_reward_version'] = TRAINING_REWARD_VERSION
        sample.reward = reward
    mixed = sum(len(set(values)) > 1 for values in groups.values()) / len(groups)
    return dict(episodes=len(episodes), groups=len(groups), mixed_group_fraction=mixed,
                mean=sum(episodes.values())/len(episodes), minimum=min(episodes.values()),
                maximum=max(episodes.values()))


def log_rollout(rollout_id, args, samples, *unused):
    from slime.observability.logging_utils import log
    values = summarize(samples, 'train') | episode_metrics(samples)
    assert values['train/episodes'] == 48 and values['train/groups'] == 6
    values.update({'train/step':rollout_id+1, 'train/rollout_policy_update':rollout_id})
    if rollout_id >= 2:
        shaped = apply_training_reward(samples)
        assert shaped['episodes'] == 48 and shaped['groups'] == 6
        values.update({'train/reward_tiebreak_applied': 1,
                       'train/reward_tiebreak_mixed_group_fraction': shaped['mixed_group_fraction'],
                       'train/shaped_reward': shaped['mean'],
                       'train/shaped_reward_min': shaped['minimum'],
                       'train/shaped_reward_max': shaped['maximum']})
    else:
        shaped = dict(mixed_group_fraction=values['train/success_mixed_group_fraction'])
        values['train/reward_tiebreak_applied'] = 0
    # Reserve learner time using this batch's real work, not episode count alone.
    workload = dict(turns=len(samples), tokens=sum(len(s.tokens) for s in samples))
    (Path(os.environ['INTERACT_JOB_DIR'])/f'optimizer-workload-{rollout_id}.json').write_text(
        json.dumps(workload)+'\n')
    record(values)
    log(args, values, step_key='train/step')
    if shaped['mixed_group_fraction'] == 0:
        (Path(os.environ['INTERACT_JOB_DIR'])/'no_reward_variance.json').write_text(json.dumps(values)+'\n')
        raise RuntimeError('COOKING_NO_REWARD_VARIANCE: no informative groups; no optimizer update')
    return False


def log_eval(completed, args, data, extra):
    from slime.observability.logging_utils import log
    from examples.interact.archive.cooking_calibration.cooking_training_plan import should_evaluate
    assert should_evaluate(completed)
    samples = [s for dataset in data.values() for s in dataset['samples']]
    values = summarize(samples, 'eval')
    groups = {}
    for s in samples:
        groups.setdefault(s.metadata['group_key'], set()).add(s.metadata['episode_id'])
    assert len(groups) == 10 and {len(v) for v in groups.values()} == {4}
    assert values['eval/episodes'] == 40
    values['eval/step'] = completed
    record(values)
    log(args, values, step_key='eval/step')
    return True
