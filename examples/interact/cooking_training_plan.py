"""Validated full cooking experiment contract; does not launch compute."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT/'interact-runs/cooking-rl-v1-20260914-budget-v2'
TARGET_UPDATES = 12
EVAL_UPDATES = (0, 3, 6, 9, 12)
GROUPS_PER_UPDATE = 6
ROLLOUTS_PER_GROUP = 8
EVAL_ATTEMPTS = 4


def should_evaluate(completed_updates):
    if not 0 <= completed_updates <= TARGET_UPDATES:
        raise ValueError('completed update outside experiment')
    return completed_updates in EVAL_UPDATES


def validate_split(split=SPLIT):
    manifest = json.loads((split/'manifest.json').read_text())
    members = {}
    for name, count in (('train', 40), ('validation', 10)):
        raw = (split/f'{name}.jsonl').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest['files'][name]['sha256']
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == count
        members[name] = {row['metadata']['episode']['task_id'] for row in rows}
        assert len(members[name]) == count
        assert members[name] == set(manifest['files'][name]['scenarios'])
        assert all(row['metadata']['split'] == name for row in rows)
    assert members['train'].isdisjoint(members['validation'])
    for name, digest in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT.parent/'cook-bench-engine/bench/v5'/name).read_bytes()).hexdigest() == digest
    return manifest


def experiment():
    manifest = validate_split()
    return dict(
        version='cooking-full-12-eval3-v1', target_updates=TARGET_UPDATES,
        eval_completed_updates=list(EVAL_UPDATES), checkpoint_every_updates=1,
        train_scenarios=40, validation_scenarios=10,
        groups_per_update=GROUPS_PER_UPDATE, rollouts_per_group=ROLLOUTS_PER_GROUP,
        global_batch_episodes=GROUPS_PER_UPDATE*ROLLOUTS_PER_GROUP,
        train_episodes_total=TARGET_UPDATES*GROUPS_PER_UPDATE*ROLLOUTS_PER_GROUP,
        eval_attempts_per_scenario=EVAL_ATTEMPTS,
        eval_episodes_per_round=10*EVAL_ATTEMPTS,
        eval_episodes_total=len(EVAL_UPDATES)*10*EVAL_ATTEMPTS,
        renderer='vulkan', max_running_requests=4,
        source_split_version=manifest['version'], source_split_files=manifest['files'],
        reward_version='native_outcome_v1', human='scripted',
        base_model='Qwen3.5-4B', freeze_vision=True, learning_rate=5e-7,
        wandb_project='interact-slime-rl', wandb_identity='one persistent cooking-only run',
        rollout_unit='complete native episode; 16-turn prefixes are not training data',
        readiness='contract validated; GPU handoff and full-episode learner/resume validation pending',
    )


if __name__ == '__main__':
    print(json.dumps(experiment(), indent=2))
