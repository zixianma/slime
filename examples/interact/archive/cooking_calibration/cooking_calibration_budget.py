"""Shared rerun budget and CPU-only validation; never submits compute."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ALLOCATION_SECONDS = 7 * 3600
CLEANUP_SECONDS = 180
STAGE_LIMITS = (300, 180, 22200, 600, 180)


def stage_deadline(global_deadline, started, limit, remaining_limits):
    return min(global_deadline-sum(remaining_limits), started+limit)


def split_path():
    return Path(os.environ.get('COOKING_SPLIT_DIR',
                str(ROOT/'interact-runs/cooking-rl-v1-20260914-budget-v2')))


def validate_profile(split=None, settings=None):
    import yaml
    split = split or split_path()
    manifest = json.loads((split/'manifest.json').read_text())
    if settings is None:
        settings = yaml.safe_load((ROOT/'examples/interact/configs/cooking_qwen35_rollout.yaml').read_text())['interact']
    assert manifest['version'] == 'cooking-rl-v1-20260914-budget-v2'
    assert manifest['profile']['wall_seconds'] == 3600
    assert settings['max_decisions'] == 200
    for name, info in manifest['files'].items():
        content = (split/f'{name}.jsonl').read_bytes()
        assert hashlib.sha256(content).hexdigest() == info['sha256']
        rows = [json.loads(line) for line in content.splitlines()]
        assert len(rows) == info['count']
        assert all(row['metadata']['episode']['config'] == manifest['profile'] for row in rows)
    # Three rollout phases, each at most 2h per composite episode, plus learner
    # startup/update overhead. Stage deadlines still bound unexpected stalls.
    assert STAGE_LIMITS[2] >= 3 * 2 * manifest['profile']['wall_seconds'] + 600
    assert sum(STAGE_LIMITS) + CLEANUP_SECONDS <= ALLOCATION_SECONDS
    return dict(split=str(split), version=manifest['version'], max_decisions=settings['max_decisions'],
                composite_wall_seconds=2*manifest['profile']['wall_seconds'],
                allocation_seconds=ALLOCATION_SECONDS, stage_limits=STAGE_LIMITS)


if __name__ == '__main__':
    print(json.dumps(validate_profile(), indent=2))
