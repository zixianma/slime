"""Freeze a score-independent cooking development split; never launch compute."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from interact_env import EpisodeSpec


def prepare(output, *, extended_budget=False):
    native = ROOT.parent / 'cook-bench-engine/bench/v5'
    source = native / 'cases_composite_v4.json'
    cases = json.loads(source.read_text())['cases']
    cells = defaultdict(list)
    for case in cases:
        cells[case['cell']].append(case)
    assert len(cases) == 50 and len(cells) == 10
    seed = 'cooking-rl-v1-20260914'
    rank = lambda case: hashlib.sha256(f"{seed}:{case['name']}".encode()).hexdigest()
    groups = [sorted(cells[cell], key=rank) for cell in sorted(cells)]
    assert all(len(group) == 5 for group in groups)
    validation = [group[0] for group in groups]
    train = [group[i] for i in range(1, 5) for group in groups]
    # Four preselected scenarios for the bounded pipeline calibration only.
    calibration = [validation[i] for i in (0, 3, 6, 9)]
    profile = dict(human='scripted', persona='baseline', observation='frames',
                   latency_ticks=0, scripted_accept=True, wall_seconds=3600 if extended_budget else 1800,
                   reward_version='native_outcome_v1')
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, values in [('train', train), ('validation', validation), ('calibration_validation', calibration)]:
        rows = [dict(prompt=[dict(role='user', content='Native assistant episode')],
                     metadata=dict(episode=asdict(EpisodeSpec('cooking', c['name'], seed=0, config=profile)),
                                   split='train' if name == 'train' else 'validation', cell=c['cell']))
                for c in values]
        content = ''.join(json.dumps(row) + '\n' for row in rows)
        (output / f'{name}.jsonl').write_text(content)
        files[name] = dict(count=len(rows), sha256=hashlib.sha256(content.encode()).hexdigest(),
                           scenarios=[c['name'] for c in values])
    assert not set(files['train']['scenarios']) & set(files['validation']['scenarios'])
    manifest = dict(version=seed + ('-budget-v2' if extended_budget else ''), selection='SHA256-ranked: one held-out variant per recipe/layout cell',
        generalization='Held-out scenario variants; recipes and layouts overlap. Development validation, not unseen-recipe testing.',
        profile=profile, source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (source, native/'grid.json')}, files=files,
        calibration_note='Four preselected validation scenarios for pipeline checks; not the full validation estimate.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), train=40, validation=10, calibration_validation=4)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extended-budget', action='store_true',
                        help='Versioned rerun: 3600s native base budget (7200s for composite tasks); unchanged split')
    args = parser.parse_args()
    prepare(args.output, extended_budget=args.extended_budget)
