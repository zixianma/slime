"""Export matched validation evidence, without model inference or inferred pairing."""
import hashlib
import base64
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'interact-runs/validation-replay'
BASE = Path('/gpfs/scrubbed/zixianma/checkpoints/web')


GROUP_SIZE = 8


def collect(root, version, evaluation):
    found = {}
    for path in (root / 'episodes').glob('*/training_audit.json'):
        audit = json.loads(path.read_text())
        first = audit[0]
        if first['metadata']['evaluation'] != evaluation or first['weight_versions'] != [version]:
            continue
        # Validation reused rollout IDs across checkpoints. Training rollout IDs
        # are globally increasing, so match the deterministic sample slot within
        # each scenario group instead.
        attempt = first['rollout_id'] if evaluation else first['rollout_id'] % GROUP_SIZE
        key = (first['metadata']['group_key'], attempt)
        assert key not in found, key
        found[key] = (path.parent, audit)
    assert len(found) == 48, len(found)
    return found


def export(item):
    directory, audit = item
    report = json.loads((directory / 'report.json').read_text())
    spec = json.loads((directory / 'spec.json').read_text())
    decisions = [json.loads(line) for line in (directory / 'decisions.jsonl').read_text().splitlines()]
    turns = []
    for record, entry in zip(decisions, audit, strict=True):
        d = record['decision']
        images = []
        for uri in d['observation']['images']:
            header, encoded = uri.split(',', 1)
            assert header == 'data:image/png;base64'
            data = base64.b64decode(encoded, validate=True)
            name = hashlib.sha256(data).hexdigest() + '.png'
            target = OUT / 'images' / name
            if not target.exists():
                target.write_bytes(data)
            images.append('images/' + name)
        turns.append(dict(tick=d['tick'], images=images, raw=record['response'],
                          prompt=d['observation']['user'], prompt_hash=entry['metadata']['prompt_hash']))
    prompt = turns[0]['prompt']
    task = prompt.split("== WHAT THEY'RE TRYING TO DO ==", 1)[1].split('== HOW IT IS DONE (reference) ==', 1)[0].strip()
    return dict(id=directory.name, source=str(directory), spec=spec, report=report, turns=turns, task=task)


def main():
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    (OUT / 'pairs').mkdir(exist_ok=True)
    index = []
    roots = (BASE / 'slime-qwen35-292972/pilot05', BASE / 'slime-qwen35-294135/continue15')
    for split, evaluation in (('validation', True), ('training', False)):
        old = collect(roots[0], '1', evaluation)
        new = collect(roots[1], '4', evaluation)
        assert old.keys() == new.keys()
        for key in sorted(old):
            a, b = export(old[key]), export(new[key])
            assert a['spec'] == b['spec']
            assert a['turns'][0]['prompt_hash'] == b['turns'][0]['prompt_hash']
            assert a['turns'][0]['images'] == b['turns'][0]['images']
            ar, br = a['report'], b['report']
            assert ar['episode'] == br['episode']
            # Validate the designed error identity while allowing timing/outcomes to diverge.
            assert [x['kind'] for x in ar['beats']] == [x['kind'] for x in br['beats']]
            category = 'improved' if br['success'] > ar['success'] else 'regressed' if br['success'] < ar['success'] else 'neutral'
            file = f'pairs/{len(index)}.json'
            (OUT / file).write_text(json.dumps(dict(before=a, after=b), ensure_ascii=False))
            index.append(dict(file=file, split=split, task=ar['task'], scenario=ar['episode'], attempt_id=key[1],
                              category=category, before=ar['success'], after=br['success']))
    expected = {'validation': (25, 35), 'training': (33, 37)}
    for split, (before, after) in expected.items():
        rows = [x for x in index if x['split'] == split]
        assert len(rows) == 48
        assert sum(x['before'] for x in rows) == before
        assert sum(x['after'] for x in rows) == after
    (OUT / 'index.json').write_text(json.dumps(index))
    shutil.copyfile(Path(__file__).with_name('validation_replay.html'), OUT / 'index.html')
    legacy = OUT.parent / 'screensim-step0-vs-step12-direct.html'
    archived = OUT.parent / 'screensim-training-comparison-unmatched.html'
    if legacy.exists() and not archived.exists():
        shutil.copyfile(legacy, archived)
    legacy.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0;url=validation-replay/">'
        '<title>Matched validation replay</title>'
        '<p>The earlier training comparison used unmatched scenarios. '
        '<a href="validation-replay/">Open the corrected paired validation replay.</a></p></html>')
    print(json.dumps(dict(output=str(OUT), pairs=len(index), splits={split:dict(
        pairs=sum(x['split'] == split for x in index),
        before=sum(x['before'] for x in index if x['split'] == split),
        after=sum(x['after'] for x in index if x['split'] == split),
        categories={k:sum(x['split'] == split and x['category'] == k for x in index)
                    for k in ['improved','regressed','neutral']}) for split in expected})))


if __name__ == '__main__':
    main()
