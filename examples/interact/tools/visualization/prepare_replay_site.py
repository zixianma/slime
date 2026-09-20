"""Create a minimal sanitized hosting export; never uploads or deploys anything."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

BEAT = ('kind','detected_tick','injected_at','detected','accepted','prevented','fallback','skipped','slip_paths')
EVENT = ('kind','repair','slip','beat','t','text','who','credited')
REPORT = ('success','f1','false_flags','final_tick')
SECRET = re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{24,}|\b(?:WANDB_API_KEY|VERCEL_TOKEN|OPENAI_API_KEY)\s*[=:]\s*[^\s]+')


def select(value, keys):
    return {key:value[key] for key in keys if key in value}


def export(source, output, audience):
    assert not output.exists(), 'Never overwrite an existing hosted export'
    index = json.loads((source/'index.json').read_text())
    assert len(index) == 96
    assert {item['split'] for item in index} == {'training', 'validation'}
    assert sum(item['split'] == 'training' for item in index) == 48
    assert sum(item['split'] == 'validation' for item in index) == 48
    prepared = {}
    images = set()
    for item in index:
        name = item['file']
        assert re.fullmatch(r'pairs/\d+\.json', name)
        original = json.loads((source/name).read_text())
        pair = {}
        for side in ('before','after'):
            ep = original[side]
            report = select(ep['report'], REPORT)
            report['beats'] = [select(b,BEAT) for b in ep['report']['beats']]
            report['timeline'] = [select(e,EVENT) for e in ep['report']['timeline']]
            turns = [select(t, ('tick','images','raw','prompt','prompt_hash')) for t in ep['turns']]
            for turn in turns:
                for image in turn['images']:
                    assert re.fullmatch(r'images/[a-f0-9]{64}\.png', image)
                    data = (source/image).read_bytes()
                    assert hashlib.sha256(data).hexdigest() == Path(image).stem
                    images.add(image)
            pair[side] = dict(id=ep['id'], source='Episode '+ep['id'], task=ep['task'], report=report, turns=turns)
        text = json.dumps(pair, ensure_ascii=False)
        assert '/gpfs/' not in text and not SECRET.search(text), 'Unexpected path or potential credential; review before export'
        prepared[name] = text
    html = (source/'index.html').read_text()
    assert not SECRET.search(html)
    public = output/'public'
    (public/'pairs').mkdir(parents=True)
    (public/'images').mkdir()
    for name, text in prepared.items():
        (public/name).write_text(text)
    for name in sorted(images):
        shutil.copyfile(source/name, public/name)
    (public/'index.html').write_text(html)
    (public/'index.json').write_text(json.dumps(index))
    (public/'robots.txt').write_text('User-agent: *\nDisallow: /\n')
    audit = dict(pairs=len(index), unique_images=len(images), audience=audience,
                 bytes=sum(p.stat().st_size for p in public.rglob('*') if p.is_file()),
                 source=str(source), source_html_sha256=hashlib.sha256(html.encode()).hexdigest(),
                 access_requirement=('Owner-only authentication must cover HTML, JSON and images before publication'
                                     if audience == 'owner-only' else
                                     'Anonymous public access requested; sanitized fields and credential scan applied'),
                 removed='Absolute source paths, unused full reports and specs',
                 retained='Task instructions, raw responses, prompts, event timelines and original screenshots',
                 credential_scan='No high-confidence credential patterns found; not a proof of absence')
    (output/'export-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audience',choices=('owner-only','public'),default='owner-only')
    args = parser.parse_args()
    print(json.dumps(export(args.source,args.output,args.audience),indent=2))
