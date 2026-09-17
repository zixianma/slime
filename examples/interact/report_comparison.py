"""Summarize completed local rollouts; never upload or launch compute."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from examples.interact.comparison_metrics import summarize


def wilson(successes, n):
    if not n: return None
    z=1.959963984540054
    p=successes/n; denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0,center-delta),min(1,center+delta)]


def read_records(path):
    if not path.exists(): return []
    text=path.read_text()
    lines=text.splitlines(keepends=True)
    # A running process may be in the middle of its final append. Other corruption
    # is an error, not silently omitted data.
    if lines and not lines[-1].endswith('\n'): lines=lines[:-1]
    rows=[json.loads(line) for line in lines]
    keys=[(r['scenario_id'],r['attempt']) for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError('duplicate scenario/attempt records')
    return rows


def report(job):
    manifest=json.loads((job.parent/'manifest.json').read_text())
    models=[]
    record_sets=[]
    for index,spec in enumerate(manifest['models']):
        records=read_records(job/f'model-{index}/episodes.jsonl')
        record_sets.append(records)
        cells=defaultdict(list)
        for row in records:
            if row['status']=='completed': cells[row['scenario_id']].append(row)
        details=[]
        for scenario in manifest['scenarios']:
            rows=cells[scenario['id']]; n=len(rows)
            successes=sum(r['components']['success'] for r in rows)
            contrast={}
            for k in (8,16):
                if n>=k:
                    contrast[str(k)]=1-(math.comb(successes,k)+math.comb(n-successes,k))/math.comb(n,k)
            details.append(dict(scenario_id=scenario['id'],split=scenario['split'],
                completed=n,planned=scenario['attempts'],successes=successes,
                success_rate=successes/n if n else None,success_wilson95=wilson(successes,n),
                f1=sum(r['components']['f1'] for r in rows)/n if n else None,
                empirical_mixed_group_probability=contrast))
        state_path=job/f'model-{index}/summary.json'
        status=json.loads(state_path.read_text())['status'] if state_path.exists() else 'running'
        models.append(dict(model=spec['id'],status=status,metrics=summarize(records,manifest['scenarios']),
                           scenarios=details))
    finished_keys=[{(r['scenario_id'],r['attempt']) for r in rows if r['status']=='completed'}
                   for rows in record_sets]
    common=set.intersection(*finished_keys)
    matched=[]
    for model,rows in zip(models,record_sets,strict=True):
        subset=[r for r in rows if r['status']=='completed' and (r['scenario_id'],r['attempt']) in common]
        matched.append(dict(model=model['model'],metrics=summarize(subset,manifest['scenarios'])))
    return dict(job=job.name,models=models,matched_completed_episodes=len(common),matched_metrics=matched,
        caveats=['Frozen policy screen, no evidence of RL improvement.',
                 'Scripted baseline human, not full benchmark personas.',
                 'Wilson intervals are per-scenario binomial approximations, not task-generalization intervals.',
                 'Matched completed cases can be deadline/length selected; partial coverage is not a full-suite ranking.',
                 'Empirical mixed-group probability resamples observed outcomes without replacement; zero observed successes is not proof of zero true success.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=report(args.job_dir)
    if args.output:
        with args.output.open('x') as out: out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'job':result['job'],'matched_completed_episodes':result['matched_completed_episodes'],
        'models':[{k:v for k,v in m.items() if k!='scenarios'} for m in result['models']]},indent=2))


if __name__=='__main__': main()
