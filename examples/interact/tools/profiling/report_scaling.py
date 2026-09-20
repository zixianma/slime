"""Read-only comparison of the approved scaling jobs, including partial progress."""
import argparse
import csv
import json
from pathlib import Path
import statistics

ROOT = Path('/gpfs/scrubbed/zixianma/checkpoints/web')
JOBS = ((295509, 2, 3600), (295510, 4, 1800))


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # an in-flight final append is not a completed measurement
    return result


def distribution(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p90=values[min(len(values)-1, int(.9*len(values)))])


def summarize(root, job, gpus, limit):
    turns = rows(root/'turns.jsonl')
    completed = [r for r in turns if r['status'] == 'completed']
    segments = rows(root/'segments.jsonl')
    progress = rows(root/'progress.json')
    result = json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else {}
    elapsed = result.get('elapsed_s', progress[-1]['elapsed_job_s'] if progress else None)
    if not result and turns:
        elapsed = max([elapsed or 0]+[r.get('elapsed_job_s',0) for r in turns]) or None
    summary = dict(job=job, gpus=gpus, limit_s=limit, elapsed_s=elapsed,
                   status=result.get('status', 'running' if root.exists() else 'not_started'),
                   engine={}, server_startup_s=result.get('server_startup_s'), error=result.get('error'))
    for engine in ('cooking','screensim'):
        data = [r for r in completed if r['engine'] == engine]
        summary['engine'][engine] = dict(completed_turns=len(data),
            completed_segments=sum(r['engine']==engine and r['status'] in ('prefix_completed','episode_completed') for r in segments),
            full_episodes=sum(r['engine']==engine and r['status']=='episode_completed' for r in segments),
            failures=sum(r['engine']==engine and r['status']=='failed' for r in segments),
            phase_seconds=result.get('stats',{}).get(engine,{}).get('phase_seconds'),
            timings={k:distribution([r.get(k) for r in data]) for k in
                     ('image_decode_s','chat_template_s','processor_tokenize_s','http_roundtrip_s','env_step_s','total_s','completion_tokens')},
            server_timings={k:distribution([r.get('server',{}).get(k) for r in data]) for k in
                            ('e2e_latency','queue_time','prefill_launch_latency','inference_time')})
    summary['completed_turns_per_allocated_gpu_hour'] = len(completed)/(gpus*elapsed/3600) if elapsed else None
    util = []
    if (root/'gpu-utilization.csv').exists():
        for row in csv.reader((root/'gpu-utilization.csv').read_text().splitlines()):
            try:
                util.append(float(row[3]))
            except (ValueError, IndexError):
                pass
    summary['gpu_utilization_percent'] = distribution(util)
    summary['browser_backends'] = [json.loads(p.read_text()) for p in (root/'episodes').glob('*/browser_backend.json')]
    return summary


def comparison():
    result = [summarize(ROOT/f'engine-scaling-{job}',job,gpus,limit) for job,gpus,limit in JOBS]
    return dict(jobs=result, caveat='Live prefix profiling, not RL. Per-engine counts matter; do not compare mixed totals at different phases. GPU utilization includes startup.',
                both_finished=all((ROOT/f'engine-scaling-{job}/result.json').exists() for job,_,_ in JOBS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(comparison(), indent=2))
