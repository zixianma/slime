"""Copy the two approved scalar histories into a new, attributed W&B run.

Never mutates source runs or uploads checkpoints, episodes, credentials, source
code, or system telemetry. Export is read-only; --publish explicitly creates the
merged run. Original custom step coordinates are retained without offsets.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

PROJECT = 'zixianma/interact-slime-rl'
SOURCES = ['6neavc52', '6kv7biu2']
PREFIXES = ('train/', 'rollout/', 'eval/', 'perf/', 'multi_turn/', 'passrate/')
CONFIG_KEYS = ('interact_split_sha256', 'interact_split_version', 'interact_model_revision',
               'hf_checkpoint', 'lr', 'rollout_batch_size', 'n_samples_per_prompt',
               'rollout_temperature', 'rollout_top_p', 'rollout_top_k',
               'freeze_params_name_list', 'advantage_estimator')


def scalars(row):
    result = {}
    for key, value in row.items():
        if not key.startswith(PREFIXES) or value is None:
            continue
        if not isinstance(value, (int, float, bool)) or not math.isfinite(value):
            raise ValueError(f'non-finite/non-scalar metric: {key}')
        result[key] = value
    return result


def points(rows, metric, axis):
    result = {}
    for row in rows:
        if metric not in row or row[metric] is None:
            continue
        if axis not in row or row[axis] is None:
            raise ValueError(f'{metric} has no {axis}')
        step = row[axis]
        if step in result and result[step] != row[metric]:
            raise ValueError(f'conflicting {metric} at {step}')
        result[step] = row[metric]
    return result


def validate(bundle):
    if [s['id'] for s in bundle] != SOURCES:
        raise ValueError('unexpected source runs')
    for key in CONFIG_KEYS:
        if bundle[0]['config'].get(key) != bundle[1]['config'].get(key):
            raise ValueError(f'source configuration mismatch: {key}')
    expected_steps = [set(range(2)), set(range(2,9))]
    for source, expected in zip(bundle, expected_steps, strict=True):
        if set(points(source['rows'], 'train/grad_norm', 'train/step')) != expected:
            raise ValueError('source optimizer history is incomplete')
        if set(points(source['rows'], 'train/success', 'train/step')) != expected:
            raise ValueError('source episode-success history is incomplete')
    rows = [r for s in bundle for r in s['rows']]
    expected_eval = {0: 25/48, 2: 27/48, 6: 32/48, 9: 29/48}
    actual = points(rows, 'eval/success', 'eval/step')
    if actual.keys() != expected_eval.keys() or any(abs(actual[k]-v)>1e-10 for k,v in expected_eval.items()):
        raise ValueError('validation history does not match completed native evaluations')
    return rows


def compare_histories(expected, actual):
    metrics = set().union(*(r.keys() for r in expected))
    for metric in metrics:
        before = [r[metric] for r in expected if metric in r]
        after = [r[metric] for r in actual if metric in r]
        if before != after:
            raise ValueError(f'merged read-back mismatch: {metric}')
    return len(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    import wandb
    api = wandb.Api(timeout=60)
    snapshot = args.output/'source_histories.json'
    if args.verify_only:
        bundle = json.loads(snapshot.read_text())
        receipt = args.output/'merged_run.json'
        location = json.loads(receipt.read_text())
        actual = [scalars(r) for r in api.run(f"{PROJECT}/{location['id']}").scan_history(page_size=100)]
        count = compare_histories(validate(bundle), actual)
        location.update(status='verified', scalar_metrics=count)
        receipt.write_text(json.dumps(location, indent=2)+'\n')
        print(json.dumps(location))
        return
    if not args.publish:
        args.output.mkdir(parents=True, exist_ok=False)
        bundle = []
        for source_id in SOURCES:
            run = api.run(f'{PROJECT}/{source_id}')
            raw = list(run.scan_history(page_size=100))
            rows = [scalars(row) for row in sorted(raw, key=lambda r:r['_step'])]
            rows = [row for row in rows if row]
            source = dict(id=source_id, url=run.url, state=run.state,
                          config={k:run.config.get(k) for k in CONFIG_KEYS}, rows=rows)
            bundle.append(source)
            print(json.dumps(dict(id=source_id, state=run.state, rows=len(rows),
                optimizer=points(rows,'train/grad_norm','train/step'),
                evaluation=points(rows,'eval/success','eval/step'))), flush=True)
        snapshot.write_text(json.dumps(bundle, indent=2)+'\n')
        validate(bundle)
        print('EXPORT_VALIDATED', flush=True)
        return
    bundle = json.loads(snapshot.read_text())
    rows = validate(bundle)
    receipt = args.output/'merged_run.json'
    if receipt.exists():
        raise ValueError('merge already created; inspect its receipt rather than publish again')
    for source in bundle:
        if api.run(f"{PROJECT}/{source['id']}").state != 'finished':
            raise ValueError('source run is not finished')
    run_id = uuid.uuid4().hex[:12]
    config = dict(bundle[0]['config'], historical_merge=True,
                  source_runs=[s['url'] for s in bundle], completed_updates=9,
                  source_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                  train_axis='zero-based optimizer index 0..8; success is from the corresponding pre-update collection',
                  eval_axis='completed optimizer updates: 0, 2, 6, 9',
                  timing_note='Merge timestamps/system usage are not training timestamps or GPU usage; see source runs.',
                  rollout_zero_note='The first collection was replayed before any successful update; its perf metrics measure file loading.')
    with wandb.init(entity='zixianma', project='interact-slime-rl', id=run_id,
                    name='screensim-qwen35-4b-9updates-merged', group='screensim-qwen35-9updates',
                    job_type='history-merge', tags=['screensim','qwen3.5-4b','merged-history'],
                    resume='never', mode='online', config=config, dir=str(args.output),
                    notes='Combined scalar history from jobs 292972 and 293612. Original runs are unchanged. No new training.',
                    settings=wandb.Settings(console='off', disable_code=True, disable_git=True,
                                            x_disable_stats=True, x_disable_meta=True)) as run:
        receipt.write_text(json.dumps(dict(id=run.id,url=run.url,status='uploading'),indent=2)+'\n')
        for axis in ('train/step','rollout/step','eval/step'):
            run.define_metric(axis)
        for prefix, axis in [('train/*','train/step'), ('rollout/*','rollout/step'),
                             ('eval/*','eval/step'), ('perf/*','rollout/step'),
                             ('multi_turn/*','rollout/step'), ('passrate/*','rollout/step')]:
            run.define_metric(prefix, step_metric=axis, step_sync=False)
        for source in bundle:
            for row in source['rows']:
                run.log({**row, 'merge/source_run_id':source['id']})
        macro = points(rows, 'eval/task_macro_success', 'eval/step')
        best = max(macro, key=macro.get)
        run.summary.update(dict(completed_updates=9, historical_merge=True,
            best_eval_completed_updates=best, best_eval_task_macro_success=macro[best],
            final_eval_task_macro_success=macro[max(macro)]))
        url = run.url
    # Read-back: compare every copied scalar point, not sampled history/summary.
    for attempt in range(6):
        api.flush()
        actual_rows = [scalars(r) for r in api.run(f'{PROJECT}/{run_id}').scan_history(page_size=100)]
        try:
            count = compare_histories(rows, actual_rows)
            break
        except ValueError:
            if attempt == 5:
                raise
            time.sleep(2)
    receipt.write_text(json.dumps(dict(id=run_id,url=url,status='verified',
        source_runs=SOURCES, scalar_metrics=count, source_rows=len(rows)),indent=2)+'\n')
    print(json.dumps(dict(url=url,status='verified',scalar_metrics=count)), flush=True)


if __name__ == '__main__':
    main()
