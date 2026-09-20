"""Bounded live SGLang/native profiling. No learner, rewards or optimizer updates."""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODEL = os.environ.get(
    "Q35_MODEL",
    "/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison/models/Qwen3.5-4B",
)


def append(path, row):
    with path.open('a') as out:
        out.write(json.dumps(row)+'\n')


def prepare(path):
    cooking = ROOT/'interact-runs/cooking-rl-v1-20260914-budget-v2/calibration_validation.jsonl'
    rows = [json.loads(line)['metadata']['episode'] for line in cooking.read_text().splitlines()]
    screen_root = Path('/gpfs/scrubbed/zixianma/checkpoints/web/slime-qwen35-292972/pilot05/episodes')
    # Same two ScreenSim cases used in CPU profiling; frozen original specs.
    screens = [json.loads((screen_root/name/'spec.json').read_text()) for name in
               ('26411d1ba671451284cc91930fe39512', 'd05deda41a174b2aa520042fcd347e11')]
    data = dict(version='scaling-profile-v1', model=MODEL, seed=20260914,
                concurrency=8, max_turns=16, max_response_tokens=512,
                scenarios={'cooking':rows, 'screensim':screens},
                mode='live_policy_16_turn_prefix_not_RL',
                engine_order=['cooking','screensim'])
    with path.open('x') as out:
        out.write(json.dumps(data, indent=2)+'\n')


def server_command(port, gpu, max_running_requests=4):
    if max_running_requests not in (4, 6, 8):
        raise ValueError('profile supports request limits 4, 6, or 8')
    return [sys.executable, '-m', 'sglang.launch_server', '--model-path', MODEL,
            '--host', '127.0.0.1', '--port', str(port), '--base-gpu-id', str(gpu), '--tp-size', '1',
            '--context-length', '16384', '--mem-fraction-static', '.35', '--disable-cuda-graph',
            '--max-running-requests', str(max_running_requests), '--chunked-prefill-size', '2048',
            '--enable-deterministic-inference', '--enable-metrics', '--log-level', 'warning']


async def collect(args, manifest, ports, overall_start, deadline):
    import httpx
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    from interact_env import Action, Decision, Environment, EpisodeSpec
    from interact_env.slime_bridge.generate import encode_decision, apply_completion
    from slime.utils.types import Sample
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    stats = {}
    rollout_started = time.monotonic()
    phase_seconds = (deadline-rollout_started-30)/len(manifest['engine_order'])
    if phase_seconds < 120:
        raise RuntimeError('startup left insufficient profiling time')
    async with httpx.AsyncClient(trust_env=False, timeout=240) as client:
        for engine in manifest['engine_order']:
            phase_start = time.monotonic()
            phase_end = phase_start+phase_seconds
            counter = 0
            stats[engine] = dict(turns=0, generated_requests=0, segments=0, full_episodes=0, failures=0)
            print('PHASE_START', engine, 'seconds', phase_seconds, flush=True)
            async def worker(worker_id):
                nonlocal counter
                while time.monotonic() < phase_end:
                    index = counter
                    counter += 1
                    specs = manifest['scenarios'][engine]
                    spec = EpisodeSpec(**specs[index % len(specs)])
                    env = Environment(spec, args.output/'episodes', timeout=180,
                                      python=str(ROOT.parent/'.venv/bin/python'))
                    row = dict(engine=engine, index=index, task=spec.task_id, episode_id=env.episode_id,
                               server=worker_id % args.gpus, turns=0, status='running')
                    started = time.monotonic()
                    timing = None
                    try:
                        reset_start = time.perf_counter()
                        decision = await env.reset()
                        row['reset_s'] = time.perf_counter()-reset_start
                        while isinstance(decision, Decision) and row['turns'] < manifest['max_turns']:
                            turn_start = time.perf_counter()
                            timing = dict(engine=engine, index=index, episode_id=env.episode_id,
                                          decision_id=decision.decision_id, status='encoding')
                            prompt, ids, images, _ = encode_decision(asdict(decision.observation), tokenizer, processor, timing)
                            if len(ids)+512 > 16384:
                                raise RuntimeError('context overflow; no silent truncation')
                            sampling_group = manifest.get('sampling_group_keys', {}).get(spec.group_key, spec.group_key)
                            seed_key = f"{manifest['seed']}:{sampling_group}:{index}:{decision.decision_id}"
                            seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:4], 'little') % (2**31)
                            payload = dict(text=prompt, image_data=list(decision.observation.images),
                                           return_logprob=True, sampling_params=dict(max_new_tokens=512,
                                           temperature=.8, top_p=1., top_k=-1, sampling_seed=seed,
                                           skip_special_tokens=False))
                            request_start = time.perf_counter()
                            timing.update(status='requesting', prompt_tokens=len(ids), images=len(images),
                                          setup_s=request_start-turn_start-sum(timing[k] for k in
                                              ('image_decode_s','chat_template_s','processor_tokenize_s')))
                            response = await client.post(f'http://127.0.0.1:{ports[worker_id % args.gpus]}/generate', json=payload)
                            response.raise_for_status()
                            output = response.json()
                            timing['http_roundtrip_s'] = time.perf_counter()-request_start
                            turn = apply_completion(Sample(tokens=ids, metadata={}), output)
                            timing.update(completion_tokens=turn.response_length, server={k:v for k,v in output['meta_info'].items()
                                if k in ('e2e_latency','queue_time','prefill_launch_delay','prefill_launch_latency',
                                         'prefill_finished_ts','request_received_ts','request_sent_to_scheduler_ts',
                                         'decode_finished_ts','inference_time','decode_throughput')})
                            stats[engine]['generated_requests'] += 1
                            timing['status'] = 'environment_step'
                            step_start = time.perf_counter()
                            decision = await env.step(Action(env.episode_id, decision.decision_id, turn.response))
                            timing.update(env_step_s=time.perf_counter()-step_start,
                                          total_s=time.perf_counter()-turn_start, status='completed',
                                          elapsed_job_s=time.monotonic()-overall_start)
                            append(args.output/'turns.jsonl', timing)
                            timing = None
                            row['turns'] += 1
                            stats[engine]['turns'] += 1
                        row['status'] = 'prefix_completed' if isinstance(decision, Decision) else 'episode_completed'
                        stats[engine]['segments'] += 1
                        if row['status'] == 'episode_completed':
                            stats[engine]['full_episodes'] += 1
                            row['outcome'] = decision.outcome
                    except asyncio.CancelledError:
                        row['status'] = 'deadline_interrupted'
                        raise
                    except Exception as exc:
                        row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                        stats[engine]['failures'] += 1
                        raise  # fail closed; no automatic retry loops
                    finally:
                        if timing is not None:
                            timing.update(status='incomplete', elapsed_job_s=time.monotonic()-overall_start)
                            append(args.output/'turns.jsonl', timing)
                        await env.close()
                        row['seconds'] = time.monotonic()-started
                        append(args.output/'segments.jsonl', row)
            workers = [asyncio.create_task(worker(i)) for i in range(manifest['concurrency'])]
            try:
                while time.monotonic() < phase_end:
                    for task in workers:
                        if task.done() and task.exception():
                            raise task.exception()
                    progress = dict(phase=engine, elapsed_job_s=time.monotonic()-overall_start,
                                    elapsed_phase_s=time.monotonic()-phase_start, gpus=args.gpus, stats=stats)
                    (args.output/'progress.json').write_text(json.dumps(progress)+'\n')
                    print('PROGRESS '+json.dumps(progress), flush=True)
                    await asyncio.sleep(min(15, max(.01, phase_end-time.monotonic())))
            finally:
                for task in workers:
                    task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
            stats[engine]['phase_seconds'] = time.monotonic()-phase_start
            print('PHASE_END', engine, json.dumps(stats[engine]), flush=True)
    return dict(stats=stats, rollout_seconds=time.monotonic()-rollout_started)


def main(args):
    import httpx
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['INTERACT_PROFILE'] = '1'
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert args.gpus in (2,4) and args.seconds == {2:3600,4:1800}[args.gpus]
    servers, ports, logs = [], [], []
    deadline = start+args.seconds-90
    def interrupted(*_):
        raise KeyboardInterrupt('allocation interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = dict(status='failed', gpus=args.gpus, cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
                  host=os.uname().nodename, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  sampling='fixed per-scenario/attempt/turn seeds; batching may affect exact trajectories')
    sampler = None
    try:
        log = (args.output/'gpu-utilization.csv').open('w')
        logs.append(log)
        sampler = subprocess.Popen(['nvidia-smi', '--id='+os.environ['CUDA_VISIBLE_DEVICES'],
            '--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw',
            '--format=csv,noheader,nounits', '-l', '5'], stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        for gpu in range(args.gpus):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0))
                port = sock.getsockname()[1]
            ports.append(port)
            log = (args.output/f'server-{gpu}.log').open('w')
            logs.append(log)
            servers.append(subprocess.Popen(server_command(port,gpu), stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True))
        ready = set()
        with httpx.Client(trust_env=False, timeout=2) as client:
            while len(ready) < args.gpus and time.monotonic() < min(start+600,deadline):
                for i, server in enumerate(servers):
                    if server.poll() is not None:
                        raise RuntimeError(f'server {i} exited {server.returncode}')
                    if i not in ready:
                        try:
                            if client.get(f'http://127.0.0.1:{ports[i]}/health').status_code == 200:
                                ready.add(i)
                                print('SERVER_READY', i, 'elapsed', time.monotonic()-start, flush=True)
                        except httpx.HTTPError:
                            pass
                time.sleep(1)
        if len(ready) != args.gpus:
            raise RuntimeError('server startup deadline')
        result['server_startup_s'] = time.monotonic()-start
        result.update(asyncio.run(collect(args,manifest,ports,start,deadline)), status='completed')
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        owned = servers+([sampler] if sampler else [])
        for process in owned:
            if process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
        cleanup_deadline = time.monotonic()+25
        for process in owned:
            try:
                process.wait(timeout=max(.1,cleanup_deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()
        result['elapsed_s'] = time.monotonic()-start
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print('RESULT '+json.dumps(result),flush=True)
        from report_scaling import comparison
        (args.output/'comparison_at_finish.json').write_text(json.dumps(comparison(),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--seconds', type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        main(args)
