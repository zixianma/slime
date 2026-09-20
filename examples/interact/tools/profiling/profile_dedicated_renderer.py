"""Two isolated GPU steps: one Chromium renderer, one rollout policy. No RL updates."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

from profile_scaling import ROOT, collect, server_command


def allocation():
    ids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).split()
    if len(ids) != 1:
        raise RuntimeError(f'Expected one isolated GPU in this step, got {ids}')
    return {'gpu_uuids':ids, 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'host':os.uname().nodename, 'cpus':os.environ.get('SLURM_CPUS_PER_TASK')}


def main(args):
    start = time.monotonic()
    directory = args.output/args.role
    directory.mkdir(parents=True, exist_ok=False)
    own = allocation()
    (directory/'allocation.json').write_text(json.dumps(own)+'\n')
    print('ALLOCATION', args.role, json.dumps(own), flush=True)
    os.environ['INTERACT_PROFILE'] = '1'
    children = []
    logs = []
    def stopped(*_):
        raise KeyboardInterrupt('job step stopped')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    result = {'status':'failed', 'allocation':own, 'allocated_gpus_total':2,
              'policy_gpus':1, 'renderer_gpus':1, 'optimizer_updates':0,
              'max_running_requests':args.max_running_requests}
    try:
        log = (directory/'gpu-utilization.csv').open('w')
        logs.append(log)
        children.append(subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits','-l','5'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        if args.role == 'policy':
            log = (directory/'server.log').open('w')
            logs.append(log)
            command = server_command(args.port, 0, args.max_running_requests)
            (directory/'server_command.json').write_text(json.dumps(command)+'\n')
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(process)
            code = process.wait()
            if code:
                raise RuntimeError(f'policy server exited {code}')
        else:
            import httpx
            with httpx.Client(trust_env=False, timeout=2) as client:
                while time.monotonic()-start < 600:
                    try:
                        ready = client.get(f'http://127.0.0.1:{args.port}/health').status_code == 200
                    except httpx.HTTPError:
                        ready = False
                    if ready:
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError('policy startup deadline')
            policy = json.loads((args.output/'policy/allocation.json').read_text())
            if set(own['gpu_uuids']) & set(policy['gpu_uuids']):
                raise RuntimeError('GPU isolation failed: renderer and policy share a device')
            result['policy_allocation'] = policy
            result['server_wait_s'] = time.monotonic()-start
            manifest = json.loads((ROOT/'examples/interact/configs/scaling_profile_v1.json').read_text())
            manifest['engine_order'] = ['cooking']
            manifest['max_running_requests'] = args.max_running_requests
            from interact_env import EpisodeSpec
            manifest['sampling_group_keys'] = {}
            for raw in manifest['scenarios']['cooking']:
                original_group = EpisodeSpec(**raw).group_key
                raw['config']['renderer'] = 'vulkan'
                manifest['sampling_group_keys'][EpisodeSpec(**raw).group_key] = original_group
            manifest['mode'] = 'dedicated_hardware_renderer_live_policy_16_turn_prefix_not_RL'
            (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            settings = SimpleNamespace(output=directory, gpus=1)
            result.update(asyncio.run(collect(settings, manifest, [args.port], start, start+args.seconds-90)))
        result['status'] = 'completed'
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic()+20
        for child in children:
            try:
                child.wait(timeout=max(.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for log in logs:
            log.close()
        result['elapsed_s'] = time.monotonic()-start
        (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print('RESULT', args.role, json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['policy','renderer'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--max-running-requests', type=int, choices=[4, 8], default=4)
    main(parser.parse_args())
