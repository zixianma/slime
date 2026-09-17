"""Approved two-hour real learner acceptance; no automatic retries."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
ALLOCATION_SECONDS = 7200
CLEANUP_SECONDS = 180
EXTRA_ARGS = ['--rollout-num-gpus', '2', '--sglang-config',
              'examples/interact/configs/cooking_gpu_layout.yaml', '--custom-config-path',
              'examples/interact/configs/cooking_hardware_acceptance.yaml']


def stages(checkpoint):
    return [
        ('00-preflight', ['python','examples/interact/cooking_calibration_preflight.py'], 300, {}),
        ('01-kernels', ['bash','examples/interact/qwen35_kernel_stage.sh'], 180, {}),
        ('02-train', ['bash','examples/interact/run_cooking_calibration.sh']+EXTRA_ARGS, 5700, {}),
        ('03-restore', ['bash','examples/interact/run_cooking_calibration.sh']+EXTRA_ARGS+
         ['--load',checkpoint,'--use-checkpoint-opt-param-scheduler'], 600, {'COOKING_RESTORE_ONLY':'1'}),
        ('04-verify', ['python','examples/interact/cooking_calibration_verify.py'], 120, {}),
    ]


def main():
    from examples.interact.cooking_calibration_budget import validate_profile
    validate_profile()
    job = Path(os.environ['INTERACT_JOB_DIR'])
    checkpoint = f"/gpfs/scrubbed/zixianma/checkpoints/web/slime-cooking-{os.environ['SLURM_JOB_ID']}/calibration01/checkpoints"
    plan = stages(checkpoint)
    assert sum(s[2] for s in plan)+CLEANUP_SECONDS <= ALLOCATION_SECONDS
    deadline = time.monotonic()+ALLOCATION_SECONDS-CLEANUP_SECONDS
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    statuses = []
    for i, (name, command, limit, env) in enumerate(plan):
        if stopping or time.monotonic() >= deadline:
            break
        start = time.monotonic()
        cutoff = min(start+limit, deadline-sum(s[2] for s in plan[i+1:]))
        print('STAGE_START', name, flush=True)
        with (job/f'{name}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **env},
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while child.poll() is None and not stopping and time.monotonic() < cutoff:
                    time.sleep(1)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        status = dict(name=name, returncode=child.returncode, seconds=time.monotonic()-start,
                      deadline_reached=time.monotonic()>=cutoff)
        statuses.append(status)
        (job/f'{name}.status.json').write_text(json.dumps(status)+'\n')
        (job/'controller-status.json').write_text(json.dumps(statuses,indent=2)+'\n')
        print('STAGE_END', json.dumps(status), flush=True)
        if child.returncode:
            break
    return 0 if len(statuses)==len(plan) and all(s['returncode']==0 for s in statuses) else 1


if __name__ == '__main__':
    raise SystemExit(main())
