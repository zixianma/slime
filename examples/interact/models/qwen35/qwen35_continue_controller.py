"""Own every continuation worker; fail closed and release the bounded allocation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from examples.interact.models.qwen35.qwen35_continue_profile import ROOT, PARENT, ORDER, TARGET


def main():
    job = Path(os.environ['INTERACT_JOB_DIR'])
    deadline = time.monotonic() + 10620
    stopped = False
    statuses = []
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stages = [
        ('00-preflight', ['python', 'examples/interact/models/qwen35/qwen35_continue_preflight.py'], 180),
        ('01-kernels', ['bash', 'examples/interact/models/qwen35/qwen35_kernel_stage.sh'], 180),
        ('02-train', ['bash', 'examples/interact/models/qwen35/run_qwen35_train.sh',
            '--load', str(PARENT/'checkpoints'), '--use-checkpoint-opt-param-scheduler',
            '--prompt-data', str(ORDER), '--num-rollout', str(TARGET), '--eval-interval', '3',
            '--skip-eval-before-train',
            '--eval-function-path', 'examples.interact.models.qwen35.qwen35_continue_eval.generate',
            '--custom-eval-rollout-log-function-path', 'examples.interact.models.qwen35.qwen35_continue_eval.log'], 10080),
        ('03-verify', ['python', 'examples/interact/models/qwen35/qwen35_continue_verify.py'], 180),
    ]
    try:
        for name, argv, limit in stages:
            if stopped or time.monotonic() >= deadline:
                break
            start = time.monotonic()
            print('STAGE_START', name, flush=True)
            with (job/f'{name}.log').open('x') as out:
                child = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                try:
                    while child.poll() is None and not stopped and time.monotonic() < min(deadline, start+limit):
                        time.sleep(1)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            result = dict(name=name, returncode=child.returncode, seconds=time.monotonic()-start)
            statuses.append(result)
            (job/f'{name}.status.json').write_text(json.dumps(result)+'\n')
            print('STAGE_END', result, flush=True)
            if child.returncode != 0:
                break
    finally:
        (job/'controller-status.json').write_text(json.dumps(statuses, indent=2)+'\n')
    return 0 if len(statuses) == len(stages) and all(s['returncode'] == 0 for s in statuses) else 1


if __name__ == '__main__':
    raise SystemExit(main())
