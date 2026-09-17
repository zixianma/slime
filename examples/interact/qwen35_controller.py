"""Job-scoped queue for compatibility checks and RL; owns and reaps workers."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    job = Path(os.environ['INTERACT_JOB_DIR'])
    queue = job / 'queue'
    queue.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 7050
    stopped = False
    statuses = []
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def stage(name, argv, timeout):
        start = time.monotonic()
        print(f'STAGE_START {name}', flush=True)
        with (job / f'{name}.log').open('x') as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                until = min(deadline, start + timeout)
                while child.poll() is None and not stopped and time.monotonic() < until:
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
        (job / f'{name}.status.json').write_text(json.dumps(result, indent=2)+'\n')
        print(f'STAGE_END {result}', flush=True)

    try:
        stage('00-runtime', ['bash', 'examples/interact/setup_qwen35_runtime.sh'], 1500)
        done = set()
        idle_since = time.monotonic()
        while not stopped and time.monotonic() < deadline:
            if (queue / 'FINISH').exists():
                break
            pending = [p for p in sorted(queue.glob('*.json')) if p.name not in done]
            if not pending:
                # No unattended idle allocation beyond a bounded setup handoff.
                if time.monotonic() - idle_since > 600:
                    print('NO_STAGE_FOR_600_SECONDS', flush=True)
                    break
                time.sleep(2)
                continue
            for path in pending:
                data = json.loads(path.read_text())
                stage(path.stem, data['argv'], data.get('timeout', 1200))
                done.add(path.name)
                idle_since = time.monotonic()
                if stopped or time.monotonic() >= deadline:
                    break
    finally:
        (job / 'controller-status.json').write_text(json.dumps(statuses, indent=2)+'\n')
    return 0 if statuses and all(s['returncode'] == 0 for s in statuses) else 1


if __name__ == '__main__':
    raise SystemExit(main())
