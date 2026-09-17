"""Own and reap every stage; the prepared seven-hour rerun is not yet approved."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from cooking_calibration_budget import ALLOCATION_SECONDS, CLEANUP_SECONDS, STAGE_LIMITS, validate_profile, stage_deadline

ROOT = Path(__file__).resolve().parents[2]


def main():
    validate_profile()
    job = Path(os.environ['INTERACT_JOB_DIR'])
    checkpoint = f"/gpfs/scrubbed/zixianma/checkpoints/web/slime-cooking-{os.environ['SLURM_JOB_ID']}/calibration01/checkpoints"
    stages = [
        ('00-preflight',['python','examples/interact/cooking_calibration_preflight.py'],300,{}),
        ('01-kernels',['bash','examples/interact/qwen35_kernel_stage.sh'],180,{}),
        ('02-train',['bash','examples/interact/run_cooking_calibration.sh'],STAGE_LIMITS[2],{}),
        ('03-restore',['bash','examples/interact/run_cooking_calibration.sh','--load',checkpoint,
                       '--use-checkpoint-opt-param-scheduler'],600,{'COOKING_RESTORE_ONLY':'1'}),
        ('04-verify',['python','examples/interact/cooking_calibration_verify.py'],180,{})]
    assert tuple(s[2] for s in stages) == STAGE_LIMITS
    deadline = time.monotonic()+ALLOCATION_SECONDS-CLEANUP_SECONDS
    stopping = False
    def stop(*unused):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    statuses=[]
    try:
        for name,argv,limit,env in stages:
            if stopping or time.monotonic() >= deadline:
                break
            start=time.monotonic()
            print('STAGE_START',name,flush=True)
            with (job/f'{name}.log').open('x') as out:
                child=subprocess.Popen(argv,cwd=ROOT,env={**os.environ,**env},stdout=out,
                                       stderr=subprocess.STDOUT,start_new_session=True)
                # Reserve all subsequent stage budgets, even after a slow train.
                cutoff = stage_deadline(deadline, start, limit,
                                        [s[2] for s in stages[len(statuses)+1:]])
                try:
                    while child.poll() is None and not stopping and time.monotonic()<cutoff:
                        time.sleep(1)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            status=dict(name=name,returncode=child.returncode,seconds=time.monotonic()-start,
                        deadline_reached=time.monotonic()>=cutoff)
            statuses.append(status)
            (job/f'{name}.status.json').write_text(json.dumps(status)+'\n')
            print('STAGE_END',status,flush=True)
            if child.returncode:
                break
    finally:
        (job/'controller-status.json').write_text(json.dumps(statuses,indent=2)+'\n')
    return 0 if len(statuses)==len(stages) and all(s['returncode']==0 for s in statuses) else 1


if __name__ == '__main__':
    raise SystemExit(main())
