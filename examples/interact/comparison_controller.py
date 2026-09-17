"""Own all three one-GPU workers inside one approved, bounded allocation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
RUNTIME=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/screensim-qwen-compare')
DATA=Path('/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison')


def main():
    job=DATA/('job-'+os.environ['SLURM_JOB_ID'])
    job.mkdir(parents=True,exist_ok=False)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2').split(',')
    if len(visible)!=3: raise RuntimeError(f'expected three allocated devices, got {len(visible)}')
    children=[]; stopped=False; start=time.monotonic()
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for index,gpu in enumerate(visible):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     TOKENIZERS_PARALLELISM='false')
            log=(job/f'model-{index}.log').open('w')
            cmd=[str(RUNTIME/'venv/bin/python'),str(ROOT/'examples/interact/compare_qwen.py'),
                 '--manifest',str(DATA/'manifest.json'),'--model-index',str(index),
                 '--output',str(job/f'model-{index}'),'--seconds','6800']
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((child,log,index))
            print(json.dumps({'event':'START','index':index,'pid':child.pid,'gpu':gpu}),flush=True)
        while any(p.poll() is None for p,_,_ in children) and not stopped and time.monotonic()-start<7000:
            time.sleep(2)
    finally:
        for p,_,_ in children:
            if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
        until=time.monotonic()+45
        while any(p.poll() is None for p,_,_ in children) and time.monotonic()<until: time.sleep(1)
        states=[]
        for p,log,index in children:
            if p.poll() is None: os.killpg(p.pid,signal.SIGKILL)
            code=p.wait();log.close();states.append(dict(index=index,returncode=code))
        (job/'controller-status.json').write_text(json.dumps(states,indent=2)+'\n')
        print(json.dumps({'event':'FINISHED','workers':states,'seconds':time.monotonic()-start}),flush=True)
    return 0 if len(states)==3 and all(s['returncode']==0 for s in states) else 1


if __name__=='__main__': sys.exit(main())
