"""Own all stages inside the approved allocation; execute local queued stages.

Queue files contain explicit argv arrays, not shell source. The controller exits
on a FINISH sentinel or before its one-hour allocation expires.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
job = Path(os.environ["INTERACT_JOB_DIR"])
job.mkdir(parents=True, exist_ok=True)
queue = Path(os.environ.get("INTERACT_QUEUE_DIR", str(job / "queue")))
queue.mkdir(parents=True, exist_ok=True)
done = set()
deadline = time.monotonic() + int(os.environ.get("INTERACT_CONTROLLER_SECONDS", "3500"))


def stage(name, argv, timeout):
    print(f"STAGE_START {name}: {argv}", flush=True)
    started = time.time()
    with (job / f"{name}.log").open("w") as log:
        proc = subprocess.Popen(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = {"returncode": proc.wait(timeout=max(1, min(timeout, int(deadline - time.monotonic()))))}
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            status = {"error": "timeout", "returncode": 124}
    status.update(name=name, elapsed=time.time() - started)
    (job / f"{name}.status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"STAGE_END {status}", flush=True)
    return status["returncode"]


code = stage("preflight", [sys.executable, "examples/interact/screensim_preflight.py"], 900)
if code:
    sys.exit(code)
if os.environ.get("INTERACT_SMALL_INTIME_TEST") == "1":
    # Fresh checkpoint, three homogeneous GRPO groups, four episodes per group.
    # Save only the final checkpoint; no automatic extension or second allocation.
    sys.exit(stage("train-intime", ["bash", "examples/interact/run_screensim_train.sh",
        "--prompt-data", "examples/interact/configs/screensim_intime_tasks.jsonl",
        "--num-rollout", "3", "--n-samples-per-prompt", "4", "--global-batch-size", "4",
        "--save-interval", "3"], 1200))
if os.environ.get("INTERACT_INTERACTIVE_QUEUE") != "1":
    for name, argv, timeout in (
        ("rollout", [sys.executable, "examples/interact/screensim_rollout_probe.py"], 600),
        ("logprob_replay", [sys.executable, "examples/interact/check_rollout_logprobs.py"], 240),
        ("train", ["bash", "examples/interact/run_screensim_train.sh"], 1500),
    ):
        code = stage(name, argv, timeout)
        if code:
            sys.exit(code)
    sys.exit(0)
while time.monotonic() < deadline:
    if (queue / "FINISH").exists():
        break
    for path in sorted(queue.glob("*.json")):
        if path.name in done:
            continue
        data = json.loads(path.read_text())
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            break
        stage(path.stem, data["argv"], min(data.get("timeout", 1200), remaining))
        done.add(path.name)
    time.sleep(2)
print("CONTROLLER_FINISHED", flush=True)
