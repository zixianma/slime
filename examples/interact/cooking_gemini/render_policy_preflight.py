"""Verify that native CookSim rendering and SGLang can safely share one GPU."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import httpx

from examples.interact.cooking_gemini.cooking_render_probe import run_probe
from examples.interact.tools.profiling.profile_scaling import server_command


def terminate(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=25)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    def interrupted(signum, _frame):
        raise TimeoutError(f"preflight interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run_root = Path(os.environ["COOKING_RUN_DIR"])
    root = run_root / ("preflight-" + os.environ["SLURM_JOB_ID"])
    selected = json.loads((root / "selected.json").read_text())
    gpu = int(selected["renderer_gpu"])
    target = root / "shared-renderer-policy"
    target.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "failed",
        "renderer_gpu": gpu,
        "renderer_uuid": selected["renderer_uuid"],
        "policy_requests": 24,
    }
    server = sampler = None
    handles = []
    started = time.monotonic()
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = server_command(port, gpu, max_running_requests=6)
        (target / "server-command.json").write_text(json.dumps(command) + "\n")
        server_log = (target / "server.log").open("w")
        handles.append(server_log)
        server = subprocess.Popen(
            command, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True
        )
        gpu_log = (target / "gpu-utilization.csv").open("w")
        handles.append(gpu_log)
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--id=" + selected["renderer_uuid"],
                "--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "-l",
                "2",
            ],
            stdout=gpu_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 480
        with httpx.Client(trust_env=False, timeout=2) as client:
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"shared policy server exited {server.returncode}")
                try:
                    if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("shared policy server startup deadline")
        result["server_startup_s"] = time.monotonic() - started
        renderer_probe = run_probe(
            str(target / "renderer-24-probe"), "vulkan", concurrency=24, turns=1,
        )
        result["renderer_24_probe"] = renderer_probe
        if renderer_probe["status"] != "passed":
            raise RuntimeError("24-way shared-GPU renderer stress probe failed")
        probe = run_probe(
            str(target / "policy-6-probe"), "vulkan", concurrency=6, turns=4,
            policy_url=f"http://127.0.0.1:{port}",
        )
        result["probe"] = probe
        if probe["status"] != "passed":
            raise RuntimeError("shared renderer/policy stress probe failed")
        if server.poll() is not None:
            raise RuntimeError(f"shared policy server exited after probe: {server.returncode}")
        result["status"] = "passed"
        result["elapsed_s"] = time.monotonic() - started
        print("SHARED_RENDER_POLICY_PREFLIGHT_PASSED " + json.dumps(result), flush=True)
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_s"] = time.monotonic() - started
        print("SHARED_RENDER_POLICY_PREFLIGHT_FAILED " + json.dumps(result), flush=True)
        raise
    finally:
        terminate(server)
        terminate(sampler)
        for handle in handles:
            handle.close()
        (target / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
