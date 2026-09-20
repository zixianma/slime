"""Real SGLang + official Slime hook + native ScreenSim. No optimizer claim."""
import asyncio
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import httpx
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from interact_env.slime_bridge.generate import generate
from interact_env.slime_bridge.rewards import normalize
from slime.utils.types import Sample
from slime.utils.http_utils import init_http_client

job = Path(os.environ["INTERACT_JOB_DIR"])
data_dir = Path(os.environ.get("INTERACT_DATA_DIR", str(job)))
data_dir.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--spec", action="append")
parser.add_argument("--samples", type=int, default=2)
parser.add_argument("--tag", default="rollout")
parser.add_argument("--checkpoint", default="/gpfs/scrubbed/zixianma/checkpoints/web/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--multimodal-request-format", choices=['ids', 'text'], default='ids')
parser.add_argument("--deterministic", action='store_true')
options = parser.parse_args()
if not options.tag.replace("-", "").replace("_", "").isalnum() or not 2 <= options.samples <= 8:
    raise ValueError("invalid artifact tag or sample count")
checkpoint = options.checkpoint
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", checkpoint,
       "--host", "127.0.0.1", "--port", str(port), "--base-gpu-id", "0", "--tp-size", "1",
       "--context-length", "16384", "--mem-fraction-static", "0.4", "--disable-cuda-graph",
       "--max-running-requests", "2", "--chunked-prefill-size", "2048", "--log-level", "warning"]
if options.deterministic:
    cmd.append('--enable-deterministic-inference')
server_log_path = job / f"sglang-{int(time.time())}.log"
server_log = server_log_path.open("w")
server = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)


def stop(*_):
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()
    except ProcessLookupError:
        pass
    server_log.close()


signal.signal(signal.SIGTERM, lambda *_: (stop(), sys.exit(143)))


async def rollout():
    args = SimpleNamespace(
        hf_checkpoint=checkpoint, rollout_num_engines=1, sglang_server_concurrency=2,
        use_distributed_post=False, sglang_router_ip="127.0.0.1", sglang_router_port=port,
        rollout_temperature=0.8, rollout_top_p=1.0, rollout_top_k=-1,
        rollout_max_response_len=512, rollout_stop=None, rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False, sglang_dp_size=1, rollout_max_context_len=16384,
        reward_key=None, advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=True,
        sglang_enable_deterministic_inference=options.deterministic,
        rollout_seed=1234, n_samples_per_prompt=options.samples,
        interact={"output_root": str(data_dir / "episodes"), "inference_timeout": 240,
                  "worker_python": str(ROOT.parent / '.venv/bin/python'),
                  "multimodal_request_format": options.multimodal_request_format})
    init_http_client(args)
    from slime.rollout.sglang_rollout import GenerateState
    state = GenerateState(args)
    trajectories = []
    groups = []
    for group_index, path in enumerate(options.spec or ["examples/interact/configs/screensim.json"]):
        episode = json.loads((ROOT / path).read_text())
        samples = [Sample(index=group_index * options.samples + i, rollout_id=group_index * options.samples + i,
                          group_index=group_index, metadata={"episode": episode}) for i in range(options.samples)]
        group = await asyncio.gather(*(generate(args, s, {
            **state.sampling_params,
            **({'sampling_seed': 1234+s.index} if options.deterministic else {})}) for s in samples))
        trajectories.extend(group)
        groups.append({"episode": episode, "rewards": [t[0].reward for t in group]})
    flat = [s for t in trajectories for s in t]
    raw, advantages = normalize(args, flat)
    torch.save({"samples": [s.to_dict() for s in flat], "advantages": advantages}, data_dir / f"{options.tag}.pt")
    summary = {"trajectories": len(trajectories), "turns": [len(t) for t in trajectories],
               "rewards": [t[0].reward for t in trajectories], "nonzero_advantages": any(a != 0 for a in advantages),
               "optimizer_updates": 0, "human": "scripted_human_v1", "groups": groups,
               "sampling": "unconstrained; not the original vLLM JSON-mode baseline"}
    (job / f"{options.tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    import slime.utils.http_utils as http_utils
    await http_utils._http_client.aclose()


try:
    deadline = time.monotonic() + 360
    with httpx.Client(trust_env=False) as client:
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"SGLang exited {server.returncode}; see {server_log_path}")
            try:
                if client.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("SGLang startup timed out")
    print("SGLANG_READY", flush=True)
    asyncio.run(rollout())
finally:
    stop()
