#!/usr/bin/env python3
"""CookSim/Gemini RL phase, including checkpoint-safe continuations."""

import hashlib
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    from slime.utils.arguments import parse_args
    args = parse_args()
    worker_python = os.environ.get("COOKING_WORKER_PYTHON")
    if not worker_python or not Path(worker_python).is_file():
        raise ValueError("COOKING_WORKER_PYTHON must name the CookBench worker interpreter")
    args.interact["worker_python"] = worker_python
    split = Path(os.environ["COOKING_GEMINI_SPLIT"])
    manifest_path = split / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"].startswith("cooking-single-error-gemini37-")
    human = manifest["human"]
    assert human["kind"] == "engine-native Gemini"
    assert human["model"] == "gemini-3.7-flash"
    assert human["stochastic"] is True and human["paid_api_calls"] is True
    expected_persona = human.get("persona", "baseline")
    assert expected_persona in ("baseline", "classic_novice")
    if expected_persona == "classic_novice":
        timing = manifest.get("timing", {"base_wall_seconds": 1800,
                                          "policy_decision_limit": 200})
        assert timing.get("policy_decision_limit") == args.interact.get("max_decisions") == 200
    assert manifest["reward"]["version"] == "cooking_prevention_turns_v1"
    for name, info in manifest["files"].items():
        data_path = split / f"{name}.jsonl"
        assert hashlib.sha256(data_path.read_bytes()).hexdigest() == info["sha256"]
        for line in data_path.read_text().splitlines():
            assert json.loads(line)["metadata"]["episode"]["config"]["persona"] == expected_persona
            assert json.loads(line)["metadata"]["episode"]["config"]["wall_seconds"] == manifest.get("timing", {}).get("base_wall_seconds", 1800)
    pool = Path(manifest["case_pool"]["path"])
    assert hashlib.sha256(pool.read_bytes()).hexdigest() == manifest["case_pool"]["sha256"]
    assert Path(args.prompt_data).resolve() == (split / "train.jsonl").resolve()
    assert len(args.eval_prompt_data) == 2
    assert Path(args.eval_prompt_data[1]).resolve() == (split / "validation.jsonl").resolve()
    assert args.num_rollout is not None and args.num_rollout > 0
    assert (args.rollout_batch_size, args.n_samples_per_prompt,
            args.global_batch_size) == (6, 4, 24)
    assert args.n_samples_per_eval_prompt == 2 and args.eval_interval == 3
    assert args.tensor_model_parallel_size == 2 and args.actor_num_gpus_per_node == 4
    assert args.actor_num_gpus_per_node // args.tensor_model_parallel_size == 2
    assert args.rollout_function_path == "examples.interact.cooking_gemini.speculative_rollout.generate_rollout"
    assert args.eval_function_path == "slime.rollout.sglang_rollout.generate_rollout"
    assert args.interact.get("spill_visual_inputs") is True
    expected_concurrency = 6 if os.environ.get("COOKING_SHARED_RENDERER") == "1" else 4
    assert (args.sglang_server_concurrency == expected_concurrency and
            args.sglang_max_running_requests == expected_concurrency)
    args.interact_split_version = manifest["version"]
    args.interact_split_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    args.interact_engine = "cooking"
    args.interact_model_revision = os.environ.get(
        "COOKING_MODEL_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    args.interact_human_profile = manifest["human"]
    shared_renderer = os.environ.get("COOKING_SHARED_RENDERER") == "1"
    continuation = (Path(args.load) / "latest_checkpointed_iteration.txt").is_file()
    if continuation:
        load = Path(args.load)
        assert load != Path(args.save)
        latest = (load / "latest_checkpointed_iteration.txt").read_text().strip()
        assert latest.isdigit()
        latest_rollout = int(latest)
        assert 0 <= latest_rollout < args.num_rollout
        assert (load / f"iter_{latest_rollout:07d}" / ".metadata").is_file()
        assert args.use_checkpoint_opt_param_scheduler
        # Every completed update must remain restartable when an allocation ends.
        assert args.save_interval == 1
    args.interact_training_plan = {
        "phase": (f"single-error-gemini-continue-to-{args.num_rollout}-updates" if continuation else
                  f"single-error-gemini-{expected_persona}-fresh-to-{args.num_rollout}-updates"),
        "train_scenarios": manifest["files"]["train"]["count"],
        "validation_scenarios": manifest["files"]["validation"]["count"],
        "groups_per_update": 6, "attempts_per_group": 4,
        "rollout_collection": "launch 5 attempts/group; accept first 4 complete",
        "rollout_server_concurrency": expected_concurrency,
        "episodes_per_update": 24, "validation_attempts": 2,
        "reward": manifest["reward"], "renderer": "vulkan",
        "visual_tensor_storage": "disk-backed; lazy microbatch materialization",
        "rollout_topology": ("GPU0/1 shared renderer + policy; 4 TP1 policy servers"
                             if shared_renderer else
                             "1 dedicated renderer + 3 TP1 policy servers"),
        "learner_topology": "TP2 x DP2 on all 4 GPUs after renderer handoff",
    }
    if os.environ.get("INTERACT_PARSE_ONLY") == "1":
        print("COOKING_GEMINI_ARGS_OK", args.interact_training_plan, flush=True)
        return
    from examples.interact.tracking.wandb_resume import configure_tracking
    configure_tracking(args, os.environ)
    import ray
    from train import train
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        ray.init(address="local", num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")),
                 num_gpus=4, object_store_memory=4 * 1024**3, include_dashboard=False)
        train(args)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
