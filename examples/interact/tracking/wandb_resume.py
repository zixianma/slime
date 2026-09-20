"""Fail-closed, read-only checks before resuming the ScreenSim training curve."""
import json
from pathlib import Path

from examples.interact.tracking.merge_qwen35_wandb import CONFIG_KEYS, points

CANONICAL_PATH = "zixianma/interact-slime-rl/d0df345ba8f8"


def validate_history(rows, completed, *, success_one_based=False):
    """Require complete optimizer/episode history up to, but not beyond, the load."""
    from examples.interact.tracking.merge_qwen35_wandb import points

    expected = set(range(completed))
    expected_success = set(range(1, completed + 1)) if success_one_based else expected
    for metric, metric_expected in (("train/grad_norm", expected),
                                    ("train/success", expected_success)):
        if set(points(rows, metric, "train/step")) != metric_expected:
            raise ValueError(f"W&B/checkpoint mismatch for {metric}; reconcile history before resuming")
    for row in rows:
        train_upper = completed if success_one_based else completed - 1
        for axis, upper in (("train/step", train_upper), ("rollout/step", completed-1),
                            ("eval/step", completed)):
            value = row.get(axis)
            if value is not None and (not isinstance(value, (int, float)) or
                                      not 0 <= value <= upper or int(value) != value):
                raise ValueError(f"W&B contains invalid/ahead-of-checkpoint {axis}: {value}")


def configure_tracking(args, env, api=None):
    """Resolve explicit ID, parent receipt, then this experiment's canonical ID.

    No cloud writes. Called before Ray/GPU workers start. Fresh experiments still
    get a new ID; a numeric Slime checkpoint always requires verified resumption.
    """
    load = Path(args.load)
    marker = load / "latest_checkpointed_iteration.txt"
    receipt = load / "wandb_run.json"
    explicit = env.get("INTERACT_WANDB_RUN_PATH")
    requested_id = getattr(args, "wandb_run_id", None)
    if explicit and requested_id and explicit.rsplit("/", 1)[-1] != requested_id:
        raise ValueError("Conflicting explicit W&B run IDs")
    if not marker.exists():
        if explicit or requested_id or env.get("INTERACT_PARENT_WANDB_RUN"):
            raise ValueError("W&B resume requires a numeric Slime checkpoint, not base HF weights")
        args.interact_tracking_receipt = str(Path(args.save) / "wandb_run.json")
        return
    raw = marker.read_text().strip()
    if not raw.isdigit():
        raise ValueError("W&B resume requires a numeric checkpoint iteration")
    completed = int(raw) + 1
    if not (load / f"iter_{int(raw):07d}" / ".metadata").is_file():
        raise ValueError("Checkpoint metadata is missing")
    if getattr(args, "start_rollout_id", None) not in (None, completed):
        raise ValueError("Explicit start_rollout_id disagrees with checkpoint")
    if not args.use_wandb or args.wandb_mode != "online":
        raise ValueError("ScreenSim continuations require online W&B resume")
    if not explicit and requested_id:
        explicit = f"{args.wandb_team}/{args.wandb_project}/{requested_id}"
    if not explicit and receipt.exists():
        saved = json.loads(receipt.read_text())
        explicit = f"{saved['entity']}/{saved['project']}/{saved['run_id']}"
    target = explicit or CANONICAL_PATH
    parts = target.split("/")
    if len(parts) != 3 or not all(parts) or parts[:2] != [args.wandb_team, args.wandb_project]:
        raise ValueError("Resume destination must match the configured W&B entity/project")
    if api is None:
        import wandb
        api = wandb.Api(timeout=30)
    run = api.run(target)  # Missing run, auth failure, or network failure must stop launch.
    if run.state == "running":
        raise ValueError("Target W&B run is still running; do not start a second primary writer")
    for key in CONFIG_KEYS:
        if key not in run.config or run.config[key] != getattr(args, key, None):
            raise ValueError(f"W&B resume configuration mismatch: {key}")
    rows = list(run.scan_history(page_size=100))
    validate_history(
        rows, completed,
        success_one_based=getattr(args, "interact_engine", None) == "cooking",
    )
    args.wandb_run_id = parts[2]
    args.interact_resume_completed_updates = completed
    eval_steps = set(points(rows, "eval/success", "eval/step"))
    boundary_eval_step = completed - 1
    args.interact_resume_eval_step = (
        boundary_eval_step
        if (getattr(args, "interact_engine", None) == "cooking"
            and args.eval_interval and completed % args.eval_interval == 0
            and boundary_eval_step not in eval_steps)
        else None
    )
    args.interact_tracking_receipt = str(Path(args.save) / "wandb_run.json")
    args.interact_canonical_wandb_path = target
    print(f"WANDB_RESUME_VERIFIED {target} completed_updates={completed}", flush=True)
