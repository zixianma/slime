import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.interact import wandb_resume as resume
from slime.observability import wandb_utils


@pytest.fixture
def args(tmp_path):
    load = tmp_path / "parent"
    load.mkdir()
    (load / "latest_checkpointed_iteration.txt").write_text("1")
    (load / "iter_0000001").mkdir()
    (load / "iter_0000001" / ".metadata").touch()
    values = {key: f"test-{key}" for key in resume.CONFIG_KEYS}
    values.update(load=str(load), save=str(tmp_path / "next"),
                  wandb_team="zixianma", wandb_project="interact-slime-rl",
                  wandb_run_id=None, use_wandb=True, wandb_mode="online",
                  wandb_key=None, wandb_random_suffix=False, wandb_group="new-job",
                  wandb_dir=None, rank=0)
    return SimpleNamespace(**values)


def fake_api(args, rows=None):
    if rows is None:
        rows = [{"train/step": i, "train/grad_norm": .5, "train/success": .6,
                 "rollout/step": i} for i in range(2)] + [{"eval/step": 2, "eval/success": .7}]
    run = SimpleNamespace(state="finished", config={k: getattr(args, k) for k in resume.CONFIG_KEYS},
                          scan_history=Mock(return_value=rows))
    return SimpleNamespace(run=Mock(return_value=run))


def test_checkpoint_defaults_to_canonical_curve(args):
    api = fake_api(args)
    resume.configure_tracking(args, {}, api)
    api.run.assert_called_once_with(resume.CANONICAL_PATH)
    assert args.wandb_run_id == "d0df345ba8f8"
    assert args.interact_resume_completed_updates == 2


def test_cooking_history_accepts_one_based_episode_metrics():
    rows = [
        {"train/step": 0, "train/grad_norm": 1.0, "eval/step": 0},
        {"train/step": 1, "train/grad_norm": 1.0, "train/success": 0.8},
        {"train/step": 2, "train/grad_norm": 1.0, "train/success": 0.9},
        {"train/step": 3, "train/success": 0.9, "eval/step": 2},
    ]
    resume.validate_history(rows, 3, success_one_based=True)


def test_cooking_resume_replays_missing_boundary_eval(args):
    from pathlib import Path
    load = Path(args.load)
    (load / "latest_checkpointed_iteration.txt").write_text("2")
    (load / "iter_0000002").mkdir()
    (load / "iter_0000002" / ".metadata").touch()
    args.interact_engine = "cooking"
    args.eval_interval = 3
    rows = [
        {"train/step": i, "train/grad_norm": .5, "rollout/step": i}
        for i in range(3)
    ] + [
        {"train/step": i + 1, "train/success": .6}
        for i in range(3)
    ] + [{"eval/step": 0, "eval/success": .65}]
    resume.configure_tracking(args, {}, fake_api(args, rows))
    assert args.interact_resume_completed_updates == 3
    assert args.interact_resume_eval_step == 2

    rows.append({"eval/step": 2, "eval/success": .8})
    args.wandb_run_id = None
    resume.configure_tracking(args, {}, fake_api(args, rows))
    assert args.interact_resume_eval_step is None


def test_parent_receipt_is_reused_and_explicit_destination_wins(args):
    from pathlib import Path
    (Path(args.load) / "wandb_run.json").write_text(json.dumps(
        dict(entity=args.wandb_team, project=args.wandb_project, run_id="saved")))
    api = fake_api(args)
    resume.configure_tracking(args, {}, api)
    assert args.wandb_run_id == "saved"
    args.wandb_run_id = None
    resume.configure_tracking(args, {"INTERACT_WANDB_RUN_PATH": resume.CANONICAL_PATH}, api)
    assert args.wandb_run_id == "d0df345ba8f8"


@pytest.mark.parametrize("kind", ["running", "config", "missing", "ahead", "offline", "wrong-project"])
def test_unsafe_resume_fails_before_launch(args, kind):
    api = fake_api(args)
    env = {}
    if kind == "running":
        api.run.return_value.state = "running"
    elif kind == "config":
        api.run.return_value.config["lr"] = "different"
    elif kind == "missing":
        api.run.side_effect = ValueError("run does not exist")
    elif kind == "ahead":
        api.run.return_value.scan_history.return_value.append({"eval/step": 3})
    elif kind == "offline":
        args.wandb_mode = "offline"
    elif kind == "wrong-project":
        env["INTERACT_WANDB_RUN_PATH"] = "someone/else/run"
    with pytest.raises(ValueError):
        resume.configure_tracking(args, env, api)
    assert args.wandb_run_id is None


@pytest.mark.parametrize("rows", [
    [{"train/step": 0, "train/grad_norm": 1}],
    [{"train/step": 0, "train/grad_norm": 1, "train/success": 1},
     {"train/step": 0, "train/grad_norm": 2}],
    [{"train/step": 0, "train/grad_norm": 1, "train/success": 1},
     {"rollout/step": 1}],
])
def test_missing_conflicting_or_ahead_history_rejected(rows):
    with pytest.raises(ValueError):
        resume.validate_history(rows, 1)


def test_primary_uses_must_resume_keeps_name_and_saves_identity(args, monkeypatch):
    from pathlib import Path
    resume.configure_tracking(args, {}, fake_api(args))
    init = Mock()
    monkeypatch.setattr(wandb_utils.wandb, "init", init)
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id=args.wandb_run_id))
    metrics = Mock()
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", metrics)
    wandb_utils.init_wandb_primary(args)
    kwargs = init.call_args.kwargs
    assert kwargs["id"] == "d0df345ba8f8" and kwargs["resume"] == "must"
    assert "name" not in kwargs and "group" not in kwargs
    assert json.loads(Path(args.interact_tracking_receipt).read_text())["run_id"] == "d0df345ba8f8"
    metrics.assert_any_call("train/*", step_metric="train/step")
    metrics.assert_any_call("eval/*", step_metric="eval/step")


def test_primary_rejects_wrong_returned_id(args, monkeypatch):
    args.wandb_run_id = "intended"
    monkeypatch.setattr(wandb_utils.wandb, "init", Mock())
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="wrong"))
    with pytest.raises(RuntimeError, match="different run ID"):
        wandb_utils.init_wandb_primary(args)


def test_fresh_run_does_not_request_resume(args, monkeypatch):
    init = Mock()
    monkeypatch.setattr(wandb_utils.wandb, "init", init)
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="new"))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", Mock())
    wandb_utils.init_wandb_primary(args)
    assert "resume" not in init.call_args.kwargs and "id" not in init.call_args.kwargs
    assert args.wandb_run_id == "new"


def test_secondary_receives_same_id(args, monkeypatch):
    args.wandb_run_id = "canonical"
    init = Mock()
    monkeypatch.setattr(wandb_utils.wandb, "init", init)
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", Mock())
    wandb_utils.init_wandb_secondary(args)
    assert init.call_args.kwargs["id"] == "canonical"


def test_credential_wrapper_preserves_explicit_interact_destination(tmp_path):
    from examples.interact.with_wandb_env import build_environment
    dotenv = tmp_path / "test.env"
    dotenv.write_text("WANDB_API_KEY=fake-test-key\nWANDB_RUN_ID=unrelated\n")
    env = build_environment(dotenv, {"INTERACT_WANDB_RUN_PATH": resume.CANONICAL_PATH})
    assert env["INTERACT_WANDB_RUN_PATH"] == resume.CANONICAL_PATH
    assert "WANDB_RUN_ID" not in env
