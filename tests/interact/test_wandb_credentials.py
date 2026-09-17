import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "examples/interact/with_wandb_env.py"
spec = importlib.util.spec_from_file_location("with_wandb_env", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_wandb_credentials_are_loaded(tmp_path):
    source = tmp_path / "test.env"
    source.write_text("WANDB_API_KEY=fake-test-key\nOPENAI_API_KEY=not-for-this-job\nWANDB_PROJECT=baseline\n")
    env = module.build_environment(source, {"PATH": "/bin", "WANDB_PROJECT": "interact-slime-rl",
                                          "WANDB_RUN_ID": "baseline-run", "WANDB_RESUME": "must"})
    assert env["WANDB_API_KEY"] == "fake-test-key"
    assert env["WANDB_PROJECT"] == "interact-slime-rl"
    assert "OPENAI_API_KEY" not in env
    assert "WANDB_RUN_ID" not in env and "WANDB_RESUME" not in env


def test_no_unapproved_wandb_endpoint(tmp_path):
    source = tmp_path / "test.env"
    source.write_text("WANDB_API_KEY=fake-test-key\nWANDB_BASE_URL=https://elsewhere.invalid\n")
    with pytest.raises(ValueError, match="destination approval"):
        module.build_environment(source, {})
