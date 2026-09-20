from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_handoff_launcher_is_path_and_identity_explicit():
    script = (ROOT / "examples/interact/cooking_gemini/train.sbatch").read_text()
    required = (
        "COOKING_GEMINI_SPLIT", "COOKING_GEMINI_RUN_DIR", "Q35_MODEL",
        "COOKING_WORKER_PYTHON", "COOKING_WANDB_TEAM", "WANDB_ENV_FILE", "PROVIDER_ENV_FILE",
        "COOKING_GEMINI_LOAD", "INTERACT_WANDB_RUN_PATH",
    )
    assert all(name in script for name in required)
    assert "/gpfs/home/" not in script
    assert "WANDB_API_KEY=" not in script and "GOOGLE_API_KEY=" not in script
    assert "--save-interval 1" in script
    assert "--use-checkpoint-opt-param-scheduler" in script


def test_handoff_docs_identify_paid_stochastic_user_and_resume_contract():
    text = (ROOT / "examples/interact/docs/COOKING_GEMINI_HANDOFF.md").read_text()
    assert "paid, stochastic API calls" in text
    assert "first four valid completions" in text
    assert "COOKING_GEMINI_LOAD" in text
    assert "checkpoint metadata, optimizer/scheduler state" in text
