from examples.interact.screensim.prepare_gemini_split import prompt_rows


def test_prompt_rows_preserve_split_and_gemini_spec():
    scenarios = [
        {"id": "train-a", "split": "train", "spec": {"engine": "screensim", "config": {"human": "gemini"}}},
        {"id": "val-a", "split": "validation", "spec": {"engine": "screensim", "config": {"human": "gemini"}}},
    ]
    manifest = {"split_version": "gemini-v1", "scenarios": scenarios}
    rows = prompt_rows(manifest, "validation")
    assert len(rows) == 1
    assert rows[0]["metadata"]["scenario_id"] == "val-a"
    assert rows[0]["metadata"]["episode"]["config"]["human"] == "gemini"


def test_prompt_rows_preserve_explicit_persona():
    manifest = {
        "split_version": "gemini-classic-novice-v1",
        "scenarios": [{
            "id": "train-a",
            "split": "train",
            "spec": {
                "engine": "screensim",
                "config": {"human": "gemini", "persona": "classic_novice"},
            },
        }],
    }
    rows = prompt_rows(manifest, "train")
    assert rows[0]["metadata"]["episode"]["config"]["persona"] == "classic_novice"
