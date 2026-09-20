from copy import deepcopy

import pytest

from examples.interact.screensim.prepare_rl_split import OLD_VALIDATION, build_split, prompt_rows


def source_manifest():
    tasks = {task: (1 if task == "rent_share_standard" else
                    2 if task == "rent_landlord_instant" else 3)
             for task in OLD_VALIDATION}
    tasks.update({task: 3 for task in ["hearing_setup", "lost_phone_lockdown",
                  "app_privacy_quarantine", "parent_phone_readability",
                  "stop_profiling", "keyboard_stop_correcting", "standby_bedside"]})
    return dict(engine_revision="pinned", profile="scripted_human_v1", persona="baseline",
                models=[], decoding={}, scenarios=[
                    dict(id=f"{task}__c{i+1}", split="validation" if task in OLD_VALIDATION else "train",
                         attempts=8 if task in OLD_VALIDATION else 16,
                         spec=dict(engine="screensim", task_id=task, seed=0,
                                   config=dict(episode_index=i)))
                    for task, count in tasks.items() for i in range(count)])


def test_whole_task_split_preserves_specs_and_source():
    source = source_manifest()
    original = deepcopy(source)
    result = build_split(source, "hash")
    assert source == original
    train, val = (prompt_rows(result, split) for split in ("train", "validation"))
    assert (len(train), len(val)) == (18, 12)
    tasks = lambda rows: {r["metadata"]["episode"]["task_id"] for r in rows}
    assert (len(tasks(train)), len(tasks(val))) == (6, 5)
    assert not tasks(train) & tasks(val)
    assert tasks(val) == OLD_VALIDATION | {"hearing_setup"}
    assert [s["spec"] for s in result["scenarios"]] == [s["spec"] for s in original["scenarios"]]
    assert all("attempts" not in s for s in result["scenarios"])


def test_reject_duplicate_and_changed_source():
    source = source_manifest()
    source["scenarios"][1]["id"] = source["scenarios"][0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        build_split(source, "hash")
    source = source_manifest()
    source["scenarios"][0]["split"] = "train"
    with pytest.raises(ValueError, match="source"):
        build_split(source, "hash")
