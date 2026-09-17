import asyncio
import os
from types import SimpleNamespace

import pytest

from interact_env.slime_bridge.generate import apply_completion, generate
from interact_env.slime_bridge.rewards import normalize

CHECKPOINT = os.environ.get("INTERACT_TEST_CHECKPOINT", "/gpfs/scrubbed/zixianma/checkpoints/web/Qwen2.5-VL-3B-Instruct")


def test_exact_server_tokens():
    from slime.utils.types import Sample
    turn = Sample(tokens=[1, 2, 3])
    output = dict(text="response", meta_info=dict(finish_reason={"type": "stop"},
                  prompt_tokens=3, completion_tokens=2, output_token_logprobs=[[-0.1, 42], [-0.2, 151645]]))
    apply_completion(turn, output)
    assert turn.tokens == [1, 2, 3, 42, 151645]
    assert turn.rollout_log_probs == [-0.1, -0.2] and turn.loss_mask == [1, 1]
    for override in ({"output_token_logprobs": []}, {"finish_reason": {"type": "abort"}}, {"prompt_tokens": 8}):
        with pytest.raises(RuntimeError):
            apply_completion(Sample(tokens=[1, 2, 3]), {**output, "meta_info": {**output["meta_info"], **override}})


def make_samples():
    from slime.utils.types import Sample
    return [Sample(group_index=0, rollout_id=0 if i == 0 else 1, index=i, reward=float(i > 0),
                   tokens=[1, 2], response_length=1,
                   metadata={"group_key": "cooking:fixture", "reward_version": "test",
                             "turn_index": 0 if i == 0 else i - 1, "num_turns": 1 if i == 0 else 3})
            for i in range(4)]


def test_official_slime_conversion_counts_episodes_once():
    from slime.ray.rollout import RolloutManager
    cls = RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(cls)
    manager.args = SimpleNamespace(reward_key=None, advantage_estimator="grpo", rewards_normalization=True,
                                   grpo_std_normalization=True, rollout_top_p=1.0)
    manager.custom_reward_post_process_func = normalize
    manager.custom_convert_samples_to_train_data_func = None
    data = manager._convert_samples_to_train_data(make_samples())
    assert data["rollout_ids"] == [0, 1, 1, 1]
    assert data["rollout_mask_sums"] == [1, 3, 3, 3]
    assert data["rewards"][0] == pytest.approx(-data["rewards"][1])
    assert data["rewards"][1:] == [data["rewards"][1]] * 3


def test_normalization_rejects_mixed_groups_and_incomplete_episodes():
    args = SimpleNamespace(reward_key=None, advantage_estimator="grpo")
    samples = make_samples()
    samples[1].metadata["group_key"] = "vh:other"
    with pytest.raises(ValueError, match="mixed engine"):
        normalize(args, samples)
    with pytest.raises(ValueError, match="incomplete"):
        normalize(args, make_samples()[:-1])


@pytest.mark.parametrize('evaluation', [False, True])
def test_complete_official_hook_mocking_only_inference(tmp_path, monkeypatch, evaluation):
    import json
    monkeypatch.setenv('INTERACT_PROFILE', '1')
    from transformers import AutoTokenizer
    from slime.rollout import sglang_rollout
    from slime.utils import http_utils
    from slime.utils.types import Sample
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: SimpleNamespace(
        tokenizer=tokenizer, processor=None, aborted=False))
    text = '{"text":"","flag":null}'
    completion = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)
    calls = []

    async def fake_post(url, payload, **kwargs):
        calls.append(payload)
        return dict(text=text, meta_info=dict(finish_reason={"type": "stop"},
                    prompt_tokens=len(payload["input_ids"]), completion_tokens=len(completion),
                    weight_version="mock-only", output_token_logprobs=[[-0.1, t] for t in completion]))
    monkeypatch.setattr(http_utils, "post", fake_post)
    args = SimpleNamespace(interact={"output_root": str(tmp_path), 'fixed_eval_sampling_seed': 20260913}, rollout_max_context_len=16384,
                           sglang_router_ip="127.0.0.1", sglang_router_port=1)
    parent = Sample(group_index=7, index=8, rollout_id=99, metadata={"episode": {
        "engine": "cooking", "task_id": "b3_easy_nops_medium_map_2",
        "config": {"reward_version": "assistant_detection_v1"}}})
    turns = asyncio.run(generate(args, parent, {"max_new_tokens": 384}, evaluation=evaluation))
    assert len(turns) == len(calls) == 76
    assert {t.rollout_id for t in turns} == {99}
    assert {t.group_index for t in turns} == {7}
    assert len({t.index for t in turns}) == len(turns)
    assert {t.reward for t in turns} == {0.5}
    for turn, payload in zip(turns, calls, strict=True):
        assert turn.tokens == payload["input_ids"] + completion
        assert turn.metadata["num_turns"] == len(turns)
    assert len(list(tmp_path.glob("*/training_audit.json"))) == 1
    timing = [json.loads(line) for line in next(tmp_path.glob('*/rollout_timing.jsonl')).read_text().splitlines()]
    assert timing[0]['phase'] == 'reset'
    assert len(timing) == len(turns)+1
    for row in timing[1:]:
        fields = ('image_decode_s','chat_template_s','processor_tokenize_s','request_setup_s',
                  'http_roundtrip_s','completion_apply_s','env_step_s')
        assert all(row[k] >= 0 for k in fields)
        assert sum(row[k] for k in fields) == pytest.approx(row['total_turn_s'], abs=.005)
    worker_timing = [json.loads(line) for line in next(tmp_path.glob('*/worker_timing.jsonl')).read_text().splitlines()]
    assert len(worker_timing) == len(turns)
    assert all(('sampling_seed' in c['sampling_params']) == evaluation for c in calls)
    if evaluation:
        assert len({c['sampling_params']['sampling_seed'] for c in calls}) == len(calls)
