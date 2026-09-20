import json
from dataclasses import asdict
from types import SimpleNamespace
import pytest
import torch
from interact_env.protocol import EpisodeSpec, Observation
from examples.interact.cooking_gemini.cooking_recover_episodes import recover, digest, manifest
from slime.utils.multimodal_storage import materialize_multimodal


def test_recovery_preserves_on_policy_sample_and_rejects_changes(tmp_path, monkeypatch):
    root = tmp_path/'episode'
    root.mkdir()
    spec = EpisodeSpec('cooking', 'fixture', config={'renderer':'vulkan'})
    obs = Observation('system', 'user')
    metadata = dict(evaluation=False, turn_index=0, num_turns=1, group_key=spec.group_key,
                    outcome='won', prompt_hash=obs.content_hash, finish_reason='stop')
    files = {'spec.json':asdict(spec), 'report.json':{'outcome':'won'},
        'training_audit.json':[dict(metadata=metadata, rollout_id=0, weight_versions=['1'],
            reward=1., response_length=1, tokens=[10,20], rollout_log_probs=[-.5],loss_mask=[1])]}
    for name, value in files.items():
        (root/name).write_text(json.dumps(value))
    (root/'decisions.jsonl').write_text(json.dumps(dict(decision={'observation':asdict(obs)},
        prompt_hash=obs.content_hash,response='answer'))+'\n')
    checkpoint = str(tmp_path/'base')
    contents = dict(source_job=296235, model=checkpoint,model_revision='fixture',episodes={
        '0':dict(directory=str(root),hashes={n:digest(root/n) for n in (*files,'decisions.jsonl')})})
    manifest_path = tmp_path/'manifest.json'
    manifest_path.write_text(json.dumps(contents))
    monkeypatch.setenv('COOKING_RECOVERY_MANIFEST',str(manifest_path))
    tensor = torch.tensor([[1.,2.]])
    monkeypatch.setattr('interact_env.slime_bridge.generate.encode_decision',
        lambda *a:('prompt',[10],[],{'pixel_values':tensor}))
    args = SimpleNamespace(interact_resume_completed_updates=0,hf_checkpoint=checkpoint,
                           load=checkpoint,interact_model_revision='fixture')
    sample = SimpleNamespace(index=0, group_index=0)
    state = SimpleNamespace(tokenizer=None,processor=None)
    result = recover(args,sample,spec,state)
    assert result[0].tokens == [10,20] and result[0].rollout_log_probs == [-.5]
    assert result[0].response == 'answer' and result[0].reward == 1.
    assert torch.equal(materialize_multimodal(result[0].multimodal_train_inputs,'cpu')['pixel_values'],tensor)
    args.interact_resume_completed_updates=1
    assert recover(args,sample,spec,state) is None
    args.interact_resume_completed_updates=0
    # A checkpoint-bound receipt uses explicit policy hashes, not runtime weight
    # version numbers alone (those reset when the inference service restarts).
    policy_file = tmp_path/'policy-audit.json'
    policy_file.write_text('{"policy":1}')
    contents.update(completed_updates=1,checkpoint_root=checkpoint,
                    policy_hashes={str(policy_file):digest(policy_file)})
    manifest_path.write_text(json.dumps(contents))
    manifest.cache_clear()
    args.interact_resume_completed_updates=1
    assert recover(args,sample,spec,state)[0].tokens == [10,20]
    policy_file.write_text('{"policy":2}')
    with pytest.raises(AssertionError,match='checkpoint identity changed'):
        recover(args,sample,spec,state)
    policy_file.write_text('{"policy":1}')
    (root/'report.json').write_text('{"outcome":"wrong_serve"}')
    with pytest.raises(AssertionError,match='source changed'):
        recover(args,sample,spec,state)
    manifest.cache_clear()
