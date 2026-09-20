"""Recover complete audited episodes from a specifically pinned policy.

No partial trajectories or validation episodes are eligible for recovery.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import os


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def manifest(path):
    return json.loads(Path(path).read_text())


def recover(args, sample, spec, state):
    path = os.environ.get('COOKING_RECOVERY_MANIFEST')
    if not path:
        return None
    data = manifest(path)
    if data.get('max_turns') is not None:
        assert int(args.interact['max_decisions']) == int(data['max_turns']), 'recovery decision cap mismatch'
    completed = data.get('completed_updates', 0)
    if getattr(args, 'interact_resume_completed_updates', -1) != completed:
        return None
    entry = data['episodes'].get(str(sample.index))
    if entry is None:
        return None
    assert str(Path(args.hf_checkpoint).resolve()) == data['model']
    expected_load = data.get('checkpoint_root', data['model'])
    assert str(Path(args.load).resolve()) == expected_load, 'recovery policy checkpoint mismatch'
    for name, expected in data.get('policy_hashes', {}).items():
        assert digest(name) == expected, 'recovery checkpoint identity changed'
    assert args.interact_model_revision == data['model_revision']
    root = Path(entry['directory'])
    for name, expected in entry['hashes'].items():
        assert digest(root/name) == expected, f'recovery source changed: {root/name}'
    from dataclasses import asdict
    from interact_env.protocol import Observation
    from interact_env.slime_bridge.generate import encode_decision
    from slime.utils.types import Sample
    from slime.utils.multimodal_storage import store_multimodal
    assert json.loads((root/'spec.json').read_text()) == asdict(spec)
    audits = json.loads((root/'training_audit.json').read_text())
    report = json.loads((root/'report.json').read_text())
    assert audits and len(audits) == audits[0]['metadata']['num_turns']
    turns = []
    with (root/'decisions.jsonl').open() as stream:
        for i, (audit, line) in enumerate(zip(audits, stream, strict=True)):
            recorded = json.loads(line)
            meta = audit['metadata']
            assert not meta['evaluation'] and meta['turn_index'] == i
            assert meta['group_key'] == spec.group_key and meta['outcome'] == report['outcome']
            assert audit['rollout_id'] == sample.index
            assert audit['weight_versions'] == [data.get('weight_version', '1')]
            assert audit['reward'] == float(report['outcome'] == 'won')
            obs = recorded['decision']['observation']
            assert Observation(**obs).content_hash == meta['prompt_hash'] == recorded['prompt_hash']
            prompt, ids, _, tensors = encode_decision(obs, state.tokenizer, state.processor)
            n = audit['response_length']
            assert audit['tokens'][:-n] == ids
            assert n == len(audit['rollout_log_probs']) == len(audit['loss_mask'])
            assert meta['finish_reason'] in ('stop', 'length')
            lazy = store_multimodal(tensors, root/'visual_tensors'/f'{i}.pt')
            turn = Sample(group_index=sample.group_index, index=(sample.index << 16)|i,
                rollout_id=sample.index, prompt=prompt, response=recorded['response'],
                tokens=audit['tokens'], response_length=n, multimodal_train_inputs=lazy,
                reward=audit['reward'], rollout_log_probs=audit['rollout_log_probs'],
                loss_mask=audit['loss_mask'], weight_versions=audit['weight_versions'],
                metadata={**meta, 'recovered_from_job':data['source_job']},
                status=Sample.Status.COMPLETED if meta['finish_reason']=='stop' else Sample.Status.TRUNCATED)
            turns.append(turn)
            if (i+1) % 25 == 0:
                print(f'COOKING_RECOVERY_PROGRESS episode={sample.index} turns={i+1}/{len(audits)}', flush=True)
    print(f'COOKING_RECOVERED episode={sample.index} turns={len(turns)} source={root.name}', flush=True)
    return turns


def build(root, output, since, completed='0', source_job='296235', max_turns=None):
    import datetime
    threshold = datetime.datetime.fromisoformat(since).timestamp()
    root = Path(root)
    completed = int(completed)
    source_job = int(source_job)
    assert (completed, source_job) in ((0,296235), (1,296322), (1,297329)), 'unverified recovery source'
    max_turns = int(max_turns) if max_turns is not None else None
    policy_hashes = {}
    if completed:
        assert int((root/'checkpoints/latest_checkpointed_iteration.txt').read_text()) == completed-1
        assert json.loads((root/'resume-state.json').read_text())['completed_updates'] == completed
        policy_paths = [root/'checkpoints'/f'iter_{completed-1:07d}'/'.metadata']
        policy_paths += [root/'audit'/f'weights-update{completed}-rank{r}.json' for r in (0,1)]
        policy_hashes = {str(p):digest(p) for p in policy_paths}
    else:
        assert not (root/'checkpoints/latest_checkpointed_iteration.txt').exists()
    contract = json.loads((root/'experiment.json').read_text())
    assert contract['base_model'] == 'Qwen3.5-4B'
    entries = {}
    for audit_path in sorted((root/'episodes').glob('*/training_audit.json')):
        if audit_path.stat().st_mtime < threshold:
            continue
        audits = json.loads(audit_path.read_text())
        assert audits and not audits[0]['metadata']['evaluation']
        if max_turns is not None and len(audits) > max_turns:
            continue
        index = audits[0]['rollout_id']
        assert completed*48 <= index < (completed+1)*48 and str(index) not in entries
        assert all(a['rollout_id']==index and a['weight_versions']==[str(completed+1)] for a in audits)
        directory = audit_path.parent
        names = ('training_audit.json', 'decisions.jsonl', 'report.json', 'spec.json')
        entries[str(index)] = dict(directory=str(directory), hashes={n:digest(directory/n) for n in names})
    assert entries
    result = dict(source_job=source_job, completed_updates=completed, weight_version=str(completed+1),
        max_turns=max_turns,
        policy_hashes=policy_hashes, model='/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison/models/Qwen3.5-4B',
        model_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', episodes=entries)
    if completed:
        result['checkpoint_root'] = str((root/'checkpoints').resolve())
    with Path(output).open('x') as stream:
        json.dump(result, stream, indent=2)
    print(f'Recovery manifest: {len(entries)} complete episodes; {output}')


if __name__ == '__main__':
    import sys
    build(*sys.argv[1:])
