"""Complete-episode cache, bound to policy update, task and sampling contract."""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import torch
from slime.utils.multimodal_storage import FILE_KEY


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(args, sample, spec, sampling_params):
    root = Path(os.environ['COOKING_RUN_DIR'])
    assert args.global_batch_size == 48 and isinstance(sample.index, int)
    update = sample.index//48
    assert sample.group_index//6 == update
    policy = {'completed_updates':update, 'base_revision':args.interact_model_revision}
    if update:
        # These audit snapshots are tied to a committed model + sampler checkpoint.
        state = json.loads((root/'resume-state.json').read_text())
        assert state['completed_updates'] == update
        policy['audits'] = [sha(root/'audit'/f'weights-update{update}-rank{rank}.json') for rank in (0,1)]
        policy['checkpoint_metadata'] = sha(root/'checkpoints'/f'iter_{update-1:07d}'/'.metadata')
    source_root = Path(__file__).resolve().parents[3]
    sources = ['interact_env/slime_bridge/generate.py', 'interact_env/adapters/cooking.py',
               'interact_env/worker.py', 'examples/interact/cooking_gemini/cooking_episode_cache.py']
    return dict(version=1, policy=policy, episode=asdict(spec), index=sample.index,
        group_index=sample.group_index, sampling=sampling_params,
        decision_limit=int(args.interact['max_decisions']),
        split=args.interact_split_sha256, experiment=sha(root/'experiment.json'),
        sources={name:sha(source_root/name) for name in sources})


class EpisodeCache:
    def __init__(self, root, contract):
        self.contract = json.loads(json.dumps(contract, sort_keys=True))
        key = hashlib.sha256(json.dumps(self.contract, sort_keys=True).encode()).hexdigest()
        self.path = Path(root)/'episode-cache'/f'{key}.pt'

    def validate(self, turns):
        assert turns, 'empty episode cache'
        group_key = hashlib.sha256(json.dumps(self.contract['episode'], sort_keys=True).encode()).hexdigest()
        episode_ids = set()
        rewards = set()
        versions = set()
        for i, turn in enumerate(turns):
            meta = turn.metadata
            assert meta['num_turns'] == len(turns) and meta['turn_index'] == i
            assert not meta['evaluation'] and meta['group_key'] == group_key
            assert turn.rollout_id == self.contract['index'] and turn.group_index == self.contract['group_index']
            assert turn.index == (self.contract['index'] << 16)|i
            assert turn.reward in (0.,1.) and meta['outcome']
            assert turn.response_length == len(turn.rollout_log_probs) == len(turn.loss_mask) > 0
            assert all(math.isfinite(v) for v in turn.rollout_log_probs)
            assert len(turn.tokens) > turn.response_length
            episode_ids.add(meta['episode_id'])
            rewards.add(turn.reward)
            versions.update(turn.weight_versions)
            if turn.multimodal_train_inputs is not None:
                assert set(turn.multimodal_train_inputs) == {FILE_KEY}, 'cache requires disk-backed visuals'
                assert Path(turn.multimodal_train_inputs[FILE_KEY]).is_file(), 'missing visual tensor file'
        assert len(episode_ids) == len(rewards) == len(versions) == 1

    def load(self):
        if not self.path.exists():
            return None
        data = torch.load(self.path, map_location='cpu', weights_only=False)
        assert data['contract'] == self.contract, 'episode cache identity mismatch'
        self.validate(data['turns'])
        print(f'COOKING_EPISODE_CACHE_HIT index={self.contract["index"]}', flush=True)
        return data['turns']

    def save(self, turns):
        self.validate(turns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        torch.save(dict(contract=self.contract, turns=turns), temporary)
        temporary.replace(self.path)
