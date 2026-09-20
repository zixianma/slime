"""Audit the official actor's update without changing its optimizer or loss."""
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from slime.backends.megatron_utils.actor import MegatronTrainRayActor


def snapshot(models):
    result = {}
    for i, model in enumerate(models):
        for name, p in model.named_parameters():
            if '.visual.' in name or name.endswith('self_attention.linear_qkv.weight'):
                t = p.detach().cpu().contiguous()
                assert torch.isfinite(t).all(), name
                result[f'{i}:{name}'] = hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert any('.visual.' in k for k in result)
    assert any('linear_qkv' in k for k in result)
    return result


class CookingAuditActor(MegatronTrainRayActor):
    def verify_restored_weights(self, completed=None):
        from examples.interact.archive.cooking_calibration.cooking_checkpoint_audit import verify
        return verify(self, completed, snapshot, dist.get_rank)

    def train_actor(self, rollout_id, rollout_data, external_data=None):
        before = snapshot(self.model)
        super().train_actor(rollout_id, rollout_data, external_data=external_data)
        after = snapshot(self.model)
        assert before.keys() == after.keys()
        visual = [k for k in before if '.visual.' in k]
        language = [k for k in before if 'linear_qkv' in k]
        assert all(before[k] == after[k] for k in visual)
        changed = sum(before[k] != after[k] for k in language)
        assert changed > 0, 'language weights did not change'
        result = dict(rank=dist.get_rank(), rollout_id=rollout_id,
                      frozen_visual_parameters=len(visual), changed_language_parameters=changed,
                      before=before, after=after)
        (Path(os.environ['INTERACT_JOB_DIR'])/f'weights-rank{dist.get_rank()}.json').write_text(json.dumps(result)+'\n')
        if os.environ.get('COOKING_FULL_TRAINING') == '1':
            (Path(os.environ['INTERACT_JOB_DIR'])/f'weights-update{rollout_id+1}-rank{dist.get_rank()}.json').write_text(json.dumps(result)+'\n')
