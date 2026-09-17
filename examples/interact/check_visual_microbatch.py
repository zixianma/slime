"""Exercise the real Megatron microbatch loader on one allocated CUDA device."""
from pathlib import Path
from unittest.mock import patch
import os
import torch
from slime.backends.megatron_utils import data
from slime.utils.multimodal_storage import store_multimodal, FILE_KEY


def main():
    device = torch.device('cuda:0')
    root = Path(os.environ['COOKING_RUN_DIR'])/('preflight-'+os.environ['SLURM_JOB_ID'])
    original = torch.arange(24, dtype=torch.float32).reshape(6,4)
    lazy = store_multimodal({'pixel_values':original},root/'microbatch-visual.pt')
    records = dict(tokens=[torch.tensor([1,2,3],device=device),None],
        loss_masks=[torch.tensor([1],device=device),None], total_lengths=[3,None],
        response_lengths=[1,None],multimodal_train_inputs=[lazy,{FILE_KEY:'/must-not-load.pt'}])
    iterator = data.DataIterator(records, [[0]])
    with patch.object(data.mpu,'get_tensor_model_parallel_world_size',return_value=2), \
         patch.object(data.mpu,'get_context_parallel_world_size',return_value=1), \
         patch.object(data.mpu,'get_context_parallel_rank',return_value=0):
        batch = data.get_batch(iterator, list(records))
    actual = batch['multimodal_train_inputs']['pixel_values']
    assert actual.device == device and torch.equal(actual.cpu(),original)
    assert records['multimodal_train_inputs'][0] == lazy
    (root/'microbatch-visual-passed.txt').write_text('GPU lazy microbatch materialization passed\n')
    print('GPU_VISUAL_MICROBATCH_PASSED',flush=True)


if __name__ == '__main__':
    main()
