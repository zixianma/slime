import pickle
import pytest
import torch
from slime.utils.multimodal_storage import store_multimodal, materialize_multimodal


def test_lossless_lazy_visual_roundtrip(tmp_path):
    original = {'pixel_values': torch.randn(200,32), 'image_grid_thw': torch.tensor([[1,10,20]])}
    lazy = store_multimodal(original, tmp_path/'visual.pt')
    assert len(pickle.dumps(lazy)) < 1024
    restored = materialize_multimodal(pickle.loads(pickle.dumps(lazy)), 'cpu')
    assert all(torch.equal(original[k], restored[k]) and original[k].dtype == restored[k].dtype for k in original)
    assert materialize_multimodal(original, 'cpu') is original
    assert materialize_multimodal(None, 'cpu') is None
    with pytest.raises(AssertionError, match='mixed'):
        materialize_multimodal({**lazy, 'pixel_values': original['pixel_values']}, 'cpu')
