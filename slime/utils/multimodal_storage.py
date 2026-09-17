"""Opt-in disk-backed visual tensors for long, complete-episode rollouts."""
from pathlib import Path
import torch

FILE_KEY = '__slime_multimodal_tensor_file__'


def store_multimodal(tensors, path):
    if tensors is None:
        return None
    assert tensors and all(isinstance(v, torch.Tensor) and v.device.type == 'cpu'
                           for v in tensors.values())
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(tensors, temporary)
    temporary.replace(path)
    return {FILE_KEY: str(path)}


def materialize_multimodal(value, device):
    if value is None or FILE_KEY not in value:
        return value
    assert set(value) == {FILE_KEY}, 'mixed lazy and eager visual inputs'
    tensors = torch.load(value[FILE_KEY], map_location='cpu', weights_only=True)
    assert tensors and all(isinstance(v, torch.Tensor) for v in tensors.values())
    return {k: v.to(device=device) for k, v in tensors.items()}
