"""Check actual saved tensor updates without initializing a trainer or GPU."""
import argparse
import io
import json
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def tensor_at(iteration, suffix):
    directory = args.checkpoints / f"iter_{iteration:07d}"
    metadata = FileSystemReader(directory).read_metadata()
    candidates = [(index, info) for index, info in metadata.storage_data.items()
                  if index.fqn == suffix]
    candidates.sort(key=lambda pair: str(pair[0].offset))
    if not candidates:
        raise ValueError(f"missing checkpoint parameter {suffix}: {list(metadata.state_dict_metadata)[:10]}")
    index, info = candidates[0]
    with (directory / info.relative_path).open("rb") as stream:
        stream.seek(info.offset)
        tensor = torch.load(io.BytesIO(stream.read(info.length)), map_location="cpu", weights_only=True)
    return index.fqn, tensor


results = {}
for label, suffix in (
    ("language", "language_model.decoder.layers.self_attention.linear_qkv.weight"),
    ("vision", "visual.patch_embed.proj.weight"),
):
    name, before = tensor_at(0, suffix)
    _, after = tensor_at(1, suffix)
    diff = (after.float() - before.float()).abs()
    results[label] = {"parameter": name, "shape": list(before.shape),
                      "changed_elements": int(torch.count_nonzero(diff)), "max_abs_change": diff.max().item(),
                      "finite": bool(torch.isfinite(after).all())}
assert results["language"]["changed_elements"] > 0, results
assert results["vision"]["changed_elements"] == 0, results
assert all(result["finite"] for result in results.values()), results
before_dir, after_dir = [args.checkpoints / f"iter_{i:07d}" for i in (0, 1)]
before_meta, after_meta = [FileSystemReader(d).read_metadata() for d in (before_dir, after_dir)]
checked = 0
for index, before_info in before_meta.storage_data.items():
    if not index.fqn.startswith("visual."):
        continue
    tensors = []
    for directory, info in ((before_dir, before_info), (after_dir, after_meta.storage_data[index])):
        with (directory / info.relative_path).open("rb") as stream:
            stream.seek(info.offset)
            tensors.append(torch.load(io.BytesIO(stream.read(info.length)), map_location="cpu", weights_only=True))
    assert torch.equal(*tensors), f"frozen visual weight changed: {index}"
    checked += 1
results["vision_all_shards"] = {"checked": checked, "all_unchanged": True}
args.output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
