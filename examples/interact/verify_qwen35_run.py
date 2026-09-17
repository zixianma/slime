"""Independent replay and durable checkpoint checks after the owned RL worker."""
import argparse
import io
import json
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader


def read_tensor(directory, info):
    with (directory / info.relative_path).open('rb') as stream:
        stream.seek(info.offset)
        return torch.load(io.BytesIO(stream.read(info.length)), map_location='cpu', weights_only=True)


def check_checkpoints(root):
    directories = sorted(p for p in root.glob('iter_*') if (p / '.metadata').exists())
    if len(directories) < 2:
        raise ValueError('need at least two completed checkpoints to compare updates')
    first, last = directories[0], directories[-1]
    before, after = (FileSystemReader(p).read_metadata() for p in (first, last))
    language = [i for i in before.storage_data
                if 'decoder.layers' in i.fqn and i.fqn.endswith('self_attention.linear_qkv.weight')]
    visual = [i for i in before.storage_data if i.fqn.startswith('model.visual.')]
    if not language or not visual:
        raise ValueError('missing language or vision checkpoint tensors')
    changed = 0
    maximum = 0.
    for index in language:
        a, b = read_tensor(first, before.storage_data[index]), read_tensor(last, after.storage_data[index])
        assert torch.isfinite(b).all(), index
        diff = (a.float()-b.float()).abs()
        changed += torch.count_nonzero(diff).item()
        maximum = max(maximum, diff.max().item())
    assert changed > 0, 'no change in any sampled language-attention shard'
    for index in visual:
        assert torch.equal(read_tensor(first, before.storage_data[index]),
                           read_tensor(last, after.storage_data[index])), index
    return dict(first=str(first), last=str(last), changed_language_elements=changed,
                max_language_change=maximum, frozen_vision_shards_checked=len(visual))


def replay_baseline(directory):
    from transformers import AutoModelForImageTextToText
    data = torch.load(directory / 'rollout-eval_0.pt', map_location='cpu', weights_only=False)
    selected = []
    tasks = set()
    for sample in data['samples']:
        task = sample['metadata']['task_id']
        if task not in tasks:
            selected.append(sample)
            tasks.add(task)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError('baseline audit needs three distinct validation tasks')
    model_path = '/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison/models/Qwen3.5-4B'
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    results = []
    for sample in selected:
        ids = torch.tensor([sample['tokens']], device='cuda:0')
        mm = {k: v.to('cuda:0') for k, v in sample['multimodal_train_inputs'].items()}
        count = sample['response_length']
        def score(inputs):
            with torch.inference_mode():
                logits = model(input_ids=ids, **inputs, logits_to_keep=count+1).logits[0, :-1].float()
                return (logits/.8).log_softmax(-1).gather(-1, ids[0, -count:, None]).flatten().cpu()
        actual = score(mm)
        recorded = torch.tensor(sample['rollout_log_probs'])
        diff = (actual-recorded).abs()
        ablated = score({**mm, 'pixel_values': torch.zeros_like(mm['pixel_values'])})
        results.append(dict(task=sample['metadata']['task_id'], response_tokens=count,
                            mean_abs_logprob_difference=diff.mean().item(),
                            max_abs_logprob_difference=diff.max().item(),
                            image_ablation_logprob_change=(actual-ablated).abs().mean().item(),
                            passed=bool(torch.isfinite(actual).all() and diff.mean() < .1)))
    assert all(r['passed'] for r in results), results
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    result = {'status': 'running'}
    try:
        result['checkpoints'] = check_checkpoints(args.run_dir / 'checkpoints')
        result['baseline_hf_replay'] = replay_baseline(args.run_dir)
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (args.run_dir / 'verification.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
