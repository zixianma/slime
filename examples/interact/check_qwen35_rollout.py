"""Replay a real native rollout with the independent HF checkpoint on one GPU."""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText

job = Path(os.environ['INTERACT_JOB_DIR'])
data = torch.load(job/'qwen35-native.pt', map_location='cpu', weights_only=False)
samples = [s for s in data['samples'] if s['metadata']['turn_index'] == 0]
assert len(samples) == 2
model = AutoModelForImageTextToText.from_pretrained(
    '/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison/models/Qwen3.5-4B',
    local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
results = []
for sample in samples:
    count = min(128, sample['response_length'])
    prompt_len = len(sample['tokens']) - sample['response_length']
    ids = torch.tensor([sample['tokens'][:prompt_len+count]], device='cuda:0')
    mm = {k: v.to('cuda:0') for k,v in sample['multimodal_train_inputs'].items()}
    def score(inputs):
        with torch.inference_mode():
            logits = model(input_ids=ids, **inputs, logits_to_keep=count+1).logits[0, :-1].float()
            return (logits/.8).log_softmax(-1).gather(-1, ids[0,-count:,None]).flatten().cpu()
    actual = score(mm)
    recorded = torch.tensor(sample['rollout_log_probs'][:count])
    ablated = score({**mm, 'pixel_values': torch.zeros_like(mm['pixel_values'])})
    diff = (actual-recorded).abs()
    results.append(dict(episode_id=sample['metadata']['episode_id'], response_tokens=count,
        mean_abs_logprob_difference=diff.mean().item(), max_abs_logprob_difference=diff.max().item(),
        image_ablation_logprob_change=(actual-ablated).abs().mean().item(),
        passed=bool(torch.isfinite(actual).all() and diff.mean()<.1)))
(job/'qwen35_replay.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results), flush=True)
assert all(r['passed'] for r in results), results
