"""Independent HF replay of a real rollout; diagnostic, not an RL trainer."""
import json
import os
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

job = Path(os.environ["INTERACT_JOB_DIR"])
data_dir = Path(os.environ.get("INTERACT_DATA_DIR", str(job)))
sample = torch.load(data_dir / "rollout.pt", map_location="cpu", weights_only=False)["samples"][0]
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/gpfs/scrubbed/zixianma/checkpoints/web/Qwen2.5-VL-3B-Instruct",
    dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
).to("cuda:0").eval()
ids = torch.tensor([sample["tokens"]], device="cuda:0")
mm = {k: v.to("cuda:0") for k, v in sample["multimodal_train_inputs"].items()}
count = sample["response_length"]


def replay(inputs):
    with torch.inference_mode():
        logits = model(input_ids=ids, **inputs, logits_to_keep=count + 1).logits[0, :-1].float()
        return (logits / 0.8).log_softmax(-1).gather(-1, ids[0, -count:, None]).flatten().cpu()


actual = replay(mm)
recorded = torch.tensor(sample["rollout_log_probs"])
image_ablated = replay({**mm, "pixel_values": torch.zeros_like(mm["pixel_values"])})
diff = (actual - recorded).abs()
result = {"diagnostic": "HF replay, not an optimizer update", "response_tokens": count,
          "mean_abs_logprob_difference": diff.mean().item(), "max_abs_logprob_difference": diff.max().item(),
          "image_ablation_mean_abs_logprob_change": (actual - image_ablated).abs().mean().item(),
          "finite": bool(torch.isfinite(actual).all()), "temperature": 0.8,
          "tolerance": 0.1, "within_mean_tolerance": bool(diff.mean() < 0.1)}
(job / "hf_replay.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
assert result["finite"] and result["within_mean_tolerance"], result
