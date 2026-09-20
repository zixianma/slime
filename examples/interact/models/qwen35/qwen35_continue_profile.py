"""Versioned, explicitly approved continuation plans."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ORDER = ROOT/'interact-runs/qwen35-rl-292972/train-round-robin.jsonl'
NAME = os.environ.get('Q35_CONTINUATION_PROFILE', 'continue09')
if NAME == 'continue09':
    PARENT = Path('/gpfs/scrubbed/zixianma/checkpoints/web/slime-qwen35-292972/pilot09')
    VERIFICATION = PARENT/'verification.json'
    PARENT_WANDB = '6neavc52'
    START, TARGET, EVAL_STEPS = 2, 9, [6, 9]
    SAMPLER = dict(sample_offset=12, epoch_id=0, sample_group_index=12, sample_index=96)
elif NAME == 'continue15':
    PARENT = Path('/gpfs/scrubbed/zixianma/checkpoints/web/slime-qwen35-293612/continue09')
    VERIFICATION = ROOT/'interact-runs/qwen35-continue-293612/verification.json'
    PARENT_WANDB = '6kv7biu2'
    START, TARGET, EVAL_STEPS = 9, 15, [12, 15]
    SAMPLER = dict(sample_offset=18, epoch_id=2, sample_group_index=54, sample_index=432)
else:
    raise ValueError(f'unknown continuation profile: {NAME}')
