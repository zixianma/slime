"""Bounded native screenshots and independent CPU tokenization before RL."""
import asyncio
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import torch
from transformers import AutoProcessor, AutoTokenizer
from interact_env import Action, Decision, Environment, EpisodeSpec
from interact_env.slime_bridge.generate import encode_decision

ROOT = Path(__file__).resolve().parents[2]
JOB = Path(os.environ['INTERACT_JOB_DIR'])
MODEL = '/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison/models/Qwen3.5-4B'


async def main():
    assert torch.cuda.device_count() == 2
    assert importlib.metadata.version('flash-attn') == '2.8.3'
    assert shutil.disk_usage('/gpfs/scrubbed/zixianma/checkpoints/web').free > 200*1024**3
    from examples.interact.cooking_calibration_budget import split_path, validate_profile
    split = split_path()
    validate_profile(split)
    rows = [json.loads(line) for line in (split/'train.jsonl').read_text().splitlines()][:2]
    rows += [json.loads(line) for line in (split/'calibration_validation.jsonl').read_text().splitlines()]
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    results = []
    for row in rows:
        spec = EpisodeSpec(**row['metadata']['episode'])
        if os.environ.get('COOKING_HARDWARE_ACCEPTANCE') == '1':
            spec = EpisodeSpec(spec.engine, spec.task_id, spec.seed, {**spec.config, 'renderer':'vulkan'})
        env = Environment(spec, JOB/'preflight-episodes', timeout=180,
                          python='/gpfs/home/zixianma/interact/.venv/bin/python')
        try:
            d = await env.reset()
            for _ in range(2):
                assert isinstance(d, Decision)
                prompt, ids, images, inputs = encode_decision(asdict(d.observation), tokenizer, processor)
                assert images and len(ids)+512 <= 16384
                assert inputs['image_grid_thw'].shape[0] == len(images)
                results.append(dict(case=spec.task_id, tick=d.tick, images=len(images), prompt_tokens=len(ids)))
                if _ == 0:
                    d = await env.step(Action(env.episode_id, d.decision_id, '{"text":"","flag":null}'))
        finally:
            await env.close()
    provenance = dict(status='passed', prompts=results,
        split_sha256=hashlib.sha256((split/'manifest.json').read_bytes()).hexdigest(),
        git={name:subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
             for name,path in [('slime',ROOT),('cooking',ROOT.parent/'cook-bench-engine')]},
        packages={name:importlib.metadata.version(name) for name in ['torch','transformers','sglang','ray','wandb','flash-attn']})
    (JOB/'preflight.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance),flush=True)


if __name__ == '__main__':
    asyncio.run(main())
