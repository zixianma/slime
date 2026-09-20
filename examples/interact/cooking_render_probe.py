"""Native rendering stress probe; scripted no-op responses, no RL data."""
import asyncio
import json
import os
from pathlib import Path
import time


def run_probe(output, renderer='vulkan', concurrency=4, turns=4, policy_url=None):
    from interact_env import Environment, EpisodeSpec, Decision, Action
    root = Path(__file__).resolve().parents[2]
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root/'examples/interact/configs/scaling_profile_v1.json').read_text())
    if policy_url:
        from transformers import AutoProcessor, AutoTokenizer
        from dataclasses import asdict
        from interact_env.slime_bridge.generate import encode_decision
        import httpx
        model = os.environ.get("Q35_MODEL", manifest["model"])
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    async def worker(i):
        raw = manifest['scenarios']['cooking'][i%4]
        spec = EpisodeSpec(**{**raw, 'config':{**raw['config'], 'renderer':renderer}})
        from interact_env.slime_bridge.generate import cooking_worker_prefix
        worker_python = os.environ.get("COOKING_WORKER_PYTHON", str(root.parent/'.venv/bin/python'))
        env = Environment(spec, target/'episodes', timeout=120, python=worker_python,
                          launch_prefix=cooking_worker_prefix('cooking'))
        row = dict(worker=i, task=spec.task_id, episode_id=env.episode_id, turns=0)
        start = time.monotonic()
        try:
            d = await env.reset()
            for _ in range(turns):
                assert isinstance(d, Decision)
                assert d.observation.images
                import base64
                (target/f'frame-{i}-{row["turns"]}.png').write_bytes(base64.b64decode(d.observation.images[-1].split(',',1)[1]))
                response = '{"text":"","flag":null}'
                if policy_url:
                    prompt, _, _, _ = encode_decision(asdict(d.observation), tokenizer, processor)
                    async with httpx.AsyncClient(trust_env=False, timeout=240) as client:
                        result = await client.post(policy_url+'/generate', json=dict(text=prompt,
                            image_data=list(d.observation.images), sampling_params=dict(
                                max_new_tokens=512, temperature=.8, top_p=1, top_k=-1)))
                        result.raise_for_status()
                        generated = result.json()
                        assert generated['meta_info']['completion_tokens'] > 0
                        response = generated['text']
                row['turns'] += 1
                d = await env.step(Action(env.episode_id,d.decision_id,response))
            row['status'] = 'passed'
        except Exception as exc:
            row.update(status='failed',error=str(exc))
        finally:
            await env.close()
        row['seconds'] = time.monotonic()-start
        return row
    async def collect():
        return await asyncio.gather(*(worker(i) for i in range(concurrency)))
    rows = asyncio.run(collect())
    result = dict(renderer=renderer, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  expected_gpu=os.environ.get('INTERACT_RENDER_GPU_UUID'),rows=rows,
                  status='passed' if all(r['status']=='passed' for r in rows) else 'failed')
    (target/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    import sys
    print(json.dumps(run_probe(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))),flush=True)
