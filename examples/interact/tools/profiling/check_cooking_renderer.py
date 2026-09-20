"""Matched scripted cooking observations, software vs hardware; no model calls."""
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from interact_env import Action, Decision, Environment, EpisodeSpec


async def main(output):
    import numpy as np
    from PIL import Image
    output.mkdir(parents=True, exist_ok=False)
    os.environ['INTERACT_PROFILE'] = '1'
    manifest = json.loads((ROOT/'examples/interact/configs/scaling_profile_v1.json').read_text())
    report = {'mode':'matched_scripted_actions_no_policy', 'host':os.uname().nodename,
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'observations':{}}
    for renderer in ('swiftshader', 'vulkan'):
        raw = manifest['scenarios']['cooking'][0]
        spec = EpisodeSpec(**{**raw, 'config':{**raw['config'], 'renderer':renderer}})
        env = Environment(spec, output/'episodes', timeout=180, python=str(ROOT.parent/'.venv/bin/python'))
        rows = []
        report['observations'][renderer] = rows
        try:
            start = time.monotonic()
            decision = await env.reset()
            for i in range(6):
                assert isinstance(decision, Decision), 'Unexpected early episode end'
                row = {'index':i, 'tick':decision.tick, 'elapsed_s':time.monotonic()-start,
                       'prompt':decision.observation.user, 'images':[]}
                for j, url in enumerate(decision.observation.images):
                    data = base64.b64decode(url.split(',',1)[1])
                    path = output/f'{renderer}-{i}-{j}.png'
                    im = Image.open(io.BytesIO(data)).convert('RGB')
                    im.save(path)
                    pixels = np.asarray(im).astype(float)
                    w = im.width//2
                    # Composite images have 26px labels and a 2px separator.
                    left = pixels[26:,:w-1]
                    right = pixels[26:,w+1:w+1+left.shape[1]]
                    row['images'].append({'path':str(path), 'size':im.size,
                        'brightness':float(pixels.mean()), 'std':float(pixels.std()),
                        'pane_difference':float(np.abs(left-right).mean())})
                rows.append(row)
                print(renderer, i, decision.tick, row['images'], flush=True)
                if i < 5:
                    start = time.monotonic()
                    decision = await env.step(Action(env.episode_id, decision.decision_id, '{"text":""}'))
        finally:
            await env.close()
            report[renderer+'_episode'] = str(env.directory)
            (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    pairs = []
    for a,b in zip(report['observations']['swiftshader'],report['observations']['vulkan']):
        pairs.append({'index':a['index'], 'tick_match':a['tick']==b['tick'],
                      'prompt_match':a['prompt']==b['prompt'], 'image_count_match':len(a['images'])==len(b['images'])})
    report['pairs'] = pairs
    report['status'] = 'captured_requires_visual_review'
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    asyncio.run(main(Path(sys.argv[1])))
