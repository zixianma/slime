"""Select a verified hardware-rendering device inside a 2- or 4-GPU job."""
import json
import os
from pathlib import Path
import torch
from examples.interact.cooking_render_probe import run_probe
from examples.interact.cooking_gpu_handoff import canonical_gpu_uuid


def main():
    gpu_count = torch.cuda.device_count()
    assert gpu_count in (2, 4), gpu_count
    root = Path(os.environ['COOKING_RUN_DIR'])/('preflight-'+os.environ['SLURM_JOB_ID'])
    root.mkdir(parents=True, exist_ok=False)
    os.environ['COOKING_BIND_RENDER_WORKERS'] = '1'
    results = []
    # The renderer worker's nested Slurm binding currently addresses GPU 0/1.
    # In a 4-GPU run either leaves at least three policy slots after the
    # placeholder layout is applied.
    for gpu in (0, 1):
        os.environ['COOKING_RENDER_GPU'] = str(gpu)
        os.environ['INTERACT_RENDER_GPU_UUID'] = canonical_gpu_uuid(torch.cuda.get_device_properties(gpu).uuid)
        result = run_probe(str(root/f'gpu{gpu}'), 'vulkan', 4, 2)
        results.append(result)
        (root/'attempts.json').write_text(json.dumps(results,indent=2)+'\n')
        if result['status'] == 'passed':
            selected = dict(renderer_gpu=gpu,
                            policy_gpus=[i for i in range(gpu_count) if i != gpu],
                            renderer_uuid=os.environ['INTERACT_RENDER_GPU_UUID'])
            (root/'selected.json').write_text(json.dumps(selected)+'\n')
            print('RENDER_PREFLIGHT_PASSED', json.dumps(selected), flush=True)
            return
    raise RuntimeError('Neither allocated GPU passed native rendering; refusing to start RL')


if __name__ == '__main__':
    main()
