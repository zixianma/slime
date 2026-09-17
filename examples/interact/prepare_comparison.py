"""Pin/download public checkpoints and certify a frozen ScreenSim split on CPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEST = Path('/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison')
VALIDATION = {'lock_screen_lockdown', 'siri_lock_screen_privacy',
              'rent_share_standard', 'rent_landlord_instant'}
MODELS = ['Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct', 'Qwen/Qwen3.5-4B']


def main():
    from huggingface_hub import HfApi, snapshot_download
    sys.path.insert(0, str(ROOT.parent / 'screensim-engine'))
    from screensim.tasks import HARD_TASKS, by_id
    from screensim.interact.composite import design_composite_episodes
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / 'manifest.json'
    if target.exists():
        raise FileExistsError('preserve the frozen manifest; use it rather than overwriting')
    models = []
    for name in MODELS:
        revision = ('66285546d2b821cf421d4f5eb2576359d3770cd3' if '2.5' in name
                    else HfApi().model_info(name).sha)
        local = (DEST.parent / 'Qwen2.5-VL-3B-Instruct' if '2.5' in name
                 else DEST / 'models' / name.split('/')[1])
        snapshot_download(name, revision=revision, local_dir=local,
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'],
                          max_workers=4)
        models.append(dict(id=name, revision=revision, path=str(local)))
        print(json.dumps(models[-1]), flush=True)
    scenarios = []
    for tid in HARD_TASKS:
        _, episodes = design_composite_episodes(by_id(tid), per_task=3)
        for index, episode in enumerate(episodes):
            assert episode.ok
            split = 'validation' if tid in VALIDATION else 'train'
            spec = dict(engine='screensim', task_id=tid, seed=0, config=dict(
                human='scripted', persona='baseline', observation='frames', episode_index=index,
                k_frames=2, reward_version='screensim_intime_success_v1'))
            scenarios.append(dict(id=episode.id, split=split, attempts=8 if split=='validation' else 16,
                                  spec=spec))
    assert len(scenarios)==30 and sum(s['attempts'] for s in scenarios)==408
    manifest = dict(models=models, scenarios=scenarios,
                    engine_revision=subprocess.check_output(['git','-C',str(ROOT.parent/'screensim-engine'),
                                                             'rev-parse','HEAD'],text=True).strip(),
                    profile='scripted_human_v1', persona='baseline', optimizer_updates=0,
                    decoding=dict(temperature=0.8, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                  max_new_tokens=512, enable_thinking=False, constrained=False),
                    context_limit=16384, batch_size=4, seed=20260913,
                    note='HF batched inference, all models matched; not prior SGLang or Gemini-human benchmark')
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'manifest':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                      'scenarios':len(scenarios),'episodes_per_model':408}),flush=True)


if __name__=='__main__':
    main()
