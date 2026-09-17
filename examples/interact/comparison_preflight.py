"""CPU processor/import validation plus provenance; never claims GPU validation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, AutoConfig, AutoModelForImageTextToText

ROOT=Path(__file__).resolve().parents[2]
DATA=Path('/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison')


def main():
    manifest=json.loads((DATA/'manifest.json').read_text())
    results=[]
    for spec in manifest['models']:
        config=AutoConfig.from_pretrained(spec['path'],local_files_only=True)
        model=AutoModelForImageTextToText._model_mapping[type(config)]
        processor=AutoProcessor.from_pretrained(spec['path'],local_files_only=True)
        processor.tokenizer.padding_side='left'
        msg=[{'role':'user','content':[{'type':'text','text':'Describe the screen.'},
                                      {'type':'image'},{'type':'image'}]}]
        prompt=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=processor(text=[prompt,prompt],images=[Image.new('RGB',(412,892)) for _ in range(4)],
                         padding=True,return_tensors='pt')
        assert inputs.input_ids.shape[0]==2 and inputs.image_grid_thw.shape[0]==4
        row=dict(model=spec['id'],implementation=model.__name__,input_shape=list(inputs.input_ids.shape),
                 image_grid=inputs.image_grid_thw.tolist(),prompt_suffix=prompt[-70:])
        results.append(row);print(json.dumps(row),flush=True)
    files=list((ROOT/'examples/interact').glob('*comparison*'))
    files += [ROOT/'examples/interact/compare_qwen.py',ROOT/'examples/interact/screensim_compare.sbatch']
    files += list((ROOT/'interact_env').rglob('*.py'))
    report=dict(cpu_only=True,processors=results,
        packages={n:importlib.metadata.version(n) for n in ('torch','transformers','huggingface-hub','wandb','torchvision')},
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
        manifest_sha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest())
    (DATA/'cpu-preflight.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
