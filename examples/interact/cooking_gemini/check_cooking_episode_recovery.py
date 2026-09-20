"""Bounded preflight for a pinned, complete cooking episode on an allocated node."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

from examples.interact.cooking_gemini.cooking_recover_episodes import manifest, recover
from interact_env.protocol import EpisodeSpec
from slime.utils.processing_utils import load_processor, load_tokenizer


def main():
    receipt = manifest(os.environ['COOKING_RECOVERY_MANIFEST'])
    root = Path(os.environ['COOKING_RUN_DIR'])
    completed = json.loads((root/'resume-state.json').read_text())['completed_updates']
    assert completed == receipt['completed_updates']
    index = min(map(int, receipt['episodes']))
    directory = Path(receipt['episodes'][str(index)]['directory'])
    spec = EpisodeSpec(**json.loads((directory/'spec.json').read_text()))
    args = SimpleNamespace(interact_resume_completed_updates=completed,
        hf_checkpoint=receipt['model'], load=str(root/'checkpoints'),
        interact_model_revision=receipt['model_revision'],
        interact={'max_decisions':receipt['max_turns']})
    state = SimpleNamespace(tokenizer=load_tokenizer(receipt['model'], trust_remote_code=True),
        processor=load_processor(receipt['model'], trust_remote_code=True))
    print(f'COOKING_RECOVERY_CHECK_START index={index}', flush=True)
    turns = recover(args, SimpleNamespace(index=index, group_index=index//8), spec, state)
    assert turns and len(turns) == turns[0].metadata['num_turns']
    result = dict(status='passed', index=index, turns=len(turns), completed_updates=completed,
                  manifest=os.environ['COOKING_RECOVERY_MANIFEST'])
    (root/'audit'/f'recovery-check-{os.environ["SLURM_JOB_ID"]}.json').write_text(
        json.dumps(result, indent=2)+'\n')
    print('COOKING_RECOVERY_CHECK_PASSED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
