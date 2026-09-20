"""Profile native frames and local preprocessing with recorded actions, no model calls.

Use the Qwen3.5 runtime python; native workers use the benchmark python. Output
timings are measurements on the current host, not reconstructed GPU-job timings.
"""
import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time

from interact_env import Action, Decision, Environment, EpisodeSpec
from interact_env.slime_bridge.generate import encode_decision


async def profile(source, args, tokenizer, processor):
    spec = EpisodeSpec(**json.loads((source/'spec.json').read_text()))
    # Read recorded actions before timing; never generate human/model API calls.
    assert spec.config.get('human', 'scripted') == 'scripted'
    responses = []
    with (source/'decisions.jsonl').open() as stream:
        for line in stream:
            responses.append(json.loads(line)['response'])
            if len(responses) >= args.turns:
                break
    env = Environment(spec, args.output/'episodes', timeout=180,
                      python='/gpfs/home/zixianma/interact/.venv/bin/python')
    records = []
    try:
        start = time.perf_counter()
        decision = await env.reset()
        startup = time.perf_counter()-start
        for response in responses:
            if not isinstance(decision, Decision):
                break
            row = dict(decision_id=decision.decision_id, tick=decision.tick)
            _, ids, images, _ = encode_decision(asdict(decision.observation), tokenizer, processor, row)
            row.update(prompt_tokens=len(ids), images=len(images))
            start = time.perf_counter()
            decision = await env.step(Action(env.episode_id, decision.decision_id, response))
            row['env_step_s'] = time.perf_counter()-start
            records.append(row)
            with (env.directory/'replay_timing.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
    finally:
        await env.close()
    native = [json.loads(line) for line in (env.directory/'worker_timing.jsonl').read_text().splitlines()]
    # First native segment includes engine imports/browser launch; show separately.
    steady = native[1:]
    mean = lambda values: statistics.mean(values) if values else None
    result = dict(engine=spec.engine, task=spec.task_id, source=str(source), output=str(env.directory),
                  mode='recorded_action_replay_no_inference', startup_s=startup,
                  turns=len(records), records=records, native_records=native,
                  means={key:mean([r[key] for r in records]) for key in
                         ('image_decode_s', 'chat_template_s', 'processor_tokenize_s', 'env_step_s', 'images', 'prompt_tokens')},
                  native_steady_mean_s=mean([r['native_segment_s'] for r in steady]),
                  browser_steady_mean_s=mean([sum(r['browser_api_s'].values()) for r in steady]),
                  audit_mean_s=mean([r['audit_write_s'] for r in native]))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','native_records')}), flush=True)
    return result


async def main(args):
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    torch.set_num_threads(2)
    os.environ['INTERACT_PROFILE'] = '1'
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    rows = [await profile(source, args, tokenizer, processor) for source in args.episode]
    (args.output/'summary.json').write_text(json.dumps(dict(host=os.uname().nodename,
        cpu_threads=torch.get_num_threads(), inference_measured=False, episodes=rows), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--turns', type=int, default=12)
    parser.add_argument('--model', default='/gpfs/scrubbed/zixianma/checkpoints/web/screensim-qwen-comparison/models/Qwen3.5-4B')
    asyncio.run(main(parser.parse_args()))
