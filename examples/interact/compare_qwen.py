"""One GPU / frozen HF policy. Own native episodes, batch inference, log scalars."""
import argparse
import asyncio
import base64
from collections import Counter
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from interact_env import Action, Decision, Environment, EpisodeSpec
from examples.interact.comparison_metrics import format_metrics, summarize


class BatchInference:
    def __init__(self, model_spec, manifest):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig
        self.torch = torch
        self.manifest = manifest
        self.queue = asyncio.Queue()
        self.processor = AutoProcessor.from_pretrained(model_spec['path'], local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        self.model = AutoModelForImageTextToText.from_pretrained(model_spec['path'],
            local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map={'':'cuda:0'}).eval()
        # Explicit settings: do not inherit differing model-card sampling defaults.
        self.generation = GenerationConfig(do_sample=True, temperature=0.8, top_p=1.0, top_k=0,
            repetition_penalty=1.0, max_new_tokens=512, use_cache=True,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id)
        torch.manual_seed(manifest['seed'])
        print(json.dumps({'event':'MODEL_READY','model':model_spec['id'],
                          'gpu':torch.cuda.get_device_name(0)}),flush=True)

    async def request(self, decision):
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((decision,future))
        return await future

    def generate(self, decisions):
        from PIL import Image
        texts, images = [], []
        for d in decisions:
            obs = d.observation
            content = [{'type':'text','text':obs.user}]
            for data in obs.images:
                images.append(Image.open(io.BytesIO(base64.b64decode(data.split(',',1)[1]))).convert('RGB'))
                content.append({'type':'image'})
            messages = ([{'role':'system','content':obs.system}] if obs.system else [])
            messages.append({'role':'user','content':content})
            texts.append(self.processor.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False))
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors='pt')
        if inputs.input_ids.shape[1]+512 > self.manifest['context_limit']:
            raise ValueError('context budget exceeded; never truncate the native prompt')
        inputs = inputs.to('cuda:0')
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.generation)
        suffix = output[:,inputs.input_ids.shape[1]:].tolist()
        eos = self.generation.eos_token_id
        eos = {eos} if isinstance(eos,int) else set(eos or [])
        results = []
        for tokens in suffix:
            stopped = next((i for i,t in enumerate(tokens) if t in eos),None)
            if stopped is not None:
                tokens=tokens[:stopped+1]
            raw=self.processor.tokenizer.decode(tokens,skip_special_tokens=False)
            results.append(dict(raw=raw,tokens=tokens,truncated=int(stopped is None and len(tokens)>=512),
                                batch_seconds=time.monotonic()-started))
        return results

    async def serve(self):
        while True:
            batch=[await self.queue.get()]
            await asyncio.sleep(0.025)
            while len(batch)<self.manifest['batch_size'] and not self.queue.empty():
                batch.append(self.queue.get_nowait())
            try:
                results=await asyncio.to_thread(self.generate,[d for d,_ in batch])
            except Exception as exc:
                for _,future in batch:
                    if not future.done(): future.set_exception(exc)
                raise
            for (_,future),result in zip(batch,results,strict=True):
                if not future.done(): future.set_result(result)


async def collect(args, manifest, run, records):
    policy=BatchInference(manifest['models'][args.model_index],manifest)
    queue=asyncio.Queue()
    # Round-robin attempts across every scenario prevents a deadline from hiding
    # entire tasks behind a long block of repeated attempts on the first task.
    for attempt in range(16):
        for scenario in manifest['scenarios']:
            if attempt<scenario['attempts']:
                queue.put_nowait((scenario,attempt))
    async def episode_worker():
        while not queue.empty():
            scenario,attempt=queue.get_nowait()
            env=Environment(EpisodeSpec(**scenario['spec']),args.output/'episodes',timeout=180,
                            python=str(ROOT.parent/'.venv/bin/python'))
            row=dict(scenario_id=scenario['id'],task_id=scenario['spec']['task_id'],
                     split=scenario['split'],attempt=attempt,episode_id=env.episode_id)
            started=time.monotonic()
            counts=Counter(); turns=0
            try:
                async with asyncio.timeout(900):
                    decision=await env.reset()
                    with (env.directory/'generation.jsonl').open('w') as audit:
                        while isinstance(decision,Decision):
                            if turns>=300: raise RuntimeError('decision budget exceeded')
                            output=await policy.request(decision)
                            fmt=format_metrics(output['raw'])
                            counts.update(fmt); counts['truncated']+=output['truncated']
                            audit.write(json.dumps({'decision_id':decision.decision_id,
                                'prompt_hash':decision.observation.content_hash,**output,**fmt})+'\n')
                            audit.flush(); turns+=1
                            decision=await env.step(Action(env.episode_id,decision.decision_id,output['raw']))
                    if not decision.eligible: raise RuntimeError('ineligible native episode')
                    row.update(status='completed',components=decision.components,reward=decision.reward,
                               turns=turns,format=dict(counts),native_report=decision.artifacts['native_report'])
            except asyncio.CancelledError:
                row.update(status='interrupted',turns=turns)
                raise
            except Exception as exc:
                row.update(status='failed',error=f'{type(exc).__name__}: {exc}',turns=turns)
            finally:
                await env.close()
                row['seconds']=time.monotonic()-started
                records.append(row)
                with (args.output/'episodes.jsonl').open('a') as out:
                    out.write(json.dumps(row)+'\n')
                stats=summarize(records,manifest['scenarios'])
                stats['episodes_processed']=len(records)
                run.log(stats)
                print(json.dumps({'event':'EPISODE',**{k:v for k,v in row.items()
                    if k not in ('components','native_report')},'success':row.get('components',{}).get('success')}),flush=True)
    server=asyncio.create_task(policy.serve())
    workers=[asyncio.create_task(episode_worker()) for _ in range(manifest['batch_size'])]
    joined=asyncio.gather(*workers)
    try:
        done,_=await asyncio.wait([server,joined],timeout=args.seconds,return_when=asyncio.FIRST_COMPLETED)
        if not done: raise TimeoutError('bounded comparison deadline')
        if server in done: await server
        await joined
    finally:
        for t in workers: t.cancel()
        server.cancel()
        await asyncio.gather(*workers,server,return_exceptions=True)
        if not joined.done(): joined.cancel()
        await asyncio.gather(joined,return_exceptions=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--model-index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=6900)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text())
    import wandb
    import transformers,torch
    model=manifest['models'][args.model_index]
    records=[]; status='failed'; error=None; exit_code=1
    config={k:v for k,v in manifest.items() if k!='models'}
    config.update(model=model,transformers=transformers.__version__,torch=torch.__version__,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        slurm_job_id=os.environ['SLURM_JOB_ID'],runtime='HF batched SDPA',
        randomness='global fixed torch seed; asynchronous batch order is recorded, not bitwise reproducible')
    with wandb.init(entity='zixianma',project='interact-slime-rl',mode='online',id=uuid.uuid4().hex,
        resume='never',group='screensim-qwen-compare-'+os.environ['SLURM_JOB_ID'],
        name=model['id'].split('/')[1]+'-'+os.environ['SLURM_JOB_ID'],
        dir=str(args.output),config=config,tags=['screensim','offline-rollouts','no-training','scripted-human'],
        settings=wandb.Settings(console='off',disable_code=True)) as run:
        location=dict(url=run.url,id=run.id,model=model['id'])
        (args.output/'wandb_run.json').write_text(json.dumps(location,indent=2)+'\n')
        print(json.dumps(location),flush=True)
        loop=asyncio.new_event_loop();asyncio.set_event_loop(loop)
        task=loop.create_task(collect(args,manifest,run,records))
        for sig in (signal.SIGTERM,signal.SIGINT):
            loop.add_signal_handler(sig,task.cancel)
        try:
            loop.run_until_complete(task)
            status='completed' if all(r['status']=='completed' for r in records) and len(records)==408 else 'partial'
            exit_code=0 if status=='completed' else 2
        except (Exception,asyncio.CancelledError) as exc:
            error=f'{type(exc).__name__}: {exc}'
            print(error,flush=True)
            import traceback
            traceback.print_exc()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            summary=dict(status=status,error=error,episodes=len(records),
                         metrics=summarize(records,manifest['scenarios']),**location)
            (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            run.summary.update({'comparison_status':status,'comparison_error':error,
                                'optimizer_updates':0,**summary['metrics']})
    sys.exit(exit_code)


if __name__=='__main__': main()
