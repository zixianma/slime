"""Engine-neutral Slime custom generation; no native simulator imports."""
import asyncio
import base64
import io
import hashlib
import json
import math
import os
import time
from dataclasses import asdict

from ..protocol import Action, Decision, EpisodeResult, EpisodeSpec
from ..runtime import Environment


def cooking_worker_prefix(engine):
    if engine != 'cooking' or os.environ.get('COOKING_BIND_RENDER_WORKERS') != '1':
        return []
    job = os.environ['SLURM_JOB_ID']
    gpu = int(os.environ['COOKING_RENDER_GPU'])
    assert job.isdigit() and gpu in (0,1)
    return ['srun', '--jobid='+job, '--overlap', '--exact', '--ntasks=1', '--gpus=2',
            '--gpu-bind=map_gpu:'+str(gpu), '--cpus-per-task=16', '--mem=24G']

def encode_decision(decision, tokenizer, processor, timings=None):
    from PIL import Image
    from slime.utils.processing_utils import build_processor_kwargs

    started = time.perf_counter()
    images = [Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[1]))).convert("RGB")
              for s in decision["images"]]
    decoded = time.perf_counter()
    # Native clients send the user text first, then ordered observation images.
    content = [{"type": "text", "text": decision["user"]}] + [{"type": "image"} for _ in images]
    messages = ([{"role": "system", "content": decision["system"]}] if decision["system"] else [])
    messages.append({"role": "user", "content": content if images else decision["user"]})
    template = processor if images else tokenizer
    if template is None:
        raise ValueError("image observations require a multimodal processor")
    prompt = template.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
    templated = time.perf_counter()
    if images:
        encoded = processor(text=prompt, **build_processor_kwargs({"images": images}))
        ids = encoded["input_ids"][0]
        train_inputs = {k: v for k, v in encoded.items() if k not in ("input_ids", "attention_mask")}
    else:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        train_inputs = None
    result = prompt, list(ids), images, train_inputs
    if timings is not None:
        timings.update(image_decode_s=decoded-started, chat_template_s=templated-decoded,
                       processor_tokenize_s=time.perf_counter()-templated)
    return result


def apply_completion(turn, output):
    """Use the server's exact generated token IDs, including its real stop tokens."""
    from slime.utils.types import Sample

    meta = output["meta_info"]
    reason = meta["finish_reason"]["type"]
    if reason not in ("stop", "length"):
        raise RuntimeError(f"inference did not complete: {reason}")
    pairs = meta.get("output_token_logprobs", [])
    if not pairs or any(not isinstance(p[1], int) or p[0] is None or not math.isfinite(p[0]) for p in pairs):
        raise RuntimeError("missing or invalid on-policy token likelihoods")
    if meta.get("completion_tokens", len(pairs)) != len(pairs):
        raise RuntimeError("completion token/logprob count mismatch")
    if meta.get("prompt_tokens", len(turn.tokens)) != len(turn.tokens):
        raise RuntimeError("server/learner prompt token count mismatch")
    turn.tokens = turn.tokens + [p[1] for p in pairs]
    turn.response = output["text"]
    turn.response_length = len(pairs)
    turn.rollout_log_probs = [float(p[0]) for p in pairs]
    turn.loss_mask = [1] * len(pairs)
    turn.status = Sample.Status.COMPLETED if reason == "stop" else Sample.Status.TRUNCATED
    if "weight_version" in meta:
        turn.weight_versions.append(str(meta["weight_version"]))
    turn.metadata["finish_reason"] = reason
    return turn


def decision_timeout(env, spec, turns, limit):
    """Convert a policy-horizon stop into an explicit unsuccessful episode."""
    from interact_env.adapters.cooking import parse_reply
    invalid = sum(not bool(parse_reply(turn.response)) for turn in turns)
    reward_version = spec.config.get("reward_version", "native_outcome_v1")
    receipt = env.directory / "decision_timeout.json"
    receipt.write_text(json.dumps({"outcome": "timeout", "decision_limit": limit,
                                   "completed_decisions": len(turns)}) + "\n")
    components = dict(task_success=0.0, errors_planned=0, errors_offered=0,
                      detected_errors=0, prevented_errors=0, false_flags=0,
                      assistant_calls=len(turns),
                      invalid_responses=invalid,
                      invalid_response_fraction=invalid/max(len(turns), 1),
                      native_ticks=turns[-1].metadata.get("tick", 0),
                      decision_cap_reached=1)
    # A horizon stop occurs before the native engine can emit its final report,
    # so error-event fields remain unknown/zero. It still receives the explicit
    # turn cost from the prevention reward; no prevention credit is fabricated.
    reward = (-0.05 * min(len(turns) / 200, 1)
              if reward_version == "cooking_prevention_turns_v1" else 0.0)
    return EpisodeResult(env.episode_id, "timeout", True, False, True, reward,
                         reward_version,
                         components, {"decision_timeout": str(receipt.resolve())})


async def generate(args, sample, sampling_params, evaluation=False):
    from slime.rollout.sglang_rollout import GenerateState, get_model_url
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    if getattr(args, "partial_rollout", False) or getattr(args, "group_rm", False):
        raise ValueError("initial environment bridge supports complete trajectories with native rewards only")
    if sampling_params.get("top_p", 1.0) != 1.0:
        raise ValueError("top-p replay is not implemented by this bridge; use top_p=1")
    state = GenerateState(args)
    settings = dict(getattr(args, "interact", {}))
    # Task/human/timing configuration is private and never passed to encode_decision.
    spec = EpisodeSpec(**sample.metadata["episode"])
    if spec.engine == 'cooking' and settings.get('cooking_renderer'):
        spec = EpisodeSpec(spec.engine, spec.task_id, spec.seed,
                           {**spec.config, 'renderer': settings['cooking_renderer']})
    episode_cache = None
    if spec.engine == 'cooking' and not evaluation and os.environ.get('COOKING_EPISODE_CACHE') == '1':
        from examples.interact.cooking_episode_cache import EpisodeCache, identity
        episode_cache = EpisodeCache(os.environ['COOKING_RUN_DIR'], identity(args, sample, spec, sampling_params))
        cached = await asyncio.to_thread(episode_cache.load)
        if cached is not None:
            return cached
    if spec.engine == 'cooking' and not evaluation and os.environ.get('COOKING_RECOVERY_MANIFEST'):
        from examples.interact.cooking_recover_episodes import recover
        recovered = await asyncio.to_thread(recover, args, sample, spec, state)
        if recovered is not None:
            if episode_cache is not None:
                await asyncio.to_thread(episode_cache.save, recovered)
            return recovered
    env = Environment(spec, os.environ.get("INTERACT_OUTPUT_ROOT", settings.get("output_root", "interact-runs/episodes")),
                      timeout=settings.get("worker_timeout", 180), python=settings.get("worker_python"),
                      launch_prefix=cooking_worker_prefix(spec.engine))
    rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
    if not isinstance(rollout_id, int) or not isinstance(sample.index, int):
        raise ValueError("Slime parent sample needs integer index and rollout_id")
    turns = []
    profiling = os.environ.get('INTERACT_PROFILE') == '1'
    def record_timing(row):
        if profiling:
            with (env.directory/'rollout_timing.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
    try:
        reset_started = time.perf_counter()
        decision = await env.reset()
        record_timing(dict(phase='reset', seconds=time.perf_counter()-reset_started))
        decision_limit = min(settings.get("max_decisions", 300), 65536)
        while isinstance(decision, Decision):
            if state.aborted:
                raise RuntimeError("rollout cancelled; incomplete trajectory is not training data")
            if len(turns) >= decision_limit:
                decision = decision_timeout(env, spec, turns, decision_limit)
                break
            observation = asdict(decision.observation)
            timing = dict(phase='turn', decision_id=decision.decision_id)
            started = time.perf_counter()
            prompt, ids, images, train_inputs = encode_decision(observation, state.tokenizer, state.processor,
                                                               timing if profiling else None)
            if len(ids) + sampling_params["max_new_tokens"] > args.rollout_max_context_len:
                raise RuntimeError("native prompt exceeds context budget; no silent truncation")
            spill_visual = settings.get('spill_visual_inputs', False)
            if spill_visual:
                from slime.utils.multimodal_storage import store_multimodal
                train_inputs = store_multimodal(train_inputs, env.directory/'visual_tensors'/f'{len(turns)}.pt')
            turn = Sample(group_index=sample.group_index, index=(sample.index << 16) | len(turns),
                          rollout_id=rollout_id, prompt=prompt, tokens=ids,
                          multimodal_inputs={"images": images} if images and not spill_visual else None,
                          multimodal_train_inputs=train_inputs,
                          metadata={"engine": spec.engine, "task_id": spec.task_id,
                                    "group_key": spec.group_key, "episode_id": env.episode_id,
                                    "turn_index": len(turns), "tick": decision.tick,
                                    "time_unit": decision.time_unit, "action_schema": decision.action_schema,
                                    "observation_profile": decision.observation.profile,
                                    "prompt_hash": decision.observation.content_hash, "evaluation": evaluation})
            payload = {"input_ids": ids, "sampling_params": dict(sampling_params), "return_logprob": True}
            if evaluation and settings.get("fixed_eval_sampling_seed") is not None:
                # Stable across checkpoint evaluations; independent across scenario,
                # attempt and assistant turn. This does not claim bitwise kernels.
                seed_key = f"{settings['fixed_eval_sampling_seed']}:{spec.group_key}:{sample.index}:{len(turns)}"
                payload["sampling_params"]["sampling_seed"] = int.from_bytes(
                    hashlib.sha256(seed_key.encode()).digest()[:4], "little") % (2**31)
            if images:
                payload["image_data"] = list(decision.observation.images)
                if settings.get("multimodal_request_format") == "text":
                    # Newer SGLang processors expand image placeholders themselves.
                    # Retain locally encoded IDs and verify the server token count.
                    payload.pop("input_ids")
                    payload["text"] = prompt
            request_started = time.perf_counter()
            try:
                output = await asyncio.wait_for(
                    post(get_model_url(args, "policy"), payload,
                         headers={"X-SMG-Routing-Key": env.episode_id}, max_retries=1),
                    timeout=settings.get("inference_timeout", 120))
            except TimeoutError as exc:
                failure = dict(phase='inference', episode_id=env.episode_id,
                    task=spec.task_id, decision_id=decision.decision_id,
                    completed_turns=len(turns), elapsed_s=time.perf_counter()-request_started)
                (env.directory/'failure.json').write_text(json.dumps(failure)+'\n')
                raise TimeoutError(f'Policy inference timed out: {failure}') from exc
            request_finished = time.perf_counter()
            apply_completion(turn, output)
            turns.append(turn)
            environment_started = time.perf_counter()
            decision = await env.step(Action(env.episode_id, decision.decision_id, turn.response))
            timing.update(http_roundtrip_s=request_finished-request_started,
                          request_setup_s=request_started-started-sum(timing.get(k, 0.) for k in
                              ('image_decode_s','chat_template_s','processor_tokenize_s')),
                          completion_apply_s=environment_started-request_finished,
                          env_step_s=time.perf_counter()-environment_started,
                          total_turn_s=time.perf_counter()-started,
                          prompt_tokens=len(ids), completion_tokens=turn.response_length, images=len(images),
                          server={k:v for k,v in output['meta_info'].items()
                                  if k in ('e2e_latency','queue_time','prefill_launch_delay','prefill_launch_latency',
                                           'prefill_finished_ts','request_received_ts','request_sent_to_scheduler_ts',
                                           'decode_finished_ts','inference_time','decode_throughput')})
            record_timing(timing)
        if not turns or not decision.eligible:
            raise RuntimeError("episode has no eligible completed trajectory")
        if len({v for t in turns for v in t.weight_versions}) > 1:
            raise RuntimeError("policy weights changed inside one on-policy episode")
        for turn in turns:
            turn.reward = decision.reward
            turn.metadata.update(num_turns=len(turns), reward_version=decision.reward_version,
                                 reward_components=decision.components, artifacts=decision.artifacts,
                                 outcome=decision.outcome)
        audit = [{"tokens": t.tokens, "response_length": t.response_length,
                  "rollout_id": t.rollout_id, "rollout_log_probs": t.rollout_log_probs,
                  "loss_mask": t.loss_mask, "reward": t.reward, "weight_versions": t.weight_versions,
                  "metadata": t.metadata} for t in turns]
        (env.directory / "training_audit.json").write_text(json.dumps(audit) + "\n")
        if episode_cache is not None:
            await asyncio.to_thread(episode_cache.save, turns)
        return turns
    finally:
        await env.close()
