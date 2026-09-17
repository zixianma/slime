"""One bounded cooking update using official Slime actors and rollout manager."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from slime.utils.arguments import parse_args
    args = parse_args()
    from examples.interact.cooking_calibration_budget import split_path, validate_profile
    split = split_path()
    validate_profile(split, args.interact)
    raw = (split/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    for name, info in manifest['files'].items():
        assert hashlib.sha256((split/f'{name}.jsonl').read_bytes()).hexdigest() == info['sha256']
    engine = ROOT.parent/'cook-bench-engine/bench/v5'
    for name, digest in manifest['source_hashes'].items():
        assert hashlib.sha256((engine/name).read_bytes()).hexdigest() == digest
    args.interact_split_version = manifest['version']
    args.interact_split_sha256 = hashlib.sha256(raw).hexdigest()
    args.interact_model_revision = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    args.interact_engine = 'cooking'
    args.interact_human_profile = manifest['profile']
    args.interact_eval_scope = '4 preselected calibration scenarios; not full validation'
    args.interact_max_decisions = args.interact['max_decisions']
    args.interact_effective_episode_wall_seconds = 2 * manifest['profile']['wall_seconds']
    acceptance = os.environ.get('COOKING_HARDWARE_ACCEPTANCE') == '1'
    full_training = os.environ.get('COOKING_FULL_TRAINING') == '1'
    transfer_debug = os.environ.get('COOKING_TRANSFER_DEBUG') == '1'
    reconnect_debug = os.environ.get('COOKING_RECONNECT_DEBUG') == '1'
    if transfer_debug or reconnect_debug:
        args.use_wandb = False
    if acceptance:
        assert args.rollout_num_gpus in (2, 4) and args.interact['cooking_renderer'] == 'vulkan'
        args.interact_eval_scope = 'Acceptance only: 4 calibration scenarios; not full validation'
    if full_training:
        from examples.interact.cooking_full_training import configure
        configure(args, split)
    if os.environ.get('INTERACT_PARSE_ONLY') == '1':
        print('COOKING_ARGS_OK', args.global_batch_size, args.rollout_batch_size, args.n_samples_per_prompt)
        return
    import ray
    from slime.observability.logging_utils import configure_logger, init_tracking, finish_tracking
    from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
    from examples.interact.cooking_calibration_actor import CookingAuditActor
    job = Path(os.environ['INTERACT_JOB_DIR'])
    restore = os.environ.get('COOKING_RESTORE_ONLY') == '1'
    if restore:
        args.use_wandb = False
    elif not full_training:
        assert not (Path(args.load)/'latest_checkpointed_iteration.txt').exists(), 'calibration must start from base HF model'
        assert not args.wandb_run_id, 'new cooking experiment must not reuse ScreenSim ID'
        args.interact_tracking_receipt = str(Path(args.save)/'wandb_run.json')
    def interrupted(sig, frame):
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM, interrupted)
    manager = None
    progress = dict(completed_updates=getattr(args, 'interact_resume_completed_updates', 0), status='running')
    def mark(phase, **extra):
        progress.update(phase=phase, timestamp=time.time(), **extra)
        (job/'progress.json').write_text(json.dumps(progress)+'\n')
        print('COOKING_PROGRESS '+json.dumps(progress), flush=True)
    try:
        mark('restore_startup' if restore else 'startup')
        if acceptance:
            import torch
            renderer_gpu = int(os.environ.get('COOKING_RENDER_GPU', '0'))
            assert renderer_gpu in (0,1)
            from examples.interact.cooking_gpu_handoff import canonical_gpu_uuid
            os.environ['INTERACT_RENDER_GPU_UUID'] = canonical_gpu_uuid(torch.cuda.get_device_properties(renderer_gpu).uuid)
        allocated_gpus = args.rollout_num_gpus if acceptance else 2
        allocated_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '16'))
        ray.init(address='local', num_cpus=allocated_cpus, num_gpus=allocated_gpus,
                 object_store_memory=4*1024**3, include_dashboard=False)
        configure_logger()
        pgs = create_placement_groups(args)
        if acceptance:
            pg, indices, gpu_ids = pgs['rollout']
            expected_gpu_ids = list(range(args.rollout_num_gpus))
            assert len(indices) == args.rollout_num_gpus
            assert [int(g) for g in gpu_ids] == expected_gpu_ids
            policy_gpus = [gpu_ids[i] for i in expected_gpu_ids if i != renderer_gpu]
            # Keep the full placement: the SGLang placeholder preserves the renderer gap
            # for both engine placement and same-device CUDA IPC weight transfer.
            (job/'gpu-layout.json').write_text(json.dumps(dict(renderer_uuid=os.environ['INTERACT_RENDER_GPU_UUID'],
                renderer_gpu=gpu_ids[renderer_gpu], policy_gpus=policy_gpus,
                learner_gpus=gpu_ids[:args.actor_num_gpus_per_node]))+'\n')
        init_tracking(args)
        manager, _ = create_rollout_manager(args, pgs['rollout'])
        if acceptance:
            engines, _, _, counts, offsets, _ = ray.get(manager.get_updatable_engines_and_lock.remote())
            expected_offsets = [i for i in range(args.rollout_num_gpus) if i != renderer_gpu]
            assert len(engines) == args.rollout_num_gpus-1
            assert counts == [1]*(args.rollout_num_gpus-1) and offsets == expected_offsets, (counts, offsets)
            (job/'weight-routing.json').write_text(json.dumps(dict(counts=counts, offsets=offsets))+'\n')
        actor, _ = create_training_models(args, pgs, manager, actor_cls=CookingAuditActor)
        if full_training:
            from examples.interact.cooking_full_training import train, BudgetPause, write_json
            completed = args.interact_resume_completed_updates
            if completed:
                mark('verify_restored_weights', completed_updates=completed)
                restored = ray.get([worker.verify_restored_weights.remote(completed)
                                    for worker in actor._actor_handlers])
                write_json(job/'restore-verification.json', restored)
            if reconnect_debug:
                import requests
                for publication in range(1, 4):
                    mark('reconnect_debug_publish', completed_updates=completed, publication=publication)
                    ray.get(manager.onload_weights.remote())
                    actor.update_weights()
                    ray.get(manager.onload_kv.remote())
                    urls = ray.get([engine.get_url.remote() for engine in engines])
                    for url in urls:
                        response = requests.post(url+'/generate', json={
                            'text':'What is two plus two? Answer briefly.',
                            'sampling_params':{'temperature':0, 'max_new_tokens':16}}, timeout=120)
                        response.raise_for_status()
                        assert response.json()['meta_info']['completion_tokens'] > 0
                    ray.get(manager.offload.remote())
                mark('reconnect_debug_complete', completed_updates=completed,
                     publications=3, policy_engines=len(engines), status='passed')
                return
            try:
                train(args, actor, manager, mark)
            except BudgetPause:
                # The outer timeout remains a last-resort watchdog; normal
                # deadline handling returns successfully after recording a pause.
                ray.kill(manager, no_restart=True)
                manager = None
            return
        if transfer_debug:
            import requests
            results = []
            for iteration in range(3):
                mark('debug_weight_transfer', transfer=iteration+1)
                if args.offload_rollout:
                    ray.get(manager.onload_weights.remote())
                actor.update_weights()
                if args.offload_rollout:
                    ray.get(manager.onload_kv.remote())
                url = ray.get(engines[0].get_url.remote())
                response = requests.post(url+'/generate', json={
                    'text':'What is two plus two? Answer briefly.',
                    'sampling_params':{'temperature':0, 'max_new_tokens':32}}, timeout=120)
                response.raise_for_status()
                output = response.json()
                assert output.get('text') and output['meta_info']['completion_tokens'] > 0
                results.append(dict(transfer=iteration+1, text=output['text'],
                                    meta_info=output['meta_info']))
                (job/'transfer-debug.json').write_text(json.dumps(results,indent=2)+'\n')
                if args.offload_rollout:
                    ray.get(manager.offload.remote())
            if os.environ.get('COOKING_RENDER_SMOKE'):
                from examples.interact.cooking_render_probe import run_probe
                ray.get(manager.onload_weights.remote())
                ray.get(manager.onload_kv.remote())
                probe = ray.get(ray.remote(num_cpus=1)(run_probe).remote(
                    str(job/'live-render-probe'), os.environ['COOKING_RENDER_SMOKE'], 8, 4, url))
                assert probe['status'] == 'passed', probe
                ray.get(manager.offload.remote())
                from examples.interact.cooking_gpu_handoff import assert_no_graphics
                (job/'debug-handoff.json').write_text(json.dumps(assert_no_graphics())+'\n')
            mark('debug_complete', status='passed', transfers=3, optimizer_updates=0)
            return
        if restore:
            assert args.start_rollout_id == 1
            if acceptance:
                checks = ray.get([a.verify_restored_weights.remote() for a in actor._actor_handlers])
                (job/'restored-weights.json').write_text(json.dumps(checks)+'\n')
            (job/'restore_verified.json').write_text(json.dumps(dict(completed_updates=1, optimizer_and_sampler_load=True))+'\n')
            mark('restore_verified', completed_updates=1, status='passed')
            return
        assert args.start_rollout_id == 0 and args.num_rollout == 1
        if args.offload_rollout:
            ray.get(manager.onload_weights.remote())
        actor.update_weights()
        if args.offload_rollout:
            ray.get(manager.onload_kv.remote())
        mark('baseline_evaluation')
        ray.get(manager.eval.remote(0))
        mark('training_rollouts')
        data = ray.get(manager.generate.remote(0))
        if args.offload_rollout:
            ray.get(manager.offload.remote())
        if acceptance:
            from examples.interact.cooking_gpu_handoff import assert_no_graphics
            (job/'handoff.json').write_text(json.dumps(assert_no_graphics())+'\n')
        mark('optimizer_update')
        ray.get(actor.async_train(0, data))
        mark('checkpoint_save', completed_updates=1)
        actor.save_model(0, force_sync=True)
        if args.rollout_global_dataset:
            ray.get(manager.save.remote(0))
        if not args.offload_train:
            actor.clear_memory()
        if args.offload_rollout:
            ray.get(manager.onload_weights.remote())
        actor.update_weights()
        if args.offload_rollout:
            ray.get(manager.onload_kv.remote())
        mark('post_update_evaluation')
        if not acceptance:
            ray.get(manager.eval.remote(1))
        else:
            # Weight synchronization only; no claim of post-update policy quality.
            args.interact_eval_scope = 'post-update weight refresh checked by manager weight synchronization'
            (job/'weight_refresh.json').write_text(json.dumps(dict(completed_updates=1, synchronized=True))+'\n')
        (job/'training_complete.json').write_text('{"completed_updates":1}\n')
        mark('training_complete', status='passed')
    except BaseException as exc:
        mark(progress['phase'], status='failed', error=f'{type(exc).__name__}: {exc}'[-2000:])
        raise
    finally:
        try:
            if manager is not None:
                try:
                    ray.get(manager.dispose.remote(), timeout=45)
                except Exception as cleanup_error:
                    # Do not replace the original failure with a secondary RPC error.
                    if progress['status'] != 'failed':
                        raise
                    print(f'COOKING_CLEANUP_ERROR {type(cleanup_error).__name__}: {cleanup_error}', flush=True)
        finally:
            finish_tracking(args)
            ray.shutdown()


if __name__ == '__main__':
    main()
