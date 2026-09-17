"""Full-episode cooking loop and strict same-run continuation guards."""
import json
import os
from pathlib import Path
import time

from examples.interact.cooking_training_plan import EVAL_UPDATES, TARGET_UPDATES, experiment, should_evaluate


class BudgetPause(Exception):
    """A planned, resumable stop; never an infrastructure failure."""


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def optimizer_reserve(timings, workload):
    # First measured update took ~0.53 s/turn. Until this run records its own
    # workload, allow 1 s/turn; thereafter scale both turn and token throughput.
    estimate = workload.get('turns', 0)
    if timings.get('optimizer_s') and timings.get('optimizer_workload'):
        previous = timings['optimizer_workload']
        scale = max(workload.get(k, 0)/max(previous.get(k, 0), 1) for k in ('turns', 'tokens'))
        estimate = timings['optimizer_s'] * max(1, scale) * 1.25
    return max(3600, estimate, timings.get('optimizer_s', 0)*1.25) + 600


def assert_learner_topology(args):
    """Fail closed when a launcher promises a specific learner topology."""
    expected_actor_gpus = os.environ.get('COOKING_EXPECTED_ACTOR_GPUS')
    expected_tp = os.environ.get('COOKING_EXPECTED_TP')
    expected_dp = os.environ.get('COOKING_EXPECTED_DP')
    if not (expected_actor_gpus or expected_tp or expected_dp):
        return
    assert expected_actor_gpus and expected_tp and expected_dp, 'incomplete learner topology contract'
    actor_world = args.actor_num_nodes * args.actor_num_gpus_per_node
    model_parallel = (args.tensor_model_parallel_size * args.pipeline_model_parallel_size
                      * args.context_parallel_size)
    assert actor_world == int(expected_actor_gpus), (actor_world, expected_actor_gpus)
    assert args.tensor_model_parallel_size == int(expected_tp), (
        args.tensor_model_parallel_size, expected_tp)
    assert actor_world // model_parallel == int(expected_dp), (
        actor_world, model_parallel, expected_dp)
    assert args.global_batch_size % int(expected_dp) == 0


def configure(args, split):
    assert args.num_rollout == TARGET_UPDATES
    assert (args.rollout_batch_size, args.n_samples_per_prompt, args.global_batch_size) == (6,8,48)
    assert args.n_samples_per_eval_prompt == 4
    assert args.rollout_global_dataset and args.offload_train and args.offload_rollout
    assert_learner_topology(args)
    assert Path(args.prompt_data).resolve() == (split/'train.jsonl').resolve()
    assert len(args.eval_prompt_data) == 2 and Path(args.eval_prompt_data[1]).resolve() == (split/'validation.jsonl').resolve()
    root = Path(os.environ['COOKING_RUN_DIR'])
    contract = experiment()
    contract_path = root/'experiment.json'
    if contract_path.exists():
        previous_contract = json.loads(contract_path.read_text())
        if previous_contract != contract:
            # The user explicitly shortened the still-running experiment after
            # update 1, without changing data, batch construction, or rewards.
            assert previous_contract.get('version') == 'cooking-full-15-v1'
            assert previous_contract.get('target_updates') == 15
            assert previous_contract.get('eval_completed_updates') == [0, 5, 10, 15]
            amendment = root/'schedule-amendment-12-eval3.json'
            write_json(amendment, dict(reason='User requested 12 updates with validation every 3 updates on 2026-09-15',
                completed_updates_at_change=1, previous_version=previous_contract['version'],
                new_version=contract['version']))
            write_json(contract_path, contract)
    else:
        write_json(contract_path, contract)
    receipt = Path(args.save)/'wandb_run.json'
    state_path = root/'resume-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(completed_updates=0, evaluated_updates=[])
    completed = state['completed_updates']
    assert 0 <= completed <= TARGET_UPDATES
    pointer = Path(args.save)/'latest_checkpointed_iteration.txt'
    if completed:
        assert pointer.exists() and int(pointer.read_text()) == completed-1, 'checkpoint/commit mismatch'
        assert (Path(args.save)/f'iter_{completed-1:07d}/common.pt').exists()
        sampler = Path(args.save)/f'rollout/global_dataset_state_dict_{completed-1}.pt'
        import torch
        saved = torch.load(sampler, map_location='cpu', weights_only=False)
        assert saved['sample_group_index'] == completed*6 and saved['sample_index'] == completed*48
        args.load = args.save
        # CLI validation initially sees the base HF --load and enables fresh
        # fine-tuning flags. Changing only the path later would skip optimizer
        # and RNG restoration and incorrectly retain start_rollout_id=0.
        args.finetune = False
        args.no_load_optim = False
        args.no_load_rng = False
        args.start_rollout_id = completed
        args.use_checkpoint_opt_param_scheduler = True
    else:
        assert not pointer.exists(), 'uncommitted checkpoint requires inspection before resume'
    if receipt.exists():
        identity = json.loads(receipt.read_text())
        assert identity['entity'] == args.wandb_team and identity['project'] == args.wandb_project
        args.wandb_run_id = identity['run_id']
        args.interact_tracking_receipt = None
    else:
        assert not completed, 'checkpoint without W&B identity'
        args.interact_tracking_receipt = str(receipt)
    args.interact_resume_completed_updates = completed
    # The first publish below corresponds to policy completed+1 in SGLang's
    # one-based counter, including after a process restart.
    args.update_weight_start_version = completed
    args.interact_eval_scope = 'Full development validation: 10 held-out scenarios x 4 fixed-seed attempts'
    args.interact_training_plan = contract
    reward_amendment = root/'reward-amendment-update3.json'
    if completed >= 2:
        if not reward_amendment.exists():
            failure = root/'audit'/'no_reward_variance.json'
            assert completed == 2 and failure.exists(), 'reward amendment requires the observed update-3 no-variance batch'
            observed = json.loads(failure.read_text())
            assert observed['train/step'] == 3 and observed['train/success_mixed_group_fraction'] == 0
            write_json(reward_amendment, dict(
                reason='All six update-3 scenario groups had uniform binary outcomes, producing zero GRPO advantage',
                observed_train_success=observed['train/success'],
                observed_success_mixed_group_fraction=observed['train/success_mixed_group_fraction'],
                first_optimizer_update=3,
                raw_environment_reward='native_outcome_v1',
                training_reward='native_outcome_lexicographic_v2',
                formula='task_success + 0.1 * detected_errors/errors_planned - 0.01 * min(false_flags, 10)',
                bounds={'failed':[-0.1,0.1], 'successful':[0.9,1.1]},
                cached_rollouts_reusable=True,
                timestamp=time.time()))
        amendment = json.loads(reward_amendment.read_text())
        assert amendment['first_optimizer_update'] == 3
        args.interact_training_plan = {**args.interact_training_plan, 'reward_amendment':amendment}
    amendment_path = root/'baseline-deferred.json'
    if os.environ.get('COOKING_DEFER_BASELINE') == '1' and not amendment_path.exists():
        write_json(amendment_path, dict(reason='User approved training-first recovery on 2026-09-15',
            baseline_complete=False, eval_completed_updates=list(EVAL_UPDATES[1:])))
    args.interact_defer_baseline = amendment_path.exists()
    if args.interact_defer_baseline:
        args.interact_training_plan = {**args.interact_training_plan, 'baseline_deferred': True,
                                      'eval_completed_updates': list(EVAL_UPDATES[1:])}
    args.interact_human_profile = {**args.interact_human_profile, 'renderer':'vulkan'}
    args._cooking_resume_state = state


def train(args, actor, manager, mark):
    import ray
    from examples.interact.cooking_gpu_handoff import assert_no_graphics
    root = Path(os.environ['COOKING_RUN_DIR'])
    state = args._cooking_resume_state
    completed = state['completed_updates']
    assert args.start_rollout_id == completed
    cutoff = float(os.environ['COOKING_SOFT_DEADLINE'])
    stop_at = float(os.environ.get('COOKING_HARD_DEADLINE', 'inf')) - 120
    timing_path = root/'stage-durations.json'
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    def pause(reason):
        args._cooking_budget_paused = True
        write_json(root/'budget-pause.json', dict(completed_updates=completed,
            reason=reason, timestamp=time.time(), partial_batch_is_training_data=False))
        mark('paused_at_checkpoint', completed_updates=completed, status='paused', reason=reason)
        raise BudgetPause(reason)
    def wait_with_progress(reference, phase):
        # Activity is distinct from completed-batch rewards or policy improvement.
        from slime.observability.logging_utils import log
        started = time.monotonic()
        while True:
            ready, _ = ray.wait([reference], timeout=60)
            if ready:
                return ray.get(reference)
            if time.time() >= stop_at:
                # Stop only uncommitted rollout/eval work. Completed episodes
                # have their own policy-bound disk cache, separate from sampler state.
                pause('allocation deadline during '+phase)
            values = {'progress/wall_time': time.time(),
                      'progress/phase_elapsed_s': time.monotonic()-started,
                      'progress/completed_updates': completed,
                      'progress/is_validation': int(phase == 'validation')}
            mark(phase, completed_updates=completed, phase_elapsed_s=values['progress/phase_elapsed_s'])
            log(args, values, step_key='progress/wall_time')
    def publish_weights():
        ray.get(manager.onload_weights.remote())
        actor.update_weights()
        ray.get(manager.onload_kv.remote())
    def evaluate_if_due():
        if completed == 0 and getattr(args, 'interact_defer_baseline', False):
            mark('baseline_deferred', completed_updates=0)
            return
        if should_evaluate(completed) and completed not in state['evaluated_updates']:
            if time.time() >= cutoff:
                pause('validation deferred at allocation boundary')
            mark('validation', completed_updates=completed)
            wait_with_progress(manager.eval.remote(completed), 'validation')
            state['evaluated_updates'].append(completed)
            write_json(root/'resume-state.json', state)
    publish_weights()
    evaluate_if_due()
    for rollout_id in range(completed, TARGET_UPDATES):
        if time.time() >= cutoff:
            mark('paused_at_checkpoint', completed_updates=completed, status='paused')
            return
        mark('training_rollouts', completed_updates=completed, next_update=rollout_id+1)
        rollout_started = time.monotonic()
        data = wait_with_progress(manager.generate.remote(rollout_id), 'training_rollouts')
        timings['rollouts_s'] = time.monotonic()-rollout_started
        # Never begin a known-long optimizer pass just before the outer timeout.
        workload_path = root/'audit'/f'optimizer-workload-{rollout_id}.json'
        workload = json.loads(workload_path.read_text()) if workload_path.exists() else {}
        reserve = optimizer_reserve(timings, workload)
        if time.time()+reserve >= stop_at:
            write_json(timing_path, timings)
            pause('insufficient time for optimizer and checkpoint; complete episodes cached')
        ray.get(manager.offload.remote())
        write_json(root/f'handoff-{rollout_id+1}.json', assert_no_graphics())
        mark('optimizer_update', completed_updates=completed)
        optimizer_started = time.monotonic()
        ray.get(actor.async_train(rollout_id, data))
        timings['optimizer_s'] = time.monotonic()-optimizer_started
        timings['optimizer_workload'] = workload
        write_json(timing_path, timings)
        mark('checkpoint_save', completed_updates=completed)
        actor.save_model(rollout_id, force_sync=True)
        ray.get(manager.save.remote(rollout_id))
        # Publish resumability only after BOTH learner and sampler saves succeed.
        completed = rollout_id+1
        state['completed_updates'] = completed
        write_json(root/'resume-state.json', state)
        publish_weights()
        evaluate_if_due()
        mark('update_complete', completed_updates=completed)
    mark('training_complete', completed_updates=completed, status='passed')
