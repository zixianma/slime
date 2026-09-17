import json
import sys
from types import SimpleNamespace
from examples.interact.cooking_full_training import train, configure, BudgetPause
from examples.interact.cooking_training_plan import SPLIT
import pytest


class Remote:
    def __init__(self, fn):
        self.remote = fn


def test_optimizer_reserve_scales_with_real_batch_work():
    from examples.interact.cooking_full_training import optimizer_reserve
    assert optimizer_reserve({}, {'turns':8000, 'tokens':100000}) == 8600
    previous = dict(optimizer_s=2600, optimizer_workload={'turns':5000, 'tokens':100000})
    assert optimizer_reserve(previous, {'turns':10000, 'tokens':300000}) == 10350
    assert optimizer_reserve(previous, {'turns':2500, 'tokens':50000}) == 4200


@pytest.mark.parametrize('defer_baseline', [False, True])
def test_full_loop_sparse_eval_and_checkpoints(tmp_path, monkeypatch, defer_baseline):
    monkeypatch.setenv('COOKING_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('COOKING_SOFT_DEADLINE', '99999999999')
    heartbeats = []
    polls = iter([False] + [True]*100)
    monkeypatch.setitem(sys.modules, 'ray', SimpleNamespace(get=lambda x:x,
        wait=lambda refs,timeout:(refs,[]) if next(polls) else ([],refs)))
    monkeypatch.setattr('slime.observability.logging_utils.log',
                        lambda args,values,step_key:heartbeats.append(values))
    monkeypatch.setattr('examples.interact.cooking_gpu_handoff.assert_no_graphics', lambda:{'graphics_processes':[]})
    updates, saves, evals, sampler_saves = [], [], [], []
    actor = SimpleNamespace(update_weights=lambda:None, async_train=lambda i,data:updates.append(i),
                            save_model=lambda i,force_sync:saves.append(i))
    manager = SimpleNamespace(onload_weights=Remote(lambda:None), onload_kv=Remote(lambda:None),
        offload=Remote(lambda:None), generate=Remote(lambda i:i), eval=Remote(evals.append),
        save=Remote(sampler_saves.append))
    args = SimpleNamespace(start_rollout_id=0, interact_defer_baseline=defer_baseline,
                           _cooking_resume_state={'completed_updates':0,'evaluated_updates':[]})
    train(args, actor, manager, lambda *a,**k:None)
    assert updates == saves == sampler_saves == list(range(12))
    expected_evals = [3,6,9,12] if defer_baseline else [0,3,6,9,12]
    assert evals == expected_evals
    assert len(heartbeats) == 1 and heartbeats[0]['progress/completed_updates'] == 0
    assert heartbeats[0]['progress/is_validation'] == int(not defer_baseline)
    assert json.loads((tmp_path/'resume-state.json').read_text())['completed_updates'] == 12
    args.start_rollout_id = 12
    train(args, actor, manager, lambda *a,**k:None)
    assert len(updates) == 12 and evals == expected_evals


def config_args(root):
    return SimpleNamespace(num_rollout=12, rollout_batch_size=6, n_samples_per_prompt=8,
        global_batch_size=48,n_samples_per_eval_prompt=4,rollout_global_dataset=True,
        offload_train=True,offload_rollout=True,prompt_data=str(SPLIT/'train.jsonl'),
        eval_prompt_data=['cooking',str(SPLIT/'validation.jsonl')],save=str(root/'checkpoints'),
        wandb_team='zixianma',wandb_project='interact-slime-rl',interact_human_profile={})


def test_resume_reuses_identity_and_rejects_uncommitted_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('COOKING_RUN_DIR', str(tmp_path))
    args = config_args(tmp_path)
    configure(args,SPLIT)
    receipt = tmp_path/'checkpoints/wandb_run.json'
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({'entity':'zixianma','project':'interact-slime-rl','run_id':'cooking-test'}))
    resumed = config_args(tmp_path)
    configure(resumed,SPLIT)
    assert resumed.wandb_run_id == 'cooking-test' and resumed.interact_tracking_receipt is None
    (receipt.parent/'latest_checkpointed_iteration.txt').write_text('0')
    with pytest.raises(AssertionError, match='uncommitted'):
        configure(config_args(tmp_path),SPLIT)


def test_committed_resume_preserves_policy_version_and_sampler(tmp_path, monkeypatch):
    import torch
    monkeypatch.setenv('COOKING_RUN_DIR', str(tmp_path))
    checkpoint = tmp_path/'checkpoints'
    (checkpoint/'iter_0000000').mkdir(parents=True)
    (checkpoint/'rollout').mkdir()
    (checkpoint/'iter_0000000/common.pt').touch()
    (checkpoint/'latest_checkpointed_iteration.txt').write_text('0')
    torch.save({'sample_group_index':6,'sample_index':48},
               checkpoint/'rollout/global_dataset_state_dict_0.pt')
    (checkpoint/'wandb_run.json').write_text(json.dumps(
        dict(entity='zixianma', project='interact-slime-rl', run_id='same-cooking-run')))
    (tmp_path/'resume-state.json').write_text(json.dumps(
        dict(completed_updates=1, evaluated_updates=[])))
    args = config_args(tmp_path)
    args.finetune = args.no_load_optim = args.no_load_rng = True
    args.start_rollout_id = 0
    configure(args, SPLIT)
    assert args.load == args.save and args.use_checkpoint_opt_param_scheduler
    assert not args.finetune and not args.no_load_optim and not args.no_load_rng
    assert args.start_rollout_id == 1
    assert args.interact_resume_completed_updates == args.update_weight_start_version == 1
    assert args.wandb_run_id == 'same-cooking-run'


@pytest.mark.parametrize('ready', [True, False])
def test_budget_pause_never_updates_or_commits_partial_batch(tmp_path, monkeypatch, ready):
    import time
    monkeypatch.setenv('COOKING_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('COOKING_SOFT_DEADLINE', str(time.time()+1000))
    # Ready rollout: not enough reserve for optimizer. In-flight: deadline reached.
    monkeypatch.setenv('COOKING_HARD_DEADLINE', str(time.time()+(1000 if ready else 60)))
    canceled, phases = [], []
    monkeypatch.setitem(sys.modules, 'ray', SimpleNamespace(get=lambda x:x,
        wait=lambda refs,timeout:(refs,[]) if ready else ([],refs),
        cancel=lambda ref,force:canceled.append(ref)))
    actor = SimpleNamespace(update_weights=lambda:None)
    manager = SimpleNamespace(onload_weights=Remote(lambda:None),onload_kv=Remote(lambda:None),
                              generate=Remote(lambda i:'pending'))
    args = SimpleNamespace(start_rollout_id=1,interact_defer_baseline=True,
        _cooking_resume_state={'completed_updates':1,'evaluated_updates':[]})
    with pytest.raises(BudgetPause):
        train(args, actor, manager, lambda phase,**kw:phases.append((phase,kw)))
    assert args._cooking_budget_paused and phases[-1][1]['status'] == 'paused'
    assert not (tmp_path/'resume-state.json').exists()
    assert json.loads((tmp_path/'budget-pause.json').read_text())['completed_updates'] == 1
    assert canceled == []  # main kills the owned rollout actor after recording the pause
