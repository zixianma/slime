from types import SimpleNamespace
import json
from pathlib import Path
import pytest
from examples.interact.cooking_calibration_metrics import summarize


def sample(ep, group, turn, total, reward):
    return SimpleNamespace(group_index=group, reward=reward, metadata=dict(
        episode_id=ep, engine='cooking', reward_version='native_outcome_v1',
        turn_index=turn, num_turns=total, task_id=f'case{group}',
        outcome='won' if reward else 'timeout',
        reward_components=dict(detected_errors=reward, errors_planned=2, errors_offered=2,
                               false_flags=0, invalid_response_fraction=0, native_ticks=20)))


def test_success_is_episode_weighted_and_variance_is_within_group():
    data=[sample('a',0,0,1,1),sample('b',0,0,2,0),sample('b',0,1,2,0)]
    r=summarize(data,'train')
    assert r['train/success']==.5
    assert r['train/episodes']==2
    assert r['train/success_mixed_group_fraction']==1
    # Rewards differ globally but each prompt group is constant: no GRPO signal.
    r=summarize([sample('a',0,0,1,1),sample('b',1,0,1,0)],'train')
    assert r['train/success_mixed_group_fraction']==0


def test_full_reward_tiebreak_preserves_success_priority_and_adds_variance():
    from examples.interact.cooking_full_metrics import lexicographic_reward
    def reward(success, detected, planned, false_flags):
        return lexicographic_reward(dict(task_success=success, detected_errors=detected,
                                         errors_planned=planned, false_flags=false_flags))
    assert reward(1, 0, 4, 20) == pytest.approx(.9)
    assert reward(0, 4, 4, 0) == pytest.approx(.1)
    assert reward(1, 0, 4, 20) > reward(0, 4, 4, 0)
    assert len({reward(0, 0, 4, 3), reward(0, 1, 4, 3), reward(0, 1, 4, 7)}) == 3


def test_incomplete_trajectories_rejected():
    with pytest.raises(ValueError):
        summarize([sample('b',0,0,2,0)],'train')


def test_frozen_split_and_calibration_subset():
    root=Path(__file__).resolve().parents[2]/'interact-runs/cooking-rl-v1-20260914'
    manifest=json.loads((root/'manifest.json').read_text())
    a,b,c=[set(manifest['files'][k]['scenarios']) for k in ['train','validation','calibration_validation']]
    assert (len(a),len(b),len(c))==(40,10,4)
    assert not a&b and c<=b
    assert manifest['profile']['reward_version']=='native_outcome_v1'
    assert manifest['profile']['human']=='scripted'


def test_rerun_preserves_membership_and_versions_only_budget(tmp_path):
    from examples.interact.prepare_cooking_rl import prepare
    from examples.interact.cooking_calibration_budget import validate_profile
    old, new = tmp_path/'old', tmp_path/'new'
    prepare(old)
    prepare(new, extended_budget=True)
    a, b = [json.loads((p/'manifest.json').read_text()) for p in (old, new)]
    assert a['source_hashes'] == b['source_hashes']
    for name in a['files']:
        assert a['files'][name]['scenarios'] == b['files'][name]['scenarios']
    assert a['profile'] | {'wall_seconds':3600} == b['profile']
    assert a['version'] != b['version']
    assert validate_profile(new)['composite_wall_seconds'] == 7200
    with pytest.raises(AssertionError):
        validate_profile(old)
    with pytest.raises(AssertionError):
        validate_profile(new, {'max_decisions':300})


def test_prepared_allocation_reserves_downstream_stages():
    from examples.interact.cooking_calibration_budget import ALLOCATION_SECONDS, CLEANUP_SECONDS, STAGE_LIMITS, stage_deadline
    assert sum(STAGE_LIMITS) + CLEANUP_SECONDS <= ALLOCATION_SECONDS
    script = (Path(__file__).resolve().parents[2]/'examples/interact/cooking_qwen35_calibration.sbatch').read_text()
    assert '#SBATCH --time=07:00:00' in script
    assert stage_deadline(1000, 100, 950, [100, 50]) == 850
    assert stage_deadline(1000, 100, 200, [100, 50]) == 300


def test_four_gpu_launcher_uses_all_gpus_for_data_parallel_learning(monkeypatch):
    from examples.interact.cooking_full_training import assert_learner_topology
    root = Path(__file__).resolve().parents[2]
    script = (root/'examples/interact/cooking_full4.sbatch').read_text()
    assert '--actor-num-gpus-per-node 4' in script
    assert '--tensor-model-parallel-size 1' in script
    monkeypatch.setenv('COOKING_EXPECTED_ACTOR_GPUS', '4')
    monkeypatch.setenv('COOKING_EXPECTED_TP', '1')
    monkeypatch.setenv('COOKING_EXPECTED_DP', '4')
    args = SimpleNamespace(actor_num_nodes=1, actor_num_gpus_per_node=4,
                           tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                           context_parallel_size=1, global_batch_size=48)
    assert_learner_topology(args)
    args.actor_num_gpus_per_node = 2
    with pytest.raises(AssertionError):
        assert_learner_topology(args)


def test_master_native_budget_regression(tmp_path):
    """Accelerated wall clock: no GPU/API, not a policy performance evaluation."""
    import os
    import subprocess
    import sys
    code = '''
import json, os, sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from interact_env.adapters.cooking import CookingSpec, run
from interact_env.adapters.cooking_rewards import score
class Silent:
    calls = 0
    def sup_call(self, system, user, frames):
        self.calls += 1
        clock[0] += 24  # approximate slow model seconds per native decision
        assert self.calls <= 600
        return {"text":"", "flag":None}
os.environ.update(COOKSIM_SCRIPTED_COOK='template', COOKSIM_SCRIPTED_ACCEPT='1',
                  COOKSIM_WALL_BUDGET=sys.argv[1], COOKSIM_NO_VIDEO='1',
                  COOKSIM_SCHED='v5', COOKSIM_PACE='tight', COOKSIM_ASSIST_LATENCY='0')
import e2e_stepin_rollout as native
clock = [0]
native.time = SimpleNamespace(time=lambda:clock[0])
policy = Silent()
with redirect_stdout(sys.stderr):
    report = run(CookingSpec(case='b3_master_nops_hard_map_3', wall_seconds=int(sys.argv[1])), policy)
try:
    score(report)
    eligible = True
except ValueError:
    eligible = False
print(json.dumps(dict(aborted=report.get('aborted'), outcome=report['outcome'], calls=policy.calls, eligible=eligible)))
'''
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, 'PYTHONPATH':os.pathsep.join([str(root), str(root.parent/'cook-bench-engine/tools'), str(root.parent/'cook-bench-engine')])}
    results = []
    for seconds in (1800, 3600):
        result = subprocess.run([sys.executable, '-c', code, str(seconds)], cwd=tmp_path,
                                env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr[-4000:]
        results.append(json.loads(result.stdout))
    old, new = results
    assert old['aborted'] == 'wall_clock' and not old['eligible']
    assert new['aborted'] is None and new['eligible']
    assert new['calls'] > old['calls']
