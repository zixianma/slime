from dataclasses import asdict
from types import SimpleNamespace
import pytest
from interact_env.protocol import EpisodeSpec
from examples.interact.cooking_gemini.cooking_episode_cache import EpisodeCache


def fixture():
    spec = EpisodeSpec('cooking','task')
    contract = dict(episode=asdict(spec),index=48,group_index=6,policy={'completed_updates':1},sampling={'temperature':.8})
    turn = SimpleNamespace(metadata=dict(num_turns=1,turn_index=0,evaluation=False,
        group_key=spec.group_key,outcome='won',episode_id='one'),rollout_id=48,group_index=6,
        index=48<<16,reward=1.,response_length=1,rollout_log_probs=[-.2],loss_mask=[1],
        tokens=[1,2],weight_versions=['2'],multimodal_train_inputs=None)
    return contract, [turn]


def test_cache_roundtrip_and_policy_sampling_isolation(tmp_path):
    contract, turns = fixture()
    cache = EpisodeCache(tmp_path,contract)
    assert cache.load() is None
    cache.save(turns)
    assert cache.load()[0].tokens == [1,2]
    assert EpisodeCache(tmp_path,{**contract,'policy':{'completed_updates':2}}).load() is None
    assert EpisodeCache(tmp_path,{**contract,'sampling':{'temperature':1.}}).load() is None


def test_cache_rejects_partial_and_mismatched_identity(tmp_path):
    contract, turns = fixture()
    cache = EpisodeCache(tmp_path,contract)
    turns[0].metadata['num_turns'] = 2
    with pytest.raises(AssertionError):
        cache.save(turns)
    assert not cache.path.exists()
    turns[0].metadata['num_turns'] = 1
    turns[0].rollout_id = 49
    with pytest.raises(AssertionError):
        cache.save(turns)


def test_decision_cap_is_an_explicit_zero_reward_timeout(tmp_path):
    from interact_env.slime_bridge.generate import decision_timeout
    env = SimpleNamespace(directory=tmp_path, episode_id='capped')
    spec = EpisodeSpec('cooking', 'task', config={'reward_version':'native_outcome_v1'})
    turns = [SimpleNamespace(response='{"text":"","flag":null}', metadata={'tick':398})
             for _ in range(200)]
    result = decision_timeout(env, spec, turns, 200)
    assert result.eligible and result.terminated and not result.truncated
    assert result.outcome == 'timeout' and result.reward == 0
    assert result.components['decision_cap_reached'] == 1
    assert result.components['prevented_errors'] == 0
    assert result.components['assistant_calls'] == 200
    assert (tmp_path/'decision_timeout.json').is_file()


def test_decision_cap_applies_full_prevention_reward_turn_penalty(tmp_path):
    from interact_env.slime_bridge.generate import decision_timeout
    env = SimpleNamespace(directory=tmp_path, episode_id='capped-prevention')
    spec = EpisodeSpec('cooking', 'task', config={
        'reward_version': 'cooking_prevention_turns_v1'})
    turns = [SimpleNamespace(response='{"text":"","flag":null}', metadata={'tick':398})
             for _ in range(200)]
    result = decision_timeout(env, spec, turns, 200)
    assert result.reward == pytest.approx(-0.05)
    assert result.components['errors_planned'] == 0
    assert result.components['decision_cap_reached'] == 1
