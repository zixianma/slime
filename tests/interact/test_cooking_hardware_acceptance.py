import pytest
from examples.interact.cooking_gemini import cooking_gpu_handoff as handoff
from examples.interact.archive.cooking_calibration.cooking_acceptance_controller import stages, ALLOCATION_SECONDS, CLEANUP_SECONDS


def test_acceptance_budget_and_single_rollout_gpu():
    plan = stages('/test/checkpoint')
    assert ALLOCATION_SECONDS == 7200
    assert sum(s[2] for s in plan)+CLEANUP_SECONDS <= ALLOCATION_SECONDS
    for stage in (plan[2], plan[3]):
        command = stage[1]
        assert command[command.index('--rollout-num-gpus')+1] == '2'
        assert command[command.index('--sglang-config')+1].endswith('cooking_gpu_layout.yaml')
    assert plan[3][3] == {'COOKING_RESTORE_ONLY':'1'}


def test_graphics_handoff_blocks_live_browser(monkeypatch):
    xml = '''<nvidia_smi_log><gpu><uuid>GPU-test</uuid><processes>
      <process_info><pid>123</pid><type>G</type><process_name>chromium</process_name></process_info>
      <process_info><pid>456</pid><type>C</type><process_name>python</process_name></process_info>
      </processes></gpu></nvidia_smi_log>'''
    monkeypatch.setattr(handoff.subprocess, 'check_output', lambda *a, **k: xml)
    assert handoff.graphics_processes() == [dict(uuid='GPU-test',pid=123,name='chromium',type='G')]
    with pytest.raises(RuntimeError, match='live graphics'):
        handoff.assert_no_graphics()
    monkeypatch.setattr(handoff.subprocess, 'check_output', lambda *a, **k: '<nvidia_smi_log/>')
    assert handoff.assert_no_graphics()['graphics_processes'] == []


def test_placeholder_preserves_physical_weight_transfer_offset():
    from pathlib import Path
    import yaml
    path = Path(__file__).resolve().parents[2]/'examples/interact/configs/cooking_gpu_layout.yaml'
    model = yaml.safe_load(path.read_text())['sglang'][0]
    assert model['update_weights'] is True
    assert model['num_gpus_per_engine'] == 1
    offset = 0
    engine_offsets = []
    for group in model['server_groups']:
        if group['worker_type'] != 'placeholder':
            engine_offsets.append(offset)
        offset += group['num_gpus']
    assert offset == 2 and engine_offsets == [1]


def test_torch_and_nvml_uuid_formats_match():
    raw = '6bfcef99-210c-7c34-1fc6-fcad78533e30'
    assert handoff.canonical_gpu_uuid(raw) == handoff.canonical_gpu_uuid('GPU-'+raw)


def test_render_binding_is_scoped_to_cooking(monkeypatch):
    from interact_env.slime_bridge.generate import cooking_worker_prefix
    monkeypatch.setenv('COOKING_BIND_RENDER_WORKERS', '1')
    monkeypatch.setenv('COOKING_RENDER_GPU', '1')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    assert cooking_worker_prefix('screensim') == []
    prefix = cooking_worker_prefix('cooking')
    assert '--gpu-bind=map_gpu:1' in prefix and '--jobid=123' in prefix
