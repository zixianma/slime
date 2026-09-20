import json
from examples.interact.tools.profiling.report_scaling import summarize


def test_partial_progress_counts_only_completed_turns(tmp_path):
    (tmp_path/'turns.jsonl').write_text('\n'.join(json.dumps(row) for row in [
        dict(engine='cooking',status='completed',env_step_s=10),
        dict(engine='cooking',status='incomplete',http_roundtrip_s=8)])+'\n{"inflight":')
    (tmp_path/'progress.json').write_text(json.dumps(dict(elapsed_job_s=1800)))
    result = summarize(tmp_path, 1, 2, 3600)
    assert result['engine']['cooking']['completed_turns'] == 1
    assert result['engine']['screensim']['completed_turns'] == 0
    assert result['completed_turns_per_allocated_gpu_hour'] == 1
    assert result['engine']['cooking']['timings']['env_step_s']['mean'] == 10
    assert result['engine']['cooking']['server_timings']['queue_time'] is None


def test_manifest_has_same_work_and_resource_mapping():
    from pathlib import Path
    from examples.interact.tools.profiling.profile_scaling import server_command
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root/'examples/interact/configs/scaling_profile_v1.json').read_text())
    assert manifest['concurrency'] == 8 and manifest['max_turns'] == 16
    assert len(manifest['scenarios']['cooking']) == 4
    assert len(manifest['scenarios']['screensim']) == 2
    for gpu in range(4):
        cmd = server_command(8000+gpu, gpu)
        assert cmd[cmd.index('--base-gpu-id')+1] == str(gpu)
        assert '--enable-metrics' in cmd and '--enable-deterministic-inference' in cmd


def test_request_limit_is_only_server_change():
    from examples.interact.tools.profiling.profile_scaling import server_command
    before = server_command(8000, 0)
    after = server_command(8000, 0, max_running_requests=8)
    index = before.index('--max-running-requests')+1
    assert before[index] == '4' and after[index] == '8'
    assert before[:index]+before[index+1:] == after[:index]+after[index+1:]
