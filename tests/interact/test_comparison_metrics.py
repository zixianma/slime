from examples.interact.tools.comparison.comparison_metrics import format_metrics,summarize


def test_format_metrics():
    assert format_metrics('{"text":"","flag":null}<|im_end|>')['schema_valid']==1
    assert format_metrics('Nothing detected.')['parse_valid']==0
    assert format_metrics('{"task_progress":{"text":"go back"},"flag":null}')['misplaced_text']==1
    assert format_metrics('{"text":"oops","flag":{"error":"unknown"}}')['schema_valid']==0


def test_group_contrast_not_global_variance():
    scenarios=[dict(id=x,split='train',attempts=8) for x in ('a','b')]
    records=[dict(scenario_id=x,task_id=x,split='train',attempt=i,status='completed',turns=1,
                  format=dict(parse_valid=1,schema_valid=1,misplaced_text=0,truncated=0),
                  components=dict(success=float(x=='b'),f1=0.0)) for x in ('a','b') for i in range(8)]
    stats=summarize(records,scenarios)
    assert stats['train/success']==0.5
    assert stats['train/success_mixed_group_fraction']==0
    records[0]['components']['success']=1.0
    assert summarize(records,scenarios)['train/success_mixed_group_fraction']==0.5


def test_failed_episodes_not_zero_reward():
    scenarios=[dict(id='a',split='train',attempts=8)]
    records=[dict(scenario_id='a',task_id='a',split='train',attempt=0,status='failed')]
    stats=summarize(records,scenarios)
    assert stats['train/episodes_failed']==1 and stats['train/episodes_completed']==0
    assert 'train/success' not in stats


def test_report_intervals_and_duplicate_guard(tmp_path):
    import json
    import pytest
    from examples.interact.tools.comparison.report_comparison import read_records,wilson
    low,high=wilson(0,16)
    assert low<1e-12 and high>0.15
    assert wilson(0,0) is None
    row=dict(scenario_id='a',attempt=0)
    path=tmp_path/'episodes.jsonl'
    path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='duplicate'): read_records(path)
