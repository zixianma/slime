"""Read-only scalar summaries for frozen-policy ScreenSim screening."""
from collections import defaultdict
import json
import math
import re


def format_metrics(raw):
    # Match the native AssistantV2 JSON extraction exactly, without repairing it.
    text = re.sub(r'^```(?:json)?|```$', '', raw.strip(), flags=re.M).strip()
    match = re.search(r'\{.*\}', text, re.S)
    try:
        obj = json.loads(match.group(0)) if match else None
    except (ValueError, TypeError):
        obj = None
    valid = isinstance(obj, dict) and bool(obj)
    nested = obj.get('task_progress') if valid else None
    misplaced = isinstance(nested, dict) and bool(nested.get('text')) and not obj.get('text')
    schema = valid and isinstance(obj.get('text'), str) and 'flag' in obj and (
        obj['flag'] is None or (isinstance(obj['flag'], dict) and
        obj['flag'].get('error') in ('wrong_page','wrong_value','wrong_target','extra_write')))
    return dict(parse_valid=int(valid), schema_valid=int(schema), misplaced_text=int(misplaced))


def summarize(records, scenarios):
    out = {}
    for split in ('train', 'validation'):
        expected = [s for s in scenarios if s['split']==split]
        relevant = [r for r in records if r['split']==split]
        complete = [r for r in relevant if r['status']=='completed']
        by_scenario = defaultdict(list)
        by_task = defaultdict(list)
        for row in complete:
            by_scenario[row['scenario_id']].append(row)
            by_task[row['task_id']].append(row)
        prefix = split+'/'
        out.update({prefix+'episodes_completed':len(complete), prefix+'episodes_failed':len(relevant)-len(complete),
                    prefix+'episodes_planned':sum(s['attempts'] for s in expected),
                    prefix+'scenarios_seen':len(by_scenario),
                    prefix+'scenarios_complete':sum(len(by_scenario[s['id']])==s['attempts'] for s in expected)})
        for key in ('success','goal_ok','f1','recall','precision','dt','cc','false_flags',
                    'final_tick','n_fired','n_designed','n_polls'):
            vals = [float(r['components'][key]) for r in complete
                    if isinstance(r['components'].get(key),(int,float)) and math.isfinite(r['components'][key])]
            if vals:
                out[prefix+key] = sum(vals)/len(vals)
                if len(vals)!=len(complete):
                    out[prefix+key+'_count'] = len(vals)
        for key in ('success','f1'):
            if by_task:
                # Task-macro avoids weighting tasks with three templates more heavily.
                out[prefix+'task_macro_'+key] = sum(
                    sum(r['components'][key] for r in rows)/len(rows)
                    for rows in by_task.values())/len(by_task)
        groups = defaultdict(list)
        for r in complete:
            groups[(r['scenario_id'],r['attempt']//8)].append(r)
        full = [rs for rs in groups.values() if len(rs)==8]
        out[prefix+'complete_groups_of_8'] = len(full)
        if full:
            out[prefix+'success_mixed_group_fraction'] = sum(
                len({r['components']['success'] for r in rs})>1 for rs in full)/len(full)
            out[prefix+'shaped_mixed_group_fraction'] = sum(
                len({r['components']['success']+0.25*r['components']['f1'] for r in rs})>1
                for rs in full)/len(full)
        all_turns = sum(r['turns'] for r in complete)
        if all_turns:
            for key in ('parse_valid','schema_valid','misplaced_text','truncated'):
                out[prefix+key+'_turn_fraction'] = sum(r['format'][key] for r in complete)/all_turns
            out[prefix+'turns'] = all_turns
    return out
