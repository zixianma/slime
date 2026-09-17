"""Render explicitly paired native ScreenSim rollouts for two training steps."""
import argparse
import base64
import hashlib
import glob
import html
import json
import os
from collections import defaultdict
from pathlib import Path


def batch(root, start, count):
    paths = sorted(glob.glob(str(Path(root) / "episodes/*/report.json")), key=os.path.getmtime)
    return [json.loads(Path(p).read_text()) | {"_report": p}
            for p in paths[start:start + count]]


def pair_by_task(old, new):
    grouped = defaultdict(list)
    for report in old:
        grouped[report["task"]].append(report)
    grouped_new = defaultdict(list)
    for report in new:
        grouped_new[report["task"]].append(report)
    pairs = []
    for task in sorted(grouped):
        if len(grouped[task]) != len(grouped_new[task]):
            raise ValueError(f"unbalanced task batch for {task}")
        pairs.extend((task, i + 1, grouped[task][i], grouped_new[task][i])
                     for i in range(len(grouped[task])))
    return pairs


def observations(report):
    path = Path(report["_report"]).with_name("decisions.jsonl")
    rows = [json.loads(line)["decision"] for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["observation"].get("images")]
    if not rows:
        return None, None
    first = rows[0]["observation"]["images"][-1]
    detections = [beat.get("detected_tick") for beat in report.get("beats", [])
                  if beat.get("detected_tick") is not None]
    target_tick = detections[0] if detections else rows[-1].get("tick", 0)
    near = min(rows, key=lambda row: abs(float(row.get("tick", 0)) - float(target_tick)))
    return first, near["observation"]["images"][-1]


def task_instruction(report):
    path = Path(report["_report"]).with_name("decisions.jsonl")
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        user = json.loads(line)["decision"]["observation"].get("user", "")
        marker = "== WHAT THEY'RE TRYING TO DO =="
        end = "== HOW IT IS DONE (reference) =="
        if marker in user:
            text = user.split(marker, 1)[1].split(end, 1)[0].strip()
            return text
    return "Task instruction unavailable"


def image(src, alt, asset_dir):
    if not src:
        return '<span class="missing">no image</span>'
    if src.startswith("data:image/"):
        header, encoded = src.split(",", 1)
        suffix = header.split("/", 1)[1].split(";", 1)[0]
        filename = hashlib.sha256(src.encode()).hexdigest() + "." + suffix
        target = asset_dir / filename
        if not target.exists():
            target.write_bytes(base64.b64decode(encoded))
        src = asset_dir.name + "/" + filename
    return f'<img loading="lazy" decoding="async" src="{html.escape(src, quote=True)}" alt="{html.escape(alt, quote=True)}">'


def pct(value):
    return f"{100 * value:.1f}%" if isinstance(value, (int, float)) else "—"


def delta(a, b):
    return b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else float("nan")


def checklist(report):
    beats = [beat for beat in report.get("beats", []) if not beat.get("skipped")]
    checks = [
        ("task completed in time", bool(report.get("success"))),
        ("all injected mistakes detected", bool(beats) and all(b.get("detected") for b in beats)),
        ("mistake type correct", bool(beats) and all(b.get("type_ok") for b in beats)),
        ("mistakes prevented", bool(beats) and all(b.get("prevented") for b in beats)),
        ("all warnings accepted", bool(beats) and all(b.get("accepted") for b in beats)),
        ("no false flags", report.get("false_flags") == 0),
        ("no engine fallback repair", report.get("fallback_repairs", 0) == 0),
    ]
    return "<ul class='checklist'>" + "".join(
        f"<li class='{'pass' if ok else 'fail'}'>{'✓' if ok else '✗'} {html.escape(label)}</li>"
        for label, ok in checks) + "</ul>"


def cell(report, label, asset_dir):
    first, near = observations(report)
    timeline = []
    for event in report.get("timeline", []):
        kind = event.get("kind", "event")
        who = event.get("who", "")
        text = event.get("text", "")
        if kind == "gesture":
            text = f"{text} ({'slip' if event.get('slip') else 'normal'})"
        elif kind == "fallback_repair":
            who, text = "engine", f"canonical repair: {text}"
        elif kind == "human_fix":
            who, text = "user", f"repair gesture: {text}"
        if text or kind not in {"say", "move"}:
            timeline.append(f"<li><b>tick {event.get('t', '?')} · {html.escape(who or kind)}</b>: {html.escape(str(text))}</li>")
    timeline_html = "<ol>" + "".join(timeline) + "</ol>"
    return f"""<td><div class="label">{label}</div>
      <div class="metric"><b>{'SUCCESS' if report['success'] else 'FAIL'}</b> · F1 {report.get('f1', 0):.3f} · P {pct(report.get('precision'))} · R {pct(report.get('recall'))} · Δt {report.get('dt', '—')}</div>
      <div class="shots"><figure>{image(first, label + ' initial', asset_dir)}<figcaption>initial</figcaption></figure>
      <figure>{image(near, label + ' near first detection', asset_dir)}<figcaption>near first detection</figcaption></figure></div>
      {checklist(report)}
      <details><summary>exact task instruction</summary><p>{html.escape(task_instruction(report))}</p></details>
      <details><summary>exact turn-by-turn interaction</summary>{timeline_html}</details></td>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0", type=Path, required=True)
    parser.add_argument("--step12", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asset_dir = args.output.parent / (args.output.stem + "-assets")
    asset_dir.mkdir(parents=True, exist_ok=True)
    pairs = pair_by_task(batch(args.step0, 0, 48), batch(args.step12, 192, 48))
    rows = []
    grouped = {"up": [], "down": [], "same": []}
    for task, index, old, new in pairs:
        success_delta = new["success"] - old["success"]
        cls = "up" if success_delta > 0 else "down" if success_delta < 0 else "same"
        f1_delta, p_delta, r_delta = delta(old.get("f1"), new.get("f1")), delta(old.get("precision"), new.get("precision")), delta(old.get("recall"), new.get("recall"))
        fmt = lambda x: f"{x:+.3f}" if x == x else "—"
        row = f"<tr class='{cls}' data-category='{cls}' data-task='{html.escape(task, quote=True)}'><th>{html.escape(task)}<br><small>episode {index}/8</small></th>{cell(old, 'step 0', asset_dir)}{cell(new, 'step 12', asset_dir)}<td class='delta'><b>{'↑' if success_delta > 0 else '↓' if success_delta < 0 else '='} {success_delta:+.0f}</b><br>F1 Δ {fmt(f1_delta)}<br>P Δ {fmt(p_delta)}<br>R Δ {fmt(r_delta)}</td></tr>"
        rows.append(row)
        grouped[cls].append(row)
    tasks = sorted({task for task, _, _, _ in pairs})
    labels = {"up": "Improvements", "down": "Regressions", "same": "Neutral"}
    tables = []
    for category in ("up", "down", "same"):
        tables.append(f"<section class='group' data-group='{category}'><h2>{labels[category]} <span class='count'>({len(grouped[category])})</span></h2><table><thead><tr><th>task / pair</th><th>step 0</th><th>step 12</th><th>change</th></tr></thead><tbody>{''.join(grouped[category])}</tbody></table></section>")
    task_options = "<option value='all'>All tasks</option>" + "".join(f"<option value='{html.escape(t, quote=True)}'>{html.escape(t)}</option>" for t in tasks)
    doc = f"""<!doctype html><meta charset='utf-8'><title>Direct ScreenSim comparison</title>
    <style>body{{font:14px system-ui,sans-serif;margin:1.5rem;background:#f4f6f8;color:#17202a}}h1{{margin-bottom:.2rem}}h2{{margin:2rem 0 .2rem}}.note{{color:#52606d}}.controls{{position:sticky;top:0;background:#f4f6f8;padding:.7rem 0;z-index:2}}select,button{{font:inherit;padding:.35rem .55rem;margin-right:.5rem}}table{{border-collapse:separate;border-spacing:0 10px;width:100%}}th,td{{background:white;padding:10px;vertical-align:top;border-top:1px solid #d7dde3;border-bottom:1px solid #d7dde3}}th{{width:13%;text-align:left}}td{{width:40%}}td.delta{{width:7%;text-align:center;color:#52606d}}tr.up td.delta,tr.up th{{background:#ecfdf3;color:#067647}}tr.down td.delta,tr.down th{{background:#fff1f2;color:#b42318}}tr.same td.delta,tr.same th{{background:#f8fafc}}.label{{font-weight:700;margin-bottom:4px}}.metric{{font-family:ui-monospace,monospace;font-size:12px;margin-bottom:6px}}.shots{{display:flex;gap:8px;flex-wrap:wrap}}figure{{margin:0}}img{{max-width:155px;max-height:310px;border:1px solid #bbc;background:#eee}}figcaption{{font-size:11px;color:#667085}}.checklist{{list-style:none;padding:0;margin:.5rem 0;font-size:12px;line-height:1.5}}.pass{{color:#067647}}.fail{{color:#b42318}}details{{margin-top:6px}}p{{line-height:1.4}}.hidden{{display:none!important}}.count{{font-weight:400;color:#667085;font-size:.8em}}</style>
    <h1>Direct paired comparison: ScreenSim step 0 vs. step 12</h1>
    <p class='note'>Each row pairs the same task and within-batch episode index (8 episodes/task). Green means step 12 succeeded where step 0 failed; red is the reverse. This uses training batches, not validation. Each cell now has an automatic checklist based on the native report: completion, detection, type correctness, prevention, acceptance, false flags, and engine fallback. These are engine-verifiable; they do not establish that the natural-language repair was persuasive because the current user is scripted. Expand “exact task instruction” and “exact turn-by-turn interaction” for details. In the timeline, <b>assistant</b> is Qwen’s spoken guidance, <b>user</b> is the scripted person’s accept/ignore or repair action, <b>gesture</b> is the person’s phone action, and <b>engine</b> is automatic canonical repair.</p>
    <div class='controls'><label>Show: <select id='category' onchange='filterRows()'><option value='all'>All outcomes</option><option value='up'>Improvements</option><option value='down'>Regressions</option><option value='same'>Neutral</option></select></label><label> Task: <select id='task' onchange='filterRows()'>{task_options}</select></label><button onclick="document.getElementById('category').value='all';document.getElementById('task').value='all';filterRows()">Reset</button></div>
    {''.join(tables)}
    <script>function filterRows(){{const c=document.getElementById('category').value,t=document.getElementById('task').value;document.querySelectorAll('tr[data-category]').forEach(r=>r.classList.toggle('hidden',(c!=='all'&&r.dataset.category!==c)||(t!=='all'&&r.dataset.task!==t)));document.querySelectorAll('.group').forEach(g=>{{const visible=[...g.querySelectorAll('tr[data-category]')].some(r=>!r.classList.contains('hidden'));g.classList.toggle('hidden',!visible)}})}}</script>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(doc)
    print(json.dumps({"output": str(args.output), "paired_rows": len(rows)}))


if __name__ == "__main__":
    main()
