"""Render a small HTML comparison of native ScreenSim rollout observations."""
import argparse
import base64
import glob
import html
import json
import os
from pathlib import Path


def reports(root, start=0, count=None):
    paths = sorted(glob.glob(str(Path(root) / "episodes/*/report.json")), key=os.path.getmtime)
    if count is not None:
        paths = paths[start:start + count]
    return paths


def choose(paths, tasks):
    selected = []
    for task in tasks:
        candidates = []
        for path in paths:
            report = json.loads(Path(path).read_text())
            if report["task"] == task:
                candidates.append((report["success"], path))
        # Show one success and one failure when available; this keeps the
        # comparison diagnostic instead of selecting only favorable examples.
        for desired in (1, 0):
            match = next((p for success, p in candidates if success == desired), None)
            if match and match not in selected:
                selected.append(match)
    return selected


def decision_frames(path):
    rows = [json.loads(line)["decision"] for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        return None, None, "no decisions"
    report = json.loads(Path(path).with_name("report.json").read_text())
    with_images = [row for row in rows if row["observation"].get("images")]
    if not with_images:
        return None, None, "no images"
    first = with_images[0]["observation"]["images"][-1]
    target_tick = None
    if report.get("beats"):
        target_tick = report["beats"][0].get("detected_tick")
    if target_tick is None:
        target = rows[-1]
    else:
        target = min(with_images, key=lambda row: abs(float(row.get("tick", 0)) - float(target_tick)))
    return first, target["observation"]["images"][-1], target.get("tick", "?")


def img(src, alt):
    if not src:
        return '<div class="missing">no image captured</div>'
    return f'<img src="{html.escape(src, quote=True)}" alt="{html.escape(alt, quote=True)}">'


def card(label, path):
    report = json.loads(Path(path).read_text())
    decision_path = Path(path).with_name("decisions.jsonl")
    first, detected, tick = decision_frames(decision_path)
    def metric(name):
        value = report.get(name)
        return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
    metrics = (f"success={report.get('success')} | F1={metric('f1')} | "
               f"precision={metric('precision')} | recall={metric('recall')} | dt={metric('dt')}")
    transcript = " ".join(text for who, text in report.get("transcript", []) if who == "assistant")
    return f"""
    <article><h3>{html.escape(label)} — {html.escape(report['task'])}</h3>
    <p class="metrics">{html.escape(metrics)}; first detected tick={html.escape(str(tick))}</p>
    <div class="frames"><div><b>Initial observation</b>{img(first, 'initial observation')}</div>
    <div><b>Near first detection</b>{img(detected, 'observation near first detection')}</div></div>
    <details><summary>Assistant transcript</summary><p>{html.escape(transcript)}</p></details>
    </article>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0", type=Path, required=True)
    parser.add_argument("--step12", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = ["lost_phone_lockdown", "parent_phone_readability", "keyboard_stop_correcting"]
    old = choose(reports(args.step0), tasks)
    # The step-12 training batch is the fourth training chunk: steps 9, 10,
    # 11, and 12 correspond to chunks 0, 48, 96, and 192. Chunks 144 and 336
    # are the two validation batches and are intentionally excluded.
    new = choose(reports(args.step12, start=192, count=48), tasks)
    sections = []
    for p in old:
        sections.append(card("step 0", p))
    for p in new:
        sections.append(card("step 12", p))
    source = html.escape(str(args.step0)) + "<br>" + html.escape(str(args.step12))
    document = f"""<!doctype html><meta charset="utf-8"><title>ScreenSim rollout comparison</title>
    <style>
    body {{ font: 15px system-ui,sans-serif; margin: 2rem; background:#f6f7f9; color:#17202a }}
    h1 {{ margin-bottom:.2rem }} .note {{ color:#52606d }}
    article {{ background:white; padding:1rem; margin:1rem 0; border-radius:10px; box-shadow:0 1px 4px #ccd }}
    .metrics {{ font-family:ui-monospace,monospace; color:#245 }} .frames {{ display:flex; gap:1rem; flex-wrap:wrap }}
    .frames div {{ display:flex; flex-direction:column; gap:.35rem; font-weight:600 }}
    img {{ max-width:260px; max-height:520px; border:1px solid #bbc; background:#eee }}
    details {{ margin-top:.8rem }} p {{ max-width:1000px; line-height:1.45 }}
    </style><h1>ScreenSim rollouts: step 0 vs step 12</h1>
    <p class="note">Matched task families; one successful and one failed episode where available. Images are the native observations sent to Qwen. Step 12 is the training batch, not validation. Sources:<br>{source}</p>
    {''.join(sections)}"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    print(json.dumps({"output": str(args.output), "cards": len(sections), "step0": len(old), "step12": len(new)}))


if __name__ == "__main__":
    main()
