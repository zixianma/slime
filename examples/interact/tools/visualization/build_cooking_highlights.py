"""Build curated Cooking RL highlight videos with local two-speaker TTS."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
from pathlib import Path
import sys
import tempfile
import wave

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from examples.interact.tools.visualization.replay_common import font, run, wrapped

HIGHLIGHTS = (
    {
        "id": "prevent-extra-bun",
        "title": "Stops an irreversible repeat before it happens",
        "split": "validation",
        "scenario": "s1_b3_medium_nops_hard_map_4_repeat_step",
        "error": "repeat_step",
        "summary": (
            "The base policy notices the extra bun after it has already been added. "
            "Update 12 repeatedly blocks the action, names the completed bun layer, "
            "and redirects the novice to cheese."
        ),
        "takeaway": "late correction after extra bun → early prevention and recovery",
        "turns": {"update-0": (181,), "update-12": (89,)},
        "callouts": {
            "update-0": "TOO LATE · the extra bun is already on the plate",
            "update-12": "EARLY BLOCK · redirects before the bun is added",
        },
    },
    {
        "id": "track-missing-cheese",
        "title": "Tracks the missing final ingredient instead of hallucinating completion",
        "split": "train",
        "scenario": "s1_b3_medium_nops_hard_map_2_serve_raw",
        "error": "serve_raw",
        "summary": (
            "The base policy insists that cheese is already plated while the novice "
            "says it is missing. The bottom-left plate in the top-down view appears "
            "to have meat on top, with no visible cheese. Update 12 guides the final "
            "cheese step and then serving."
        ),
        "takeaway": "hallucinated completion and timeout → grounded final-step guidance and win",
        "turns": {"update-0": (195, 196), "update-12": (63, 64)},
        "callouts": {
            "update-0": "STATE ERROR · claims missing cheese is already plated",
            "update-12": "STATE TRACKING · confirms cheese, then directs serving",
        },
    },
    {
        "id": "reject-foreign-corn",
        "title": "Preserves the recipe when the novice proposes corn",
        "split": "train",
        "scenario": "s1_b3_master_nops_hard_map_1_add_foreign",
        "error": "add_foreign",
        "summary": (
            "The base rollout gets stuck contradicting the novice about a correct "
            "tortilla–rice transition and times out. Update 12 reaches the later "
            "choice point, rejects corn explicitly, and completes the order."
        ),
        "takeaway": "state confusion and timeout → constraint-aware interception and win",
        "turns": {"update-0": (50,), "update-12": (128,)},
        "callouts": {
            "update-0": "FALSE ALARM · contradicts a correct tortilla → rice step",
            "update-12": "CONSTRAINT CHECK · explicitly blocks foreign corn",
        },
    },
)


def speak(piper: Path, model: Path, text: str, output: Path) -> None:
    clean = " ".join((text or "").split())
    if not clean:
        return
    run(
        [str(piper), "--model", str(model), "--output_file", str(output)],
        input_bytes=(clean + "\n").encode(),
    )


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


def focused_slide(
    source: Path,
    destination: Path,
    checkpoint: str,
    turn_index: int,
    total_turns: int,
    speaker: str,
    text: str,
    callout: str,
) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        canvas = Image.new("RGB", (1280, 720), "#0b1119")
        canvas.paste(image.crop((0, 0, 820, 720)), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((820, 0, 1279, 719), fill="#111c29", outline="#40546c", width=2)
    trained = checkpoint == "update-12"
    accent = "#76dda4" if trained else "#ff9f98"
    panel = "#173128" if trained else "#392126"
    label = "UPDATE 12 · RL" if trained else "UPDATE 0 · BASE"
    draw.text((844, 22), label, font=font(19, True), fill=accent)
    draw.text((844, 50), f"Policy turn {turn_index + 1}/{total_turns}", font=font(13), fill="#a9b8c9")
    draw.rounded_rectangle((838, 82, 1260, 150), radius=9, fill=panel, outline=accent, width=2)
    callout_face = font(16, True)
    for line_index, line in enumerate(wrapped(draw, callout, callout_face, 390)[:2]):
        draw.text((852, 94 + 23 * line_index), line, font=callout_face, fill=accent)
    role_label = "GEMINI USER SPEAKING" if speaker == "user" else "QWEN ASSISTANT SPEAKING"
    role_color = "#8bd8ff" if speaker == "user" else "#ffd18a"
    draw.text((844, 183), role_label, font=font(15, True), fill=role_color)
    face = font(25)
    lines = wrapped(draw, text, face, 390)
    while len(lines) * 34 > 450 and face.size > 17:
        face = font(face.size - 1)
        lines = wrapped(draw, text, face, 390)
    line_height = face.size + 8
    for line_index, line in enumerate(lines):
        draw.text((844, 218 + line_height * line_index), line, font=face, fill="#f2f5f9")
    draw.rounded_rectangle((838, 668, 1260, 702), radius=7, fill=role_color)
    draw.text((852, 675), "Audio and displayed text are the same utterance", font=font(13, True), fill="#081018")
    canvas.save(destination)


def render_clip(
    ffmpeg: str,
    piper: Path,
    user_model: Path,
    assistant_model: Path,
    site: Path,
    output_dir: Path,
    highlight: dict,
    checkpoint: str,
    episode: dict,
) -> Path:
    source_video = site / "videos" / checkpoint / f"{highlight['scenario']}.mp4"
    destination = output_dir / f"{highlight['id']}-{checkpoint}.mp4"
    with tempfile.TemporaryDirectory(prefix="cooking-highlight-") as temporary:
        scratch = Path(temporary)
        segments: list[Path] = []
        for position, turn_index in enumerate(highlight["turns"][checkpoint]):
            turn = episode["turns"][turn_index]
            turn_dir = scratch / f"turn-{position:02d}"
            turn_dir.mkdir()
            source_frame = turn_dir / "source.png"
            run(
                [
                    ffmpeg,
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_video),
                    # The trajectory renderer emits exactly one frame per turn.
                    # Fractional seeking (e.g. 195.1s) selects the NEXT frame
                    # of a 1-fps video; select the decoded frame index instead.
                    "-vf",
                    f"select=eq(n\\,{turn_index})",
                    "-frames:v",
                    "1",
                    str(source_frame),
                ]
            )
            for speaker, model, text in (
                ("user", user_model, turn.get("user", "")),
                ("assistant", assistant_model, turn.get("assistant", "")),
            ):
                if not (text or "").strip():
                    continue
                frame = turn_dir / f"{speaker}.png"
                audio = turn_dir / f"{speaker}.wav"
                segment = turn_dir / f"{speaker}.mp4"
                speak(piper, model, text, audio)
                focused_slide(
                    source_frame,
                    frame,
                    checkpoint,
                    turn_index,
                    len(episode["turns"]),
                    speaker,
                    " ".join(text.split()),
                    highlight["callouts"][checkpoint],
                )
                duration = wav_duration(audio) + 0.25
                run(
                    [
                        ffmpeg,
                        "-loglevel",
                        "error",
                        "-y",
                        "-loop",
                        "1",
                        "-i",
                        str(frame),
                        "-i",
                        str(audio),
                        "-t",
                        f"{duration:.3f}",
                        "-r",
                        "8",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "27",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "96k",
                        "-movflags",
                        "+faststart",
                        str(segment),
                    ]
                )
                segments.append(segment)
        concat_file = scratch / "segments.txt"
        concat_file.write_text("".join(f"file '{path}'\n" for path in segments))
        run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(destination),
            ]
        )
    return destination


def render_screenshots(ffmpeg: str, site: Path, output: Path, highlight: dict, checkpoint: str, episode: dict) -> None:
    directory = output / 'screenshots'
    directory.mkdir(exist_ok=True)
    video = site / 'videos' / checkpoint / f"{highlight['scenario']}.mp4"
    with tempfile.TemporaryDirectory(prefix='cooking-screenshot-') as temporary:
        for index in highlight['turns'][checkpoint]:
            if episode['turns'][index]['index'] != index:
                raise ValueError('Episode turn index does not match video frame index')
            source = Path(temporary) / f'{index}.png'
            run([ffmpeg, '-loglevel', 'error', '-y', '-i', str(video),
                 '-vf', f'select=eq(n\\,{index})', '-frames:v', '1', str(source)])
            with Image.open(source) as original:
                screenshot = original.convert('RGB').crop((20, 208, 812, 510))
                if highlight['id'] == 'track-missing-cheese' and checkpoint == 'update-0':
                    # The plate is at the bottom-left of the top-down camera,
                    # not in the first-person camera's current field of view.
                    plate = original.crop((110, 426, 167, 468)).resize((228, 168), Image.Resampling.NEAREST)
                    canvas = Image.new('RGB', (792, 496), '#101925')
                    canvas.paste(screenshot, (0, 0))
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((90, 218, 147, 260), outline='#69c8ff', width=3)
                    canvas.paste(plate, (12, 316))
                    draw.text((258, 328), 'Bottom-left plate', font=font(22, True), fill='#69c8ff')
                    draw.text((258, 367), 'Enlarged from this same frame', font=font(17), fill='#c4cfdd')
                    screenshot = canvas
                screenshot.save(directory / f"{highlight['id']}-{checkpoint}-turn-{index + 1}.png")


def card(highlight: dict, episodes: dict[str, dict]) -> str:
    task_instruction = episodes["update-0"]["task_instruction"]
    if task_instruction != episodes["update-12"]["task_instruction"]:
        raise ValueError(f"task instructions differ for {highlight['scenario']}")

    def panel(checkpoint: str, label: str) -> str:
        episode = episodes[checkpoint]
        components = episode["reward_components"]
        outcome_class = "good" if episode["success"] else "bad"
        note = (
            '<p>Look at the bottom-left plate in the top-down view: '
            'it appears topped with meat, with no visible yellow cheese.</p>'
            if highlight['id'] == 'track-missing-cheese' and checkpoint == 'update-0'
            else ''
        )
        return f"""
        <section class="policy">
          <h3>{html.escape(label)}</h3>
          <div class="policy-callout {'good' if checkpoint == 'update-12' else 'bad'}">{html.escape(highlight['callouts'][checkpoint])}</div>
          <div class="metrics">
            <span class="{outcome_class}">{html.escape(episode['outcome'])}</span>
            <span>{len(episode['turns'])} turns</span>
            <span>{components.get('false_flags', 0)} false flags</span>
          </div>
          <video controls preload="metadata" src="clips/{highlight['id']}-{checkpoint}.mp4?v=exact-frame-1"></video>
{note}
        </section>"""

    return f"""
    <article class="highlight" data-split="{highlight['split']}">
      <div class="eyebrow">{highlight['split']} · {highlight['error']}</div>
      <h2>{html.escape(highlight['title'])}</h2>
      <div class="task-instruction"><b>Original task instruction</b><p>{html.escape(task_instruction)}</p></div>
      <p>{html.escape(highlight['summary'])}</p>
      <div class="takeaway">{html.escape(highlight['takeaway'])}</div>
      <div class="pair">{panel('update-0', 'Update 0 · base')}{panel('update-12', 'Update 12 · RL')}</div>
      <details><summary>Scenario ID and selected policy turns</summary>
        <code>{highlight['scenario']}</code><br>
        Update 0: {', '.join(str(i + 1) for i in highlight['turns']['update-0'])} ·
        Update 12: {', '.join(str(i + 1) for i in highlight['turns']['update-12'])}
      </details>
    </article>"""


def page(cards: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cooking RL · Curated policy improvements</title>
<style>
:root{{font:15px Inter,ui-sans-serif,system-ui,sans-serif;color:#e8edf5;background:#0b1119;--line:#304057;--panel:#141f2d;--muted:#a9b8c9;--blue:#69c8ff;--green:#76dda4;--red:#ff9f98}}*{{box-sizing:border-box}}body{{margin:0}}header{{padding:28px max(24px,calc((100vw - 1320px)/2));background:linear-gradient(135deg,#1b2d43,#101926);border-bottom:1px solid var(--line)}}h1{{font-size:31px;margin:8px 0}}h2{{font-size:23px;margin:7px 0}}h3{{margin:0}}p{{line-height:1.55;color:#c4cfdd;max-width:950px}}a{{color:var(--blue)}}.pill,.metrics span{{display:inline-block;padding:5px 9px;border:1px solid #455c76;border-radius:999px;font-size:12px}}.voice{{margin-top:12px;color:var(--muted)}}.voice b:first-child{{color:#8bd8ff}}.voice b:last-child{{color:#ffd18a}}nav{{margin-top:14px}}button{{font:inherit;color:#e8edf5;background:#203149;border:1px solid #4a607a;border-radius:7px;padding:7px 11px;margin-right:5px;cursor:pointer}}button.active{{border-color:var(--blue);background:#29435c}}main{{max-width:1380px;margin:auto;padding:24px}}.highlight{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:22px}}.eyebrow{{color:var(--blue);text-transform:uppercase;letter-spacing:.08em;font-size:12px}}.task-instruction{{max-width:1000px;margin:12px 0;padding:11px 14px;border-left:3px solid var(--blue);background:#172638;border-radius:0 8px 8px 0}}.task-instruction b{{font-size:12px;color:var(--blue);text-transform:uppercase;letter-spacing:.06em}}.task-instruction p{{margin:5px 0 0;color:#eef4fb}}.takeaway{{display:inline-block;color:var(--green);background:#142c27;border:1px solid #2d6555;border-radius:8px;padding:8px 11px;margin:2px 0 16px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.policy{{min-width:0;background:#101925;border:1px solid #2e4056;border-radius:11px;padding:12px}}.policy-callout{{font-weight:700;border-left:3px solid currentColor;padding:7px 9px;margin:9px 0;background:#172230}}.good{{color:var(--green)}}.bad{{color:var(--red)}}.metrics{{display:flex;gap:6px;flex-wrap:wrap;margin:9px 0}}video{{display:block;width:100%;aspect-ratio:16/9;background:#05080c;border-radius:8px}}details{{margin-top:12px;color:var(--muted)}}summary{{cursor:pointer}}code{{font-size:12px}}@media(max-width:900px){{.pair{{grid-template-columns:1fr}}header{{padding:20px}}main{{padding:14px}}}}
</style></head><body>
<header><span class="pill">CookingSim</span> <span class="pill">Qwen3.5-4B</span> <span class="pill">Gemini 3.7 Flash novice</span>
<h1>What changed after RL?</h1>
<p>Curated side-by-side video segments from matched Update 0 and Update 12 scenarios. Images are aligned to their displayed policy turns. Because Gemini and the simulator are stochastic, these are qualitative examples rather than deterministic causal replays.</p>
<div class="voice"><b>Gemini user:</b> Lessac voice &nbsp;·&nbsp; <b>Qwen assistant:</b> Ryan voice. Each clip displays the utterance currently being spoken.</div>
<nav><button class="active" data-filter="all">All</button><button data-filter="validation">Validation</button><button data-filter="train">Training</button> &nbsp; <a href="../">Open full 149-pair replay</a></nav></header>
<main>{cards}</main>
<script>document.querySelectorAll('[data-filter]').forEach(button=>button.onclick=()=>{{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));button.classList.add('active');document.querySelectorAll('.highlight').forEach(card=>card.hidden=button.dataset.filter!=='all'&&card.dataset.split!==button.dataset.filter)}})</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--piper", type=Path, required=True)
    parser.add_argument("--voices", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--render-clips", action="store_true", help="Also regenerate optional narrated video assets.")
    args = parser.parse_args()
    output = args.site / "highlights"
    clips = output / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    user_model = args.voices / "en_US-lessac-medium.onnx"
    assistant_model = args.voices / "en_US-ryan-medium.onnx"
    entries = []
    jobs = []
    manifest = []
    for highlight in HIGHLIGHTS:
        episodes = {
            checkpoint: json.loads(
                (args.site / "checkpoints" / checkpoint / "episodes" / f"{highlight['scenario']}.json").read_text()
            )
            for checkpoint in ("update-0", "update-12")
        }
        for checkpoint, episode in episodes.items():
            destination = clips / f"{highlight['id']}-{checkpoint}.mp4"
            if args.render_clips and (args.force or not destination.is_file()):
                jobs.append((highlight, checkpoint, episode))
        entries.append((highlight, episodes))
        manifest.append(highlight)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                render_clip,
                ffmpeg,
                args.piper,
                user_model,
                assistant_model,
                args.site,
                clips,
                highlight,
                checkpoint,
                episode,
            ): (highlight["id"], checkpoint)
            for highlight, checkpoint, episode in jobs
        }
        for future in as_completed(futures):
            highlight_id, checkpoint = futures[future]
            future.result()
            print(f"rendered {highlight_id} {checkpoint}", flush=True)
    cards = [card(highlight, episodes) for highlight, episodes in entries]
    (output / "index.html").write_text(page("".join(cards)))
    (output / "highlights.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
