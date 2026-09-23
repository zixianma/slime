"""Render Cooking trajectories as frame-plus-dialogue MP4 videos."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from examples.interact.tools.visualization.build_cooking_full_suite_replay import partial_rows
from examples.interact.tools.visualization.replay_common import (
    cooking_response as parse_response,
    cooking_user_text as current_user_text,
    font,
    wrapped,
)


CHECKPOINTS = ("update-0", "update-3-best", "update-12")
CANVAS = (1280, 720)
IMAGE_BOX = (22, 70, 810, 650)
TEXT_X = 840
TEXT_WIDTH = 410


def result_rows(root: Path, checkpoint: str) -> dict[str, dict]:
    path = root / checkpoint / "audit/result.json"
    if path.is_file():
        rows = json.loads(path.read_text())["scenarios"]
    else:
        baseline_path = root / "update-0/audit/result.json"
        if not baseline_path.is_file():
            return {}
        baseline = {
            row["scenario_id"]: row
            for row in json.loads(baseline_path.read_text())["scenarios"]
        }
        rows = partial_rows(root / checkpoint, baseline)
    return {row["scenario_id"]: row for row in rows}


def fit_dialogue(draw: ImageDraw.ImageDraw, user: str, assistant: str) -> tuple[ImageFont.FreeTypeFont, list[str], list[str]]:
    for size in range(20, 12, -1):
        face = font(size)
        user_lines = wrapped(draw, user or "(No new spoken message)", face, TEXT_WIDTH)
        assistant_lines = wrapped(draw, assistant or "(Silent)", face, TEXT_WIDTH)
        line_height = size + 5
        if (len(user_lines) + len(assistant_lines)) * line_height <= 520:
            return face, user_lines, assistant_lines
    face = font(13)
    user_lines = wrapped(draw, user or "(No new spoken message)", face, TEXT_WIDTH)
    assistant_lines = wrapped(draw, assistant or "(Silent)", face, TEXT_WIDTH)
    available = 36
    if len(user_lines) + len(assistant_lines) > available:
        assistant_lines = assistant_lines[: max(3, available - len(user_lines))]
        assistant_lines[-1] += " …"
    return face, user_lines, assistant_lines


def render_slide(record: dict, turn: int, total: int, label: str, scenario: str) -> bytes:
    decision = record["decision"]
    observation = decision["observation"]
    assistant, flag, _ = parse_response(record.get("response", ""))
    user = current_user_text(observation.get("user", ""))
    canvas = Image.new("RGB", CANVAS, "#0c121b")
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 18), f"{label} · {scenario}", font=font(21, True), fill="#e8edf5")
    draw.text((TEXT_X, 20), f"Turn {turn + 1}/{total} · cooking tick {decision.get('tick', '—')}",
              font=font(16, True), fill="#a9b8c9")

    images = observation.get("images") or []
    if images:
        raw = base64.b64decode(images[-1].split(",", 1)[1])
        with Image.open(BytesIO(raw)) as source:
            source = source.convert("RGB")
            left, top, right, bottom = IMAGE_BOX
            scale = min((right - left) / source.width, (bottom - top) / source.height)
            resized = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS)
            x = left + (right - left - resized.width) // 2
            y = top + (bottom - top - resized.height) // 2
            canvas.paste(resized, (x, y))
            draw.rounded_rectangle((x - 2, y - 2, x + resized.width + 2, y + resized.height + 2),
                                   radius=7, outline="#4a607a", width=2)
    else:
        draw.text((260, 340), "No visual observation", font=font(22), fill="#a9b8c9")

    face, user_lines, assistant_lines = fit_dialogue(draw, user, assistant)
    line_height = face.size + 5
    y = 70
    draw.text((TEXT_X, y), "GEMINI USER", font=font(14, True), fill="#69c8ff")
    y += 25
    for line in user_lines:
        draw.text((TEXT_X, y), line, font=face, fill="#e8edf5")
        y += line_height
    y += 18
    draw.text((TEXT_X, y), "QWEN ASSISTANT", font=font(14, True), fill="#69c8ff")
    y += 25
    for line in assistant_lines:
        draw.text((TEXT_X, y), line, font=face, fill="#e8edf5")
        y += line_height
    if flag is not None:
        flag_text = "FLAG: " + json.dumps(flag, ensure_ascii=False)
        draw.rounded_rectangle((TEXT_X - 8, 656, 1260, 704), radius=8, fill="#45262b", outline="#ff9f98")
        draw.text((TEXT_X, 670), flag_text[:52], font=font(14, True), fill="#ffb1aa")
    return canvas.tobytes()


def render_video(eval_root: Path, site: Path, checkpoint: str, row: dict, force: bool) -> tuple[str, str]:
    scenario = row["scenario_id"]
    destination = site / "videos" / checkpoint / f"{scenario}.mp4"
    # A failed ffmpeg process can leave only the MP4 header behind. Do not let
    # such a partial file suppress a later retry.
    if destination.is_file() and destination.stat().st_size > 1024 and not force:
        return checkpoint, scenario
    destination.parent.mkdir(parents=True, exist_ok=True)
    decisions = eval_root / checkpoint / "episodes" / row["episode_id"] / "decisions.jsonl"
    total = sum(1 for _ in decisions.open())
    label = checkpoint.replace("-best", "").replace("update-", "Update ")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{CANVAS[0]}x{CANVAS[1]}", "-framerate", "1",
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        with decisions.open() as stream:
            for turn, line in enumerate(stream):
                try:
                    process.stdin.write(render_slide(json.loads(line), turn, total, label, scenario))
                except BrokenPipeError as error:
                    stderr = process.stderr.read().decode() if process.stderr else ""
                    code = process.wait()
                    raise RuntimeError(
                        f"ffmpeg exited early ({code}) for {checkpoint}/{scenario}: {stderr}"
                    ) from error
        process.stdin.close()
        stderr = process.stderr.read().decode() if process.stderr else ""
        code = process.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed for {checkpoint}/{scenario}: {stderr}")
    except BaseException:
        process.kill()
        raise
    return checkpoint, scenario


def write_index(site: Path) -> None:
    entries = {}
    for path in sorted((site / "videos").glob("*/*.mp4")):
        key = f"{path.parent.name}/{path.stem}"
        entries[key] = str(path.relative_to(site))
    (site / "videos/index.json").write_text(json.dumps(entries, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--left", choices=CHECKPOINTS, default="update-0")
    parser.add_argument("--right", choices=CHECKPOINTS, default="update-3-best")
    parser.add_argument("--scenario")
    parser.add_argument("--all-overlap", action="store_true")
    parser.add_argument(
        "--site-overlap",
        action="store_true",
        help="Limit --all-overlap to scenarios already exposed by both checkpoint indexes in the site.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    left, right = result_rows(args.eval_root, args.left), result_rows(args.eval_root, args.right)
    overlap = sorted(left.keys() & right.keys())
    if args.site_overlap:
        def site_scenarios(checkpoint: str) -> set[str]:
            index_path = args.site / "checkpoints" / checkpoint / "index.json"
            if not index_path.is_file():
                raise SystemExit(f"site checkpoint index does not exist: {index_path}")
            return {row["scenario"] for row in json.loads(index_path.read_text())}

        overlap = sorted(set(overlap) & site_scenarios(args.left) & site_scenarios(args.right))
    if args.scenario:
        if args.scenario not in overlap:
            raise SystemExit(f"scenario is not present in both completed checkpoint results: {args.scenario}")
        overlap = [args.scenario]
    elif not args.all_overlap:
        raise SystemExit("pass --scenario or --all-overlap")
    jobs = [(args.left, left[name]) for name in overlap] + [(args.right, right[name]) for name in overlap]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render_video, args.eval_root, args.site, checkpoint, row, args.force)
                   for checkpoint, row in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            checkpoint, scenario = future.result()
            print(f"rendered {count}/{len(jobs)} {checkpoint}/{scenario}", flush=True)
    write_index(args.site)


if __name__ == "__main__":
    main()
