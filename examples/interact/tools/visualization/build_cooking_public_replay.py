"""Package the Cooking Update 0 vs. Update 12 replay for public static hosting."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg


CHECKPOINTS = ("update-0", "update-12")


def compact_episode(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text())
    for turn in payload.get("turns", []):
        # The MP4 contains the exact visual observation. Avoid duplicating the
        # large sampled-image archive and verbose debug state in public hosting.
        turn["frame"] = None
        turn["progress"] = None
        turn["raw"] = ""
    destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-crf", type=int)
    parser.add_argument("--video-workers", type=int, default=6)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source / "index.html", args.output / "index.html")
    metadata = json.loads((args.source / "metadata.json").read_text())
    metadata["checkpoints"] = [
        checkpoint for checkpoint in metadata["checkpoints"]
        if checkpoint["directory"] in CHECKPOINTS
    ]
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "robots.txt").write_text("User-agent: *\nDisallow: /\n")

    for checkpoint in CHECKPOINTS:
        source_dir = args.source / "checkpoints" / checkpoint
        output_dir = args.output / "checkpoints" / checkpoint
        episodes_dir = output_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / "index.json", output_dir / "index.json")
        for episode in (source_dir / "episodes").glob("*.json"):
            compact_episode(episode, episodes_dir / episode.name)

    source_index = json.loads((args.source / "videos/index.json").read_text())
    video_index: dict[str, str] = {}
    video_jobs: list[tuple[Path, Path]] = []
    for key, relative in source_index.items():
        if key.split("/", 1)[0] not in CHECKPOINTS:
            continue
        source_video = args.source / relative
        if not source_video.is_file() or source_video.stat().st_size <= 1024:
            continue
        output_video = args.output / relative
        output_video.parent.mkdir(parents=True, exist_ok=True)
        video_jobs.append((source_video, output_video))
        video_index[key] = relative
    if args.video_crf is None:
        for source_video, output_video in video_jobs:
            shutil.copyfile(source_video, output_video)
    else:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        def transcode(job: tuple[Path, Path]) -> None:
            source_video, output_video = job
            subprocess.run([
                ffmpeg, "-loglevel", "error", "-y", "-i", str(source_video),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.video_crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_video),
            ], check=True)

        with ThreadPoolExecutor(max_workers=args.video_workers) as pool:
            list(pool.map(transcode, video_jobs))
    videos_dir = args.output / "videos"
    videos_dir.mkdir(exist_ok=True)
    (videos_dir / "index.json").write_text(json.dumps(video_index, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "videos": len(video_index),
        "bytes": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()),
    }))


if __name__ == "__main__":
    main()
