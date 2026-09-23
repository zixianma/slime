"""Small shared helpers for offline replay and video builders."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

from PIL import ImageDraw, ImageFont


def run(command: list[str], *, input_bytes: bytes | None = None) -> None:
    subprocess.run(command, input=input_bytes, check=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    face: ImageFont.FreeTypeFont,
    width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in (text or "").splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if draw.textlength(candidate, font=face) <= width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def cooking_response(raw: str) -> tuple[str, object, object]:
    cleaned = (raw or "").replace("<|im_end|>", "").strip()
    candidates = [cleaned]
    if "{" in cleaned and "}" in cleaned:
        candidates.append(cleaned[cleaned.index("{") : cleaned.rindex("}") + 1])
    value: object = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            break
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if isinstance(value, dict):
        return str(value.get("text") or ""), value.get("flag"), value.get("cook_progress")
    if isinstance(value, str):
        return value, None, None
    return cleaned, None, None


def cooking_user_text(prompt: str) -> str:
    marker = "== THE COOK JUST SAID =="
    if marker not in prompt:
        return ""
    value = prompt.split(marker, 1)[1].split("\n\nBelow are", 1)[0].strip()
    if value.startswith('"') and value.endswith('"'):
        try:
            return str(json.loads(value))
        except json.JSONDecodeError:
            return value[1:-1]
    return value


def cooking_task_instruction(system: str) -> str:
    match = re.search(r"The cook was asked to make:\s*(.+?)\n", system)
    return match.group(1).strip() if match else "See scenario identifier and recipe state."


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"{description} does not exist: {path}")
    return path
