#!/usr/bin/env python3
"""Render semantic highlight clips with burned-in subtitles."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import Transcript
from src.subtitles import build_ass_subtitles, subtitle_style_from_preset
from src.utils import ensure_dirs, read_json


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if p.returncode != 0:
        tail = (p.stderr or "")[-1200:]
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{tail}")


def safe_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "part"


def ffmpeg_subtitles_filter_path(path: Path) -> str:
    # Minimal escaping for ffmpeg subtitles filter.
    s = str(path)
    s = s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return s


def collect_text_snippet(transcript: Transcript, start: float, end: float, max_len: int = 220) -> str:
    parts: list[str] = []
    for seg in transcript.segments:
        if seg.end <= start or seg.start >= end:
            continue
        t = " ".join(seg.text.split())
        if t:
            parts.append(t)
    full = " ".join(parts).strip()
    if len(full) <= max_len:
        return full
    return full[: max_len - 1].rstrip() + "…"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render semantic highlight clips")
    p.add_argument("--input-dir", default="/Users/mac/Documents/reels_maker/highlights")
    p.add_argument("--segments-json", default="/Users/mac/Documents/reels_maker/highlights/semantic_segments.json")
    p.add_argument("--transcripts-dir", default="/Users/mac/Documents/reels_maker/storage/transcripts")
    p.add_argument("--output-dir", default="/Users/mac/Documents/reels_maker/highlights/semantic_cuts")
    p.add_argument("--subtitle-max-chars", type=int, default=24)
    p.add_argument("--subtitle-max-lines", type=int, default=2)
    p.add_argument("--subtitle-preset", default="readable")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    transcripts_dir = Path(args.transcripts_dir)
    output_dir = Path(args.output_dir)
    subs_dir = output_dir / "subs"
    ensure_dirs(output_dir, subs_dir)

    segments_map = json.loads(Path(args.segments_json).read_text(encoding="utf-8"))
    index_rows: list[dict[str, object]] = []

    for src_name, parts in segments_map.items():
        src_path = input_dir / src_name
        if not src_path.exists():
            raise RuntimeError(f"Missing source video: {src_path}")
        stem = src_path.stem
        transcript_path = transcripts_dir / f"{stem}.json"
        if not transcript_path.exists():
            raise RuntimeError(f"Missing transcript: {transcript_path}")

        transcript = Transcript.model_validate(read_json(transcript_path))

        for idx, part in enumerate(parts, start=1):
            label = safe_label(str(part.get("label", f"part_{idx}")))
            start = float(part["start"])
            end = float(part["end"])
            if end <= start:
                raise RuntimeError(f"Invalid range for {src_name}#{idx}: {start}-{end}")

            ass_path = subs_dir / f"{stem}_p{idx:02d}_{label}.ass"
            build_ass_subtitles(
                transcript_path,
                start_seconds=start,
                end_seconds=end,
                out_path=ass_path,
                max_chars=args.subtitle_max_chars,
                max_lines=args.subtitle_max_lines,
                style=subtitle_style_from_preset(args.subtitle_preset),
                play_res_x=1080,
                play_res_y=1920,
            )

            out_video = output_dir / f"{stem}_p{idx:02d}_{label}.mp4"
            vf = f"subtitles='{ffmpeg_subtitles_filter_path(ass_path)}'"
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.2f}",
                "-to",
                f"{end:.2f}",
                "-i",
                str(src_path),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-movflags",
                "+faststart",
                str(out_video),
            ]
            run(cmd)

            index_rows.append(
                {
                    "source": src_name,
                    "part": idx,
                    "label": label,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "duration": round(end - start, 2),
                    "output": str(out_video),
                    "subtitle": str(ass_path),
                    "snippet": collect_text_snippet(transcript, start, end),
                }
            )

    report_path = output_dir / "semantic_cuts_report.json"
    report_path.write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)
    print(f"clips: {len(index_rows)}")


if __name__ == "__main__":
    main()
