#!/usr/bin/env python3
"""Clean obvious restart fragments and create accelerated highlight versions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


# Local intervals (seconds in clip timeline) to REMOVE.
# These are obvious restart/repeat fragments detected from transcript review.
CUT_INTERVALS: dict[str, list[tuple[float, float]]] = {
    "IMG_5280_p03_weekly_flow.mp4": [(0.0, 1.57)],
    "IMG_5280_p04_contests_writing.mp4": [(10.76, 21.80)],
    "IMG_5282_p02_groups_7_8.mp4": [(0.0, 7.0), (27.0, 37.0), (47.0, 60.33)],
    "IMG_5282_p03_group_9_11.mp4": [(0.0, 6.90)],
}


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if p.returncode != 0:
        tail = (p.stderr or "")[-1400:]
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{tail}")


def media_duration(path: Path) -> float:
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    out = p.stdout.strip()
    if not out:
        raise RuntimeError(f"Cannot read duration: {path}")
    return float(out)


def normalize_cuts(cuts: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    valid: list[tuple[float, float]] = []
    for s, e in cuts:
        s = max(0.0, min(duration, float(s)))
        e = max(0.0, min(duration, float(e)))
        if e > s:
            valid.append((s, e))
    valid.sort()
    merged: list[list[float]] = []
    for s, e in valid:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]


def keep_intervals(duration: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not cuts:
        return [(0.0, duration)]
    out: list[tuple[float, float]] = []
    cur = 0.0
    for s, e in cuts:
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < duration:
        out.append((cur, duration))
    return [(s, e) for s, e in out if e - s > 0.02]


def trim_remove_intervals(src: Path, dst: Path, cuts: list[tuple[float, float]]) -> None:
    duration = media_duration(src)
    cuts_norm = normalize_cuts(cuts, duration)
    keeps = keep_intervals(duration, cuts_norm)
    if not cuts_norm:
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
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
                str(dst),
            ]
        )
        return

    parts_v = []
    parts_a = []
    filt = []
    for i, (s, e) in enumerate(keeps):
        filt.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}]")
        filt.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]")
        parts_v.append(f"[v{i}]")
        parts_a.append(f"[a{i}]")
    n = len(keeps)
    interleaved: list[str] = []
    for i in range(n):
        interleaved.append(f"[v{i}]")
        interleaved.append(f"[a{i}]")
    filt.append("".join(interleaved) + f"concat=n={n}:v=1:a=1[v][a]")
    filter_complex = ";".join(filt)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
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
            str(dst),
        ]
    )


def atempo_chain(speed: float) -> str:
    # ffmpeg atempo supports 0.5..2.0 per stage.
    if speed <= 0:
        raise ValueError("speed must be > 0")
    stages = []
    remain = speed
    while remain > 2.0:
        stages.append("atempo=2.0")
        remain /= 2.0
    while remain < 0.5:
        stages.append("atempo=0.5")
        remain /= 0.5
    stages.append(f"atempo={remain:.6f}")
    return ",".join(stages)


def speedup(src: Path, dst: Path, speed: float) -> None:
    vexpr = f"setpts=PTS/{speed:.6f}"
    aexpr = atempo_chain(speed)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-filter_complex",
            f"[0:v]{vexpr}[v];[0:a]{aexpr}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
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
            str(dst),
        ]
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean highlight clips and create accelerated versions")
    p.add_argument("--input-dir", default="/Users/mac/Documents/reels_maker/highlights/semantic_cuts")
    p.add_argument("--clean-dir", default="/Users/mac/Documents/reels_maker/highlights/semantic_cuts_clean")
    p.add_argument("--speed12-dir", default="/Users/mac/Documents/reels_maker/highlights/semantic_cuts_clean_x12")
    p.add_argument("--speed13-dir", default="/Users/mac/Documents/reels_maker/highlights/semantic_cuts_clean_x13")
    p.add_argument("--make-speed13", action="store_true", help="Also render x1.3 versions")
    p.add_argument("--resume", action="store_true", help="Skip files that are already rendered")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    clean_dir = Path(args.clean_dir)
    speed12_dir = Path(args.speed12_dir)
    speed13_dir = Path(args.speed13_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    speed12_dir.mkdir(parents=True, exist_ok=True)
    if args.make_speed13:
        speed13_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted([p for p in input_dir.glob("*.mp4") if "_p02a_" not in p.name and "_p02b_" not in p.name])
    report: list[dict[str, object]] = []
    for clip in clips:
        clean_out = clean_dir / clip.name
        cuts = CUT_INTERVALS.get(clip.name, [])
        if not (args.resume and clean_out.exists()):
            trim_remove_intervals(clip, clean_out, cuts)

        out12 = speed12_dir / clip.name.replace(".mp4", "_x12.mp4")
        if not (args.resume and out12.exists()):
            speedup(clean_out, out12, 1.2)

        out13 = None
        if args.make_speed13:
            out13 = speed13_dir / clip.name.replace(".mp4", "_x13.mp4")
            if not (args.resume and out13.exists()):
                speedup(clean_out, out13, 1.3)

        report.append(
            {
                "clip": clip.name,
                "removed_intervals": cuts,
                "clean_output": str(clean_out),
                "x12_output": str(out12),
                "x13_output": str(out13) if out13 else None,
                "clean_duration": round(media_duration(clean_out), 2),
                "x12_duration": round(media_duration(out12), 2),
                "x13_duration": round(media_duration(out13), 2) if out13 else None,
            }
        )

    report_path = clean_dir / "cleanup_speed_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)
    print(f"clips: {len(report)}")


if __name__ == "__main__":
    main()
