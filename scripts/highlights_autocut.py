#!/usr/bin/env python3
"""Auto-cut vertical highlight videos into dense speech snippets.

Algorithm:
1) Detect silence with ffmpeg silencedetect.
2) Build speech segments as complement of silences.
3) Find the best fixed-size window with max speech density.
4) Render one cut per source video.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SpeechSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class CutSuggestion:
    source: str
    duration: float
    cut_start: float
    cut_end: float
    cut_duration: float
    speech_coverage: float
    speech_density: float


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


def media_duration(path: Path) -> float:
    p = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
    )
    out = p.stdout.strip()
    if not out:
        raise RuntimeError(f"Cannot read duration for: {path}")
    return float(out)


def detect_silences(path: Path, noise_db: float, min_silence: float) -> list[tuple[float, float]]:
    p = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f",
            "null",
            "-",
        ]
    )
    stderr = p.stderr
    starts = [float(v) for v in re.findall(r"silence_start: ([0-9.]+)", stderr)]
    ends = [float(v) for v in re.findall(r"silence_end: ([0-9.]+)", stderr)]
    pairs = []
    for idx in range(min(len(starts), len(ends))):
        s = starts[idx]
        e = ends[idx]
        if e > s:
            pairs.append((s, e))
    if not pairs:
        return []
    pairs.sort()
    merged: list[list[float]] = []
    for s, e in pairs:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]


def build_speech_segments(duration: float, silences: list[tuple[float, float]], min_speech: float) -> list[SpeechSegment]:
    speech: list[SpeechSegment] = []
    cursor = 0.0
    for s, e in silences:
        if s > cursor:
            segment = SpeechSegment(cursor, s)
            if segment.duration >= min_speech:
                speech.append(segment)
        cursor = max(cursor, e)
    if cursor < duration:
        segment = SpeechSegment(cursor, duration)
        if segment.duration >= min_speech:
            speech.append(segment)
    return speech


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def best_window(duration: float, speech: list[SpeechSegment], window: float, step: float) -> tuple[float, float, float, float]:
    if duration <= window:
        coverage = sum(overlap(0.0, duration, s.start, s.end) for s in speech)
        density = coverage / max(duration, 1e-9)
        return 0.0, duration, coverage, density
    best = (0.0, window, 0.0, 0.0)
    t = 0.0
    while t + window <= duration + 1e-9:
        cov = sum(overlap(t, t + window, s.start, s.end) for s in speech)
        dens = cov / window
        if dens > best[3]:
            best = (t, t + window, cov, dens)
        t += step
    return best


def cut_video(src: Path, dst: Path, start: float, length: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.2f}",
        "-t",
        f"{length:.2f}",
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
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Cut failed for {src.name}: {p.stderr[-500:]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-cut highlights by speech density")
    p.add_argument("--input-dir", default="/Users/mac/Documents/reels_maker/highlights")
    p.add_argument("--output-dir", default="/Users/mac/Documents/reels_maker/highlights/cuts_auto")
    p.add_argument("--window-seconds", type=float, default=12.0)
    p.add_argument("--step-seconds", type=float, default=0.25)
    p.add_argument("--noise-db", type=float, default=-35.0)
    p.add_argument("--min-silence-seconds", type=float, default=0.35)
    p.add_argument("--min-speech-seconds", type=float, default=1.2)
    p.add_argument("--glob", default="*.MOV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    videos = sorted(input_dir.glob(args.glob))
    if not videos:
        raise SystemExit(f"No videos found in {input_dir} matching {args.glob}")

    suggestions: list[CutSuggestion] = []
    for v in videos:
        dur = media_duration(v)
        sil = detect_silences(v, args.noise_db, args.min_silence_seconds)
        speech = build_speech_segments(dur, sil, args.min_speech_seconds)
        start, end, cov, dens = best_window(dur, speech, args.window_seconds, args.step_seconds)
        suggestions.append(
            CutSuggestion(
                source=v.name,
                duration=round(dur, 2),
                cut_start=round(start, 2),
                cut_end=round(end, 2),
                cut_duration=round(end - start, 2),
                speech_coverage=round(cov, 2),
                speech_density=round(dens, 3),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for s in suggestions:
        src = input_dir / s.source
        dst = output_dir / f"{Path(s.source).stem}_cut_auto.mp4"
        cut_video(src, dst, s.cut_start, s.cut_duration)

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "params": {
            "window_seconds": args.window_seconds,
            "step_seconds": args.step_seconds,
            "noise_db": args.noise_db,
            "min_silence_seconds": args.min_silence_seconds,
            "min_speech_seconds": args.min_speech_seconds,
        },
        "cuts": [asdict(s) for s in suggestions],
    }
    report_path = output_dir / "autocut_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)
    for s in suggestions:
        print(
            f"{s.source}: {s.cut_start:.2f}-{s.cut_end:.2f} "
            f"(density={s.speech_density:.3f}, speech={s.speech_coverage:.2f}s)"
        )


if __name__ == "__main__":
    main()

