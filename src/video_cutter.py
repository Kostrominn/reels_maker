from __future__ import annotations

import os
from pathlib import Path

from .utils import AppConfig, run_ffmpeg, seconds_to_timestamp


def cut_clip(
    *,
    cfg: AppConfig,
    video_path: str | Path,
    start: float,
    end: float,
    output_path: str | Path,
    accurate_seek: bool = True,
    video_filters: str | None = None,
    copy_video: bool = False,
    include_audio: bool = True,
) -> Path:
    in_path = video_path if isinstance(video_path, str) and video_path.startswith("http") else Path(video_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if end <= start:
        raise ValueError(f"Invalid clip range: start={start}, end={end}")

    duration = end - start

    vcfg = cfg.video
    crf = {"low": 28, "medium": 23, "high": 18}.get(vcfg.output_quality, 18)

    audio_filters: list[str] = []
    if include_audio and getattr(vcfg, "normalize_audio", True):
        audio_filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    af = ",".join(audio_filters) if audio_filters else None

    args: list[str] = ["-y"]

    # Accurate seek: put -ss after -i (slower, but precise). For remote URLs, use a
    # two-stage seek to reduce download size while keeping accuracy.
    if isinstance(in_path, str) and in_path.startswith("http") and accurate_seek:
        pad = float(os.getenv("REELS_MAKER_REMOTE_SEEK_PAD", "2.0") or "2.0")
        pre_seek = max(0.0, float(start) - pad)
        post_seek = float(start) - pre_seek
        args += ["-ss", str(pre_seek), "-i", str(in_path), "-ss", str(post_seek), "-t", str(duration)]
    else:
        args += ["-i", str(in_path)]
        if accurate_seek:
            args += ["-ss", str(start), "-t", str(duration)]
        else:
            # Fast seek: put -ss before -i (may be less precise)
            args = ["-y", "-ss", str(start), "-i", str(in_path), "-t", str(duration)]

    if copy_video:
        args += ["-c:v", "copy", "-an"]
    else:
        args += [
            "-c:v",
            vcfg.output_codec,
            "-crf",
            str(crf),
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
        ]
        if include_audio:
            args += [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        else:
            args += ["-an"]
        if video_filters:
            args += ["-vf", video_filters]
        if include_audio and af:
            args += ["-af", af]

    args += [str(out_path)]

    run_ffmpeg(args)
    return out_path


def format_clip_name(stem: str, start: float, end: float, idx: int, ext: str = "mp4") -> str:
    s1 = seconds_to_timestamp(start).replace(":", "-")
    s2 = seconds_to_timestamp(end).replace(":", "-")
    return f"{stem}_{idx:02d}_{s1}_{s2}.{ext}"
