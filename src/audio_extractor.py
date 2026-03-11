from __future__ import annotations

import os
from pathlib import Path

from .utils import run_ffmpeg


def slice_wav(
    input_wav_path: str | Path,
    output_wav_path: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> Path:
    """
    Losslessly slices PCM WAV using ffmpeg (re-encodes to PCM s16le 16kHz mono for safety).
    """
    in_path = Path(input_wav_path)
    out_path = Path(output_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_ffmpeg(
        [
            "-y",
            "-ss",
            str(start_seconds),
            "-t",
            str(duration_seconds),
            "-i",
            str(in_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out_path),
        ]
    )

    size = out_path.stat().st_size
    if size < 1024:
        raise RuntimeError(f"Срез WAV слишком маленький ({size} байт).")
    return out_path


def extract_wav_mono(
    video_path: str | Path,
    output_wav_path: str | Path,
    *,
    sample_rate: int = 48000,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    accurate_seek: bool = False,
    channel: int | str | None = None,
) -> Path:
    """
    Extracts audio from video into WAV PCM mono.
    """
    in_path = video_path if isinstance(video_path, str) and video_path.startswith("http") else Path(video_path)
    out_path = Path(output_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pre_args: list[str] = ["-y"]
    post_args: list[str] = []
    is_remote = isinstance(video_path, str) and video_path.startswith("http")
    if is_remote:
        # Network streams from Yandex can drop mid-read; let ffmpeg reconnect.
        pre_args += [
            "-rw_timeout",
            "30000000",
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-reconnect_on_network_error",
            "1",
            "-reconnect_at_eof",
            "1",
            "-reconnect_delay_max",
            "2",
        ]
    if accurate_seek:
        if is_remote and start_seconds is not None:
            pad = float(os.getenv("REELS_MAKER_REMOTE_SEEK_PAD", "2.0") or "2.0")
            pre_seek = max(0.0, float(start_seconds) - pad)
            post_seek = float(start_seconds) - pre_seek
            pre_args += ["-ss", str(pre_seek)]
            post_args += ["-ss", str(post_seek)]
        else:
            if start_seconds is not None:
                post_args += ["-ss", str(start_seconds)]
        if duration_seconds is not None:
            post_args += ["-t", str(duration_seconds)]
    else:
        if start_seconds is not None:
            pre_args += ["-ss", str(start_seconds)]
        if duration_seconds is not None:
            pre_args += ["-t", str(duration_seconds)]

    filters: list[str] = []
    if channel is not None:
        if isinstance(channel, str):
            ch = channel.strip().lower()
            if ch in {"l", "left", "fl", "0"}:
                expr = "c0=c0"
            elif ch in {"r", "right", "fr", "1"}:
                expr = "c0=c1"
            else:
                raise RuntimeError(f"Unknown channel spec: {channel}")
        else:
            expr = f"c0=c{int(channel)}"
        filters.append(f"pan=mono|{expr}")

    args = pre_args + [
        "-i",
        str(in_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
    ]
    if filters:
        args += ["-af", ",".join(filters)]
    args += post_args + [str(out_path)]

    run_ffmpeg(args)

    # Sanity-check: a valid WAV should be non-trivial in size.
    try:
        size = out_path.stat().st_size
    except FileNotFoundError as e:
        raise RuntimeError("Не удалось создать WAV файл (ffmpeg не создал output).") from e
    if size < 1024:
        raise RuntimeError(
            f"Извлечённый WAV слишком маленький ({size} байт). "
            "Похоже, в видео нет аудио дорожки или ffmpeg не смог её извлечь."
        )
    return out_path


def extract_wav_16k_mono(
    video_path: str | Path,
    output_wav_path: str | Path,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    accurate_seek: bool = False,
    channel: int | str | None = None,
) -> Path:
    """
    Extracts audio from video into WAV PCM 16kHz mono (Whisper-friendly).
    """
    return extract_wav_mono(
        video_path,
        output_wav_path,
        sample_rate=16000,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        accurate_seek=accurate_seek,
        channel=channel,
    )
