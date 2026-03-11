from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class TranscriptionConfig(BaseModel):
    backend: str = "faster-whisper"  # faster-whisper|gigaam
    model: str = "large-v3"
    language: str | None = "ru"
    device: str = "cpu"  # cpu|cuda
    compute_type: str = "int8"  # int8|float16|float32
    cpu_threads: int | None = None  # limit CPU threads for faster-whisper on CPU
    # GigaAM-specific options
    gigaam_use_longform: bool = True  # use VAD-based longform when available
    gigaam_chunk_seconds: float = 20.0  # used when longform isn't available



class LlmConfig(BaseModel):
    # OpenAI-compatible provider name. Used for smart defaults in LlmAnalyzer.
    provider: str = "openrouter"  # openrouter|gemini|openai|groq|together|fireworks|ollama|custom
    model: str = "qwen/qwen-2.5-72b-instruct"
    # Optional explicit base URL. If omitted, provider defaults may be used.
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 2000
    min_duration: int = 15
    max_duration: int = 90
    target_duration: int = 45
    # Approximate transcript size per request (character-based).
    max_chars: int = 12000
    # Keep a bit of overlap to avoid cutting off context between chunks.
    chunk_overlap_segments: int = 1
    # Final number of clips kept after de-dup/overlap filtering.
    max_candidates: int = 6


class VideoConfig(BaseModel):
    output_quality: str = "high"
    output_format: str = "mp4"
    output_codec: str = "libx264"
    compress_input: bool = False
    compression_crf: int = 28
    min_shot_duration: float = 2.0
    enable_transitions: bool = False
    transition_duration: float = 0.3
    normalize_audio: bool = True


class CamerasConfig(BaseModel):
    enable_multicam: bool = False
    default_camera: str = "general"
    speaker_to_camera: dict[str, str] = Field(default_factory=dict)


class StorageConfig(BaseModel):
    base_path: str = "./storage"
    auto_cleanup: bool = True
    keep_intermediates: bool = False
    max_disk_usage_gb: int = 15
    warn_disk_usage_gb: int = 10


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "reels_maker.log"
    console_output: bool = True


class PerformanceConfig(BaseModel):
    parallel_processing: bool = False
    max_workers: int = 2
    chunk_size_minutes: int = 30


class AppConfig(BaseModel):
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    cameras: CamerasConfig = Field(default_factory=CamerasConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


@dataclass(frozen=True)
class StoragePaths:
    base: Path
    raw: Path
    audio: Path
    transcripts: Path
    clips: Path
    sync: Path
    output: Path


def load_env() -> None:
    load_dotenv(override=False)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    load_env()

    env_cfg = os.getenv("REELS_MAKER_CONFIG")
    cfg_path = Path(config_path or env_cfg or "./config.yaml")
    if not cfg_path.exists():
        return AppConfig()

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)


def get_storage_paths(cfg: AppConfig) -> StoragePaths:
    base_override = os.getenv("REELS_MAKER_STORAGE")
    base = Path(base_override or cfg.storage.base_path).resolve()
    return StoragePaths(
        base=base,
        raw=base / "raw",
        audio=base / "audio",
        transcripts=base / "transcripts",
        clips=base / "clips",
        sync=base / "sync",
        output=base / "output",
    )


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def check_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "FFmpeg не найден. Установите его (например: brew install ffmpeg)."
        ) from e


def run_ffmpeg(args: list[str]) -> None:
    check_ffmpeg()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *args]
    timeout_s: float | None = None
    timeout_raw = (os.getenv("REELS_MAKER_FFMPEG_TIMEOUT", "") or "").strip()
    if timeout_raw:
        try:
            parsed = float(timeout_raw)
            if parsed > 0:
                timeout_s = parsed
        except ValueError:
            timeout_s = None
    # Default guardrail for hung remote inputs (e.g., expired HTTP URLs).
    if timeout_s is None:
        timeout_s = 900.0

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out after {timeout_s:.0f}s: {' '.join(cmd[:6])} ...") from e
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip() or proc.stdout.strip()}")


def probe_duration_seconds(media_path: str | Path) -> float | None:
    """
    Returns media duration in seconds using ffprobe (part of ffmpeg).
    """
    if isinstance(media_path, str) and media_path.startswith("http"):
        p = media_path
    else:
        p = Path(media_path)
    timeout_raw = (os.getenv("REELS_MAKER_FFPROBE_TIMEOUT", "") or "").strip()
    timeout_s = 20.0
    if timeout_raw:
        try:
            parsed = float(timeout_raw)
            if parsed > 0:
                timeout_s = parsed
        except ValueError:
            timeout_s = 20.0
    env = os.environ.copy()
    if isinstance(media_path, str) and media_path.startswith("http"):
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
            env.pop(key, None)
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(p),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
            env=env,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        return float((proc.stdout or "").strip())
    except Exception:  # noqa: BLE001
        return None


def read_json(path: str | Path) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def seconds_to_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"
