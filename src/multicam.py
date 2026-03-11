from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio_extractor import extract_wav_16k_mono, extract_wav_mono
from .models import Transcript
from .subtitles import build_ass_subtitles, subtitle_style_from_preset
from .speaker_id import (
    build_reference_embeddings,
    build_reference_embeddings_from_wavs,
    embed_audio,
    load_encoder,
    load_speaker_samples,
)
from .sync import resolve_video_source
from .utils import AppConfig, ensure_dirs, get_storage_paths, probe_duration_seconds, read_json, write_json
from .video_cutter import cut_clip


def _video_stem(video_path: str | Path) -> str:
    s = str(video_path)
    if s.startswith("http"):
        # Try to parse filename from URL
        try:
            from urllib.parse import parse_qs, unquote, urlparse

            parsed = urlparse(s)
            q = parse_qs(parsed.query)
            if "filename" in q and q["filename"]:
                name = unquote(q["filename"][0])
            else:
                name = Path(unquote(parsed.path)).name
            return Path(name).stem or "remote"
        except Exception:
            return "remote"
    if s.startswith("yadisk:"):
        s = s[len("yadisk:") :].strip()
        if s.startswith("disk:"):
            s = s[len("disk:") :].strip()
        return Path(s).name and Path(s).stem or "yadisk"
    return Path(s).stem


def _escape_filter_path(path: str | Path) -> str:
    s = str(path)
    s = s.replace("\\", "\\\\")
    s = s.replace(":", "\\:")
    s = s.replace("'", "\\'")
    return s


def _parse_ass_time(value: str) -> float | None:
    m = re.match(r"^(\d+):(\d{1,2}):(\d{1,2})\.(\d{1,2})$", value.strip())
    if not m:
        return None
    hours = int(m.group(1))
    minutes = int(m.group(2))
    seconds = int(m.group(3))
    centis = int(m.group(4))
    return float(hours * 3600 + minutes * 60 + seconds) + float(centis) / 100.0


def _format_ass_time(value: float) -> str:
    t = max(0.0, float(value))
    total_centis = int(round(t * 100.0))
    cs = total_centis % 100
    total_seconds = total_centis // 100
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _retime_ass_dialogues(*, source_path: Path, out_path: Path, factor: float) -> Path:
    """
    Scale ASS Dialogue timestamps by `factor`.
    Needed when video timeline is retimed via setpts to keep subtitles in sync.
    """
    factor_f = float(factor)
    if factor_f <= 0:
        raise RuntimeError("Invalid ASS retime factor (must be > 0).")
    content = source_path.read_text(encoding="utf-8")
    out_lines: list[str] = []
    for line in content.splitlines():
        if not line.startswith("Dialogue:"):
            out_lines.append(line)
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            out_lines.append(line)
            continue
        start_t = _parse_ass_time(parts[1])
        end_t = _parse_ass_time(parts[2])
        if start_t is None or end_t is None:
            out_lines.append(line)
            continue
        parts[1] = _format_ass_time(start_t * factor_f)
        parts[2] = _format_ass_time(end_t * factor_f)
        out_lines.append(",".join(parts))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    return out_path


def _probe_wav_sr(path: str | Path) -> int | None:
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            return int(wf.getframerate())
    except Exception:
        return None


def _probe_wav_channels(path: str | Path) -> int | None:
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            return int(wf.getnchannels())
    except Exception:
        return None


def _resolve_audio_encoder(audio_encoder: str | None) -> tuple[str, bool]:
    enc = (audio_encoder or "aac").strip().lower()
    if enc == "pcm":
        enc = "pcm_s16le"
    if enc in {"aac", "aac_at"}:
        return enc, True
    if enc in {"alac", "alac_at", "pcm_s16le"}:
        return enc, False
    raise RuntimeError(
        "Неподдерживаемый аудио-кодек. Используй: aac, aac_at, alac, alac_at, pcm."
    )


def _default_multicam_extension(audio_encoder: str) -> str:
    codec, is_lossy = _resolve_audio_encoder(audio_encoder)
    if is_lossy and codec in {"aac", "aac_at"}:
        return ".mp4"
    return ".mov"


def _replace_audio_codec_arg(args: list[str], codec: str) -> list[str]:
    updated = args.copy()
    for i in range(len(updated) - 1):
        if updated[i] == "-c:a":
            updated[i + 1] = codec
            break
    return updated


def _audio_stream_decodes(path: str | Path) -> bool:
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return True
    return proc.returncode == 0


def _build_segment_audio_concat(
    *,
    segment_files: list[Path] | None = None,
    audio_parts: list[tuple[str, float, float]] | None = None,
    out_path: Path,
    audio_sr: int,
    audio_channels: int,
    accurate_seek: bool = True,
) -> Path:
    from .utils import run_ffmpeg

    if segment_files and audio_parts:
        raise RuntimeError("Не удалось собрать аудио: переданы и segment_files, и audio_parts.")
    if not segment_files and not audio_parts:
        raise RuntimeError("Не удалось собрать аудио: нет сегментов.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f"{out_path.stem}_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    wav_inputs: list[Path] = []
    if audio_parts is not None:
        for idx, (src, start, dur) in enumerate(audio_parts, start=1):
            wav_path = tmp_dir / f"seg_{idx:04d}.wav"
            is_remote = isinstance(src, str) and src.startswith("http")
            pre_args: list[str] = ["-y"]
            post_args: list[str] = []
            if accurate_seek:
                if is_remote:
                    pad = float(os.getenv("REELS_MAKER_REMOTE_SEEK_PAD", "2.0") or "2.0")
                    pre_seek = max(0.0, float(start) - pad)
                    post_seek = float(start) - pre_seek
                    pre_args += ["-ss", str(pre_seek)]
                    post_args += ["-ss", str(post_seek)]
                else:
                    post_args += ["-ss", str(float(start))]
                post_args += ["-t", str(float(dur))]
            else:
                pre_args += ["-ss", str(float(start)), "-t", str(float(dur))]

            args = pre_args + [
                "-i",
                str(src),
                "-vn",
                "-ac",
                str(audio_channels),
                "-ar",
                str(audio_sr),
                "-c:a",
                "pcm_s16le",
            ]
            args += post_args + [str(wav_path)]
            run_ffmpeg(args)
            wav_inputs.append(wav_path)
    else:
        assert segment_files is not None
        for idx, seg in enumerate(segment_files, start=1):
            wav_path = tmp_dir / f"seg_{idx:04d}.wav"
            run_ffmpeg(
                [
                    "-y",
                    "-i",
                    str(seg),
                    "-vn",
                    "-ac",
                    str(audio_channels),
                    "-ar",
                    str(audio_sr),
                    "-c:a",
                    "pcm_s16le",
                    str(wav_path),
                ]
            )
            wav_inputs.append(wav_path)

    args: list[str] = ["-y"]
    for wp in wav_inputs:
        args += ["-i", str(wp)]

    input_pads = "".join([f"[{i}:a]" for i in range(len(wav_inputs))])
    filter_complex = f"{input_pads}concat=n={len(wav_inputs)}:v=0:a=1[a]"
    args += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[a]",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    run_ffmpeg(args)
    return out_path


def _default_title_font() -> str | None:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def _normalize_title_typography(text: str) -> str:
    # Normalize odd unicode punctuation from LLM titles to avoid "strange symbols"
    # in overlays on different players/devices.
    table = {
        ord("\u00A0"): " ",  # no-break space
        ord("\u2010"): "-",
        ord("\u2011"): "-",
        ord("\u2012"): "-",
        ord("\u2013"): "-",
        ord("\u2014"): "-",
        ord("\u2015"): "-",
        ord("\u2212"): "-",
    }
    return text.translate(table)


def _wrap_title_text(text: str, *, max_chars: int = 28, max_lines: int = 3) -> str:
    words = [w for w in text.split() if w]
    if not words:
        return ""
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    word_idx = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            lines.append(" ".join(cur))
            if len(lines) >= max_lines:
                break
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
        word_idx += 1

    if len(lines) < max_lines and cur:
        lines.append(" ".join(cur))

    if word_idx < len(words) and lines:
        suffix = "..."
        last = lines[-1]
        if len(last) + len(suffix) <= max_chars:
            lines[-1] = f"{last}{suffix}"
        elif max_chars > len(suffix):
            lines[-1] = f"{last[: max_chars - len(suffix)].rstrip()}{suffix}"

    return "\n".join(lines)


def _estimate_overlay_title_width_px(text: str, fontsize: int) -> float:
    if not text:
        return 0.0
    fs = max(8, int(fontsize))
    narrow = set(" ilI1|.,:;!'`")
    wide = set("WwMm@%#&ЖШЩЮФЫД")
    units = 0.0
    for ch in text:
        if ch in narrow:
            units += 0.34
        elif ch in wide:
            units += 0.9
        else:
            units += 0.6
    return units * float(fs)


def _overlay_title_needs_scroll(text: str, fontsize: int) -> bool:
    """
    Heuristic width check for top overlay title.
    Uses character classes to estimate rendered width and decides
    whether marquee scrolling is needed.
    """
    if not text:
        return False
    estimated_px = _estimate_overlay_title_width_px(text, fontsize)
    # Panel is ~72.6% of frame width. We target 1080px vertical output.
    panel_px = 1080.0 * 0.726
    return estimated_px > panel_px


def _wrap_overlay_title_lines(
    text: str,
    *,
    fontsize: int,
    panel_width_px: float,
    side_padding_px: float,
    max_lines: int = 2,
) -> tuple[list[str], bool]:
    """
    Wrap title text into up to max_lines using approximate rendered width.
    Returns (lines, truncated), where truncated=True means text did not fit
    into max_lines and should use marquee scrolling instead of hard clipping.
    """
    words = [w for w in str(text or "").split() if w]
    if not words:
        return [], False

    max_line_px = max(40.0, float(panel_width_px) - 2.0 * float(side_padding_px))
    all_lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word]) if current else word
        if current and _estimate_overlay_title_width_px(candidate, fontsize) > max_line_px:
            all_lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        all_lines.append(" ".join(current))

    truncated = len(all_lines) > max(1, int(max_lines))
    if truncated:
        return all_lines[: max(1, int(max_lines))], True
    return all_lines, False


def _wrap_overlay_title_all_lines(
    text: str,
    *,
    fontsize: int,
    panel_width_px: float,
    side_padding_px: float,
) -> list[str]:
    words = [w for w in str(text or "").split() if w]
    if not words:
        return []

    max_line_px = max(40.0, float(panel_width_px) - 2.0 * float(side_padding_px))
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word]) if current else word
        if current and _estimate_overlay_title_width_px(candidate, fontsize) > max_line_px:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _overlay_title_pages(
    text: str,
    *,
    fontsize: int,
    panel_width_px: float,
    side_padding_px: float,
    lines_per_page: int = 2,
) -> list[list[str]]:
    lines = _wrap_overlay_title_all_lines(
        text,
        fontsize=fontsize,
        panel_width_px=panel_width_px,
        side_padding_px=side_padding_px,
    )
    if not lines:
        return []
    lpp = max(1, int(lines_per_page))
    return [lines[i : i + lpp] for i in range(0, len(lines), lpp)]


def _polish_title_text(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    # Keep semantics, just remove obvious filler and punctuation noise.
    for pat in ("в целом ", "на самом деле ", "как бы ", "по сути ", "в общем "):
        cleaned = re.sub(rf"(?i)\b{re.escape(pat.strip())}\b\s*", "", cleaned)
    cleaned = cleaned.strip(" \"'«»")
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
    cleaned = re.sub(r"[!?.,;:]{2,}", lambda m: m.group(0)[0], cleaned)
    if len(cleaned) > 140:
        cut = cleaned[:140]
        sp = cut.rfind(" ")
        if sp > 80:
            cleaned = cut[:sp].rstrip()
        else:
            cleaned = cut.rstrip()
    return cleaned


def _effective_overlay_title_seconds(title: str, base_seconds: float) -> float:
    base = max(0.0, float(base_seconds))
    text = " ".join(str(title or "").split())
    if not text:
        return base
    words = len(text.split())
    chars = len(text)
    # No change for short titles.
    if words <= 7 and chars <= 42:
        return base
    # Extend display time for long titles, capped to keep pacing snappy.
    extra_words = max(0.0, (words - 7) * 0.06)
    extra_chars = max(0.0, (chars - 42) * 0.012)
    extra = min(1.2, max(extra_words, extra_chars))
    return round(min(3.2, base + extra), 3)


def _auto_title_from_transcript(
    transcript_path: str | Path,
    *,
    start_seconds: float,
    end_seconds: float,
) -> str | None:
    try:
        data = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    segs = data.get("segments")
    if not isinstance(segs, list):
        return None

    parts: list[str] = []
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        try:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
        except Exception:
            continue
        if e <= start_seconds or s >= end_seconds:
            continue
        txt = " ".join(str(seg.get("text", "")).split()).strip()
        if not txt:
            continue
        parts.append(txt)
        if len(" ".join(parts)) >= 260:
            break

    if not parts:
        return None

    joined = " ".join(parts)
    low = joined.lower()

    if "вмк" in low and "мехмат" in low:
        return "ВМК против мехмата: куда идти?"
    if "вмк" in low and "физфак" in low:
        return "ВМК или физфак: что выбрать?"
    if "вмк" in low and "поступ" in low:
        return "Поступление на ВМК: что важно знать?"
    if "задач" in low:
        return "Как решать задачи и не застревать?"
    if ("учител" in low or "репетитор" in low) and "физ" in low:
        return "Репетитор по физике нужен или нет?"
    if "факульт" in low or "поступ" in low or "вуз" in low:
        return "Как выбрать факультет и не пожалеть?"

    phrase = re.split(r"[.!?]", joined, maxsplit=1)[0]
    phrase = re.sub(r"\s+", " ", phrase).strip(" \t\r\n-,:;")
    phrase = re.sub(r"^(ну|вот|короче|типа|знаешь|слушай)\s+", "", phrase, flags=re.IGNORECASE)
    if not phrase:
        return None
    words = phrase.split()
    if len(words) > 8:
        phrase = " ".join(words[:8])
    phrase = phrase[:64].strip()
    if len(phrase) < 8:
        return None
    if phrase[0].isalpha():
        phrase = phrase[0].upper() + phrase[1:]
    if not phrase.endswith("?"):
        phrase = f"{phrase}?"
    return phrase


def _parse_speaker_map_entries(entries: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise RuntimeError("Используй формат --speaker-map label=camera")
        label, cam = entry.split("=", 1)
        label = label.strip()
        cam = cam.strip()
        if not label or not cam:
            raise RuntimeError("Пустой label/camera в --speaker-map")
        mapping[label] = cam
    return mapping


def _normalize_camera_name(value: str, camera_names: set[str]) -> str | None:
    if value in camera_names:
        return value
    stem = _video_stem(value)
    if stem in camera_names:
        return stem
    return None


def _infer_speaker_map_from_samples(samples: list[Any]) -> dict[str, str]:
    per_label: dict[str, set[str]] = {}
    for s in samples:
        label = getattr(s, "label", None)
        video_path = getattr(s, "video_path", None)
        if not label or not video_path:
            continue
        stem = _video_stem(video_path)
        if not stem:
            continue
        per_label.setdefault(str(label), set()).add(stem)

    inferred: dict[str, str] = {}
    for label, stems in per_label.items():
        if len(stems) == 1:
            inferred[label] = next(iter(stems))
    return inferred


def _resolve_speaker_map(
    cfg: AppConfig,
    *,
    labels: list[str],
    cameras: list["CameraAudio"],
    speaker_map: dict[str, str],
    inferred_map: dict[str, str],
) -> dict[str, str]:
    camera_names = {cam.name for cam in cameras}
    sources: list[dict[str, str]] = []
    if speaker_map:
        sources.append(speaker_map)
    if inferred_map:
        sources.append(inferred_map)
    cfg_map = dict(cfg.cameras.speaker_to_camera or {})
    if cfg_map:
        sources.append(cfg_map)

    resolved: dict[str, str] = {}
    for label in labels:
        cam_value: str | None = None
        for src in sources:
            if label in src:
                cam_value = src[label]
                break
        if cam_value is None and label in camera_names:
            cam_value = label
        if cam_value is None:
            continue
        cam_norm = _normalize_camera_name(str(cam_value), camera_names)
        if cam_norm is None:
            continue
        resolved[label] = cam_norm

    return resolved


def _coalesce_segments(segments: list[dict[str, Any]], *, join_epsilon: float = 0.05) -> list[dict[str, Any]]:
    if not segments:
        return []
    merged: list[dict[str, Any]] = []
    for seg in segments:
        cam = seg.get("camera")
        audio_cam = seg.get("audio_camera") or cam
        if cam is None or audio_cam is None:
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        if end - start <= 0.01:
            continue
        normalized = {
            "camera": str(cam),
            "audio_camera": str(audio_cam),
            "start": start,
            "end": end,
        }
        if not merged:
            merged.append(normalized)
            continue
        prev = merged[-1]
        if (
            prev.get("camera") == normalized["camera"]
            and prev.get("audio_camera") == normalized["audio_camera"]
            and abs(float(prev["end"]) - normalized["start"]) <= join_epsilon
        ):
            prev["end"] = normalized["end"]
        else:
            merged.append(normalized)
    return merged


def _enforce_min_segment_duration(
    segments: list[dict[str, Any]],
    *,
    min_seconds: float,
    adjacency_epsilon: float = 0.06,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Final cleanup pass after all timeline edits.
    Reassigns ultra-short shots to neighboring camera when possible to avoid
    visual jitter introduced by pause cuts / reaction insertion.
    """
    target = max(0.05, float(min_seconds))
    base = _coalesce_segments(segments, join_epsilon=adjacency_epsilon)
    if len(base) <= 1:
        return base, {
            "enabled": True,
            "target_seconds": round(target, 3),
            "changed": False,
            "merged_count": 0,
            "dropped_count": 0,
            "remaining_short_count": 0,
        }

    segs = [dict(seg) for seg in base]
    merged_count = 0
    dropped_count = 0

    changed = True
    while changed and len(segs) > 1:
        changed = False
        i = 0
        while i < len(segs):
            seg = segs[i]
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
            seg_dur = seg_end - seg_start
            if seg_dur >= target:
                i += 1
                continue

            prev = segs[i - 1] if i > 0 else None
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            prev_touches = (
                prev is not None and abs(float(prev["end"]) - seg_start) <= float(adjacency_epsilon)
            )
            next_touches = (
                nxt is not None and abs(float(nxt["start"]) - seg_end) <= float(adjacency_epsilon)
            )

            # Best case: tiny separator between two same-camera segments.
            if (
                prev is not None
                and nxt is not None
                and prev_touches
                and next_touches
                and str(prev.get("camera")) == str(nxt.get("camera"))
                and str(prev.get("audio_camera") or prev.get("camera"))
                == str(nxt.get("audio_camera") or nxt.get("camera"))
            ):
                prev["end"] = float(nxt["end"])
                del segs[i : i + 2]
                merged_count += 1
                changed = True
                i = max(0, i - 1)
                continue

            # Otherwise absorb to the longer touching neighbor.
            absorb_prev = False
            absorb_next = False
            if prev_touches and next_touches and prev is not None and nxt is not None:
                prev_dur = float(prev["end"]) - float(prev["start"])
                next_dur = float(nxt["end"]) - float(nxt["start"])
                absorb_prev = prev_dur >= next_dur
                absorb_next = not absorb_prev
            elif prev_touches:
                absorb_prev = True
            elif next_touches:
                absorb_next = True

            if absorb_prev and prev is not None:
                prev["end"] = seg_end
                del segs[i]
                merged_count += 1
                changed = True
                i = max(0, i - 1)
                continue
            if absorb_next and nxt is not None:
                nxt["start"] = seg_start
                del segs[i]
                merged_count += 1
                changed = True
                continue

            # Isolated micro-piece near boundary: drop it.
            if i == 0 or i == len(segs) - 1 or seg_dur <= 0.35:
                del segs[i]
                dropped_count += 1
                changed = True
                continue

            i += 1

        segs = _coalesce_segments(segs, join_epsilon=adjacency_epsilon)

    if not segs:
        segs = base

    remaining_short = 0
    for seg in segs:
        if float(seg["end"]) - float(seg["start"]) < target:
            remaining_short += 1

    return segs, {
        "enabled": True,
        "target_seconds": round(target, 3),
        "changed": bool(merged_count or dropped_count),
        "merged_count": int(merged_count),
        "dropped_count": int(dropped_count),
        "remaining_short_count": int(remaining_short),
    }


@dataclass(frozen=True)
class PauseAccelProfile:
    min_pause_seconds: float
    keep_ratio: float
    keep_min_seconds: float
    keep_max_seconds: float
    merge_gap_seconds: float
    min_speech_seconds: float


def _get_pause_accel_profile(name: str | None) -> PauseAccelProfile | None:
    key = (name or "none").strip().lower()
    if key in {"", "none", "off", "false", "0"}:
        return None
    if key == "soft":
        return PauseAccelProfile(
            min_pause_seconds=0.45,
            keep_ratio=0.70,
            keep_min_seconds=0.20,
            keep_max_seconds=0.42,
            merge_gap_seconds=0.14,
            min_speech_seconds=0.10,
        )
    if key == "medium":
        return PauseAccelProfile(
            min_pause_seconds=0.38,
            keep_ratio=0.55,
            keep_min_seconds=0.16,
            keep_max_seconds=0.34,
            merge_gap_seconds=0.12,
            min_speech_seconds=0.09,
        )
    if key == "aggressive":
        return PauseAccelProfile(
            min_pause_seconds=0.32,
            keep_ratio=0.42,
            keep_min_seconds=0.12,
            keep_max_seconds=0.28,
            merge_gap_seconds=0.10,
            min_speech_seconds=0.08,
        )
    raise RuntimeError("Неподдерживаемый --pause-accel-profile. Используй: none, soft, medium, aggressive.")


def _merge_intervals(
    intervals: list[tuple[float, float]],
    *,
    merge_gap: float = 0.0,
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals, key=lambda x: x[0]):
        s = float(start)
        e = float(end)
        if e <= s:
            continue
        if not merged:
            merged.append((s, e))
            continue
        prev_s, prev_e = merged[-1]
        if s <= prev_e + float(merge_gap):
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged


def _speech_intervals_from_transcript(
    transcript_path: str | Path,
    *,
    start_seconds: float,
    end_seconds: float,
    min_speech_seconds: float,
    merge_gap_seconds: float,
) -> list[tuple[float, float]]:
    data = read_json(transcript_path)
    transcript = Transcript.model_validate(data)
    intervals: list[tuple[float, float]] = []
    min_len = max(0.01, float(min_speech_seconds))
    for seg in transcript.segments:
        text = " ".join(str(seg.text).split())
        if not text:
            continue
        s = max(float(start_seconds), float(seg.start))
        e = min(float(end_seconds), float(seg.end))
        if e - s < min_len:
            continue
        intervals.append((s, e))
    return _merge_intervals(intervals, merge_gap=float(merge_gap_seconds))


def _build_pause_cut_intervals(
    transcript_path: str | Path,
    *,
    start_seconds: float,
    end_seconds: float,
    profile_name: str,
) -> list[tuple[float, float]]:
    profile = _get_pause_accel_profile(profile_name)
    if profile is None:
        return []

    speech = _speech_intervals_from_transcript(
        transcript_path,
        start_seconds=float(start_seconds),
        end_seconds=float(end_seconds),
        min_speech_seconds=profile.min_speech_seconds,
        merge_gap_seconds=profile.merge_gap_seconds,
    )
    if not speech:
        return []

    pauses: list[tuple[float, float]] = []
    cursor = float(start_seconds)
    for s, e in speech:
        if s > cursor + 1e-6:
            pauses.append((cursor, s))
        cursor = max(cursor, e)
    if float(end_seconds) > cursor + 1e-6:
        pauses.append((cursor, float(end_seconds)))

    cuts: list[tuple[float, float]] = []
    min_cut_seconds = 0.03
    for p_start, p_end in pauses:
        pause_dur = p_end - p_start
        if pause_dur < profile.min_pause_seconds:
            continue
        keep = pause_dur * profile.keep_ratio
        keep = max(profile.keep_min_seconds, min(profile.keep_max_seconds, keep))
        keep = min(keep, pause_dur)
        cut_dur = pause_dur - keep
        if cut_dur < min_cut_seconds:
            continue
        cut_start = p_start + cut_dur * 0.5
        cut_end = p_end - cut_dur * 0.5
        if cut_end - cut_start >= min_cut_seconds:
            cuts.append((cut_start, cut_end))
    return _merge_intervals(cuts, merge_gap=1e-4)


def _apply_cut_intervals_to_segments(
    segments: list[dict[str, Any]],
    *,
    cut_intervals: list[tuple[float, float]],
    min_piece_seconds: float = 0.05,
) -> list[dict[str, Any]]:
    if not cut_intervals:
        return segments
    cuts = _merge_intervals(cut_intervals)
    if not cuts:
        return segments

    kept: list[dict[str, Any]] = []
    min_piece = max(0.01, float(min_piece_seconds))
    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        if seg_end - seg_start < min_piece:
            continue
        pieces: list[tuple[float, float]] = [(seg_start, seg_end)]
        for cut_start, cut_end in cuts:
            if not pieces:
                break
            next_pieces: list[tuple[float, float]] = []
            for p_start, p_end in pieces:
                if cut_end <= p_start or cut_start >= p_end:
                    next_pieces.append((p_start, p_end))
                    continue
                left_end = min(cut_start, p_end)
                right_start = max(cut_end, p_start)
                if left_end - p_start >= min_piece:
                    next_pieces.append((p_start, left_end))
                if p_end - right_start >= min_piece:
                    next_pieces.append((right_start, p_end))
            pieces = next_pieces
        for p_start, p_end in pieces:
            if p_end - p_start < min_piece:
                continue
            cloned = dict(seg)
            cloned["start"] = float(p_start)
            cloned["end"] = float(p_end)
            kept.append(cloned)
    return _coalesce_segments(kept, join_epsilon=0.005)


def _segments_cover_range(
    segments: list[dict[str, Any]],
    *,
    start_seconds: float,
    end_seconds: float,
    tolerance: float = 0.03,
) -> bool:
    if not segments:
        return False
    ordered = sorted(segments, key=lambda x: float(x["start"]))
    if abs(float(ordered[0]["start"]) - float(start_seconds)) > tolerance:
        return False
    if abs(float(ordered[-1]["end"]) - float(end_seconds)) > tolerance:
        return False
    cursor = float(ordered[0]["end"])
    for seg in ordered[1:]:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        if seg_start - cursor > tolerance:
            return False
        cursor = max(cursor, seg_end)
    return abs(cursor - float(end_seconds)) <= tolerance


def _pin_video_to_camera(
    segments: list[dict[str, Any]],
    *,
    camera_name: str,
) -> list[dict[str, Any]]:
    pinned: list[dict[str, Any]] = []
    for seg in segments:
        audio_cam = seg.get("audio_camera") or camera_name
        pinned.append(
            {
                "camera": camera_name,
                "audio_camera": str(audio_cam),
                "start": float(seg["start"]),
                "end": float(seg["end"]),
            }
        )
    return _coalesce_segments(pinned)


def _insert_reaction_segments(
    segments: list[dict[str, Any]],
    cameras: list["CameraAudio"],
    *,
    reaction_every_seconds: float,
    reaction_duration_seconds: float,
    reaction_min_score: float = -0.35,
    reaction_force_cutaway: bool = False,
    reaction_search_window_seconds: float = 1.2,
    reaction_search_step_seconds: float = 0.4,
) -> list[dict[str, Any]]:
    if reaction_every_seconds <= 0 or reaction_duration_seconds <= 0:
        return segments
    if len(cameras) < 2:
        return segments

    new_segments: list[dict[str, Any]] = []
    for seg in segments:
        cam = seg.get("camera")
        audio_cam = seg.get("audio_camera") or cam
        if cam is None or audio_cam is None:
            new_segments.append(seg)
            continue
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        seg_len = seg_end - seg_start
        min_side = 0.8
        min_seg_for_any_cut = reaction_duration_seconds + 2.0 * min_side
        if seg_len < min_seg_for_any_cut:
            new_segments.append(seg)
            continue

        short_mode = seg_len < (reaction_every_seconds + reaction_duration_seconds + 0.5)
        if short_mode:
            center_t = seg_start + (seg_len - reaction_duration_seconds) * 0.5
            cut_targets = [max(seg_start + min_side, min(center_t, seg_end - reaction_duration_seconds - min_side))]
        else:
            cut_targets = []
            t_cursor = seg_start + reaction_every_seconds
            while t_cursor + reaction_duration_seconds < seg_end:
                cut_targets.append(t_cursor)
                t_cursor += reaction_every_seconds

        cur = seg_start
        for t in cut_targets:
            search_w = max(0.0, float(reaction_search_window_seconds))
            search_step = max(0.1, float(reaction_search_step_seconds))
            if search_w <= 0:
                probe_times = [t]
            else:
                probe_times = list(np.arange(t - search_w, t + search_w + 1e-6, search_step))
                if not probe_times:
                    probe_times = [t]

            best_t: float | None = None
            best_other_cam: str | None = None
            best_other_score = float("-inf")
            for t_probe in probe_times:
                t_probe = max(cur + 0.05, min(t_probe, seg_end - reaction_duration_seconds - 0.05))
                t_mid = t_probe + reaction_duration_seconds * 0.5
                other_cam = None
                other_score = float("-inf")
                for cam_obj in cameras:
                    if cam_obj.name == cam:
                        continue
                    score = _camera_score_at(cam_obj, t_mid)
                    if score > other_score:
                        other_score = score
                        other_cam = cam_obj.name
                if other_cam is None:
                    continue
                if other_score > best_other_score:
                    best_other_score = other_score
                    best_other_cam = other_cam
                    best_t = t_probe

            if best_t is None or best_other_cam is None:
                continue
            if best_other_score < reaction_min_score and not reaction_force_cutaway:
                continue

            cut_start = best_t if best_other_score >= reaction_min_score else t
            cut_start = max(cur + 0.05, min(cut_start, seg_end - reaction_duration_seconds - 0.05))
            cut_end = cut_start + reaction_duration_seconds
            if t - cur > 0.05:
                new_segments.append(
                    {
                        "camera": cam,
                        "audio_camera": audio_cam,
                        "start": cur,
                        "end": cut_start,
                    }
                )
            new_segments.append(
                {
                    "camera": best_other_cam,
                    "audio_camera": audio_cam,
                    "start": cut_start,
                    "end": cut_end,
                }
            )
            cur = cut_end

        if seg_end - cur > 0.05:
            new_segments.append(
                {
                    "camera": cam,
                    "audio_camera": audio_cam,
                    "start": cur,
                    "end": seg_end,
                }
            )

    return _coalesce_segments(new_segments)


def _camera_duration_stats(segments: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
    per_camera: dict[str, float] = {}
    total = 0.0
    for seg in segments:
        cam = str(seg.get("camera") or "")
        if not cam:
            continue
        dur = float(seg["end"]) - float(seg["start"])
        if dur <= 0:
            continue
        per_camera[cam] = per_camera.get(cam, 0.0) + dur
        total += dur
    return per_camera, total


def _balance_camera_focus(
    segments: list[dict[str, Any]],
    cameras: list["CameraAudio"],
    *,
    max_dominant_share: float,
    cutaway_seconds: float = 2.4,
    min_gap_seconds: float = 7.0,
    min_side_seconds: float = 0.8,
    search_window_seconds: float = 1.2,
    search_step_seconds: float = 0.4,
    min_cutaway_seconds: float = 1.6,
    max_dominant_advantage_db: float = 2.0,
    min_other_score: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if max_dominant_share >= 0.999:
        return segments, None
    if max_dominant_share <= 0 or max_dominant_share >= 1:
        raise RuntimeError("balance_max_dominant_share должен быть в диапазоне (0, 1).")
    if cutaway_seconds <= 0:
        raise RuntimeError("balance_cutaway_seconds должен быть > 0.")
    if min_gap_seconds < 0:
        raise RuntimeError("balance_min_gap_seconds должен быть >= 0.")
    if max_dominant_advantage_db < 0:
        raise RuntimeError("max_dominant_advantage_db должен быть >= 0.")

    if len(cameras) < 2:
        return segments, None

    base_segments = _coalesce_segments(segments)
    per_cam_before, total_before = _camera_duration_stats(base_segments)
    if total_before <= 0:
        return base_segments, None

    dominant_cam = max(per_cam_before, key=lambda cam: per_cam_before[cam])
    dominant_obj = next((c for c in cameras if c.name == dominant_cam), None)
    if dominant_obj is None:
        return base_segments, None
    dominant_dur = per_cam_before[dominant_cam]
    dominant_share = dominant_dur / total_before
    if dominant_share <= max_dominant_share:
        return base_segments, {
            "enabled": True,
            "applied": False,
            "dominant_camera": dominant_cam,
            "share_before": round(dominant_share, 4),
            "share_after": round(dominant_share, 4),
            "max_dominant_share": float(max_dominant_share),
            "inserted_seconds": 0.0,
        }

    need_reduce = dominant_dur - (max_dominant_share * total_before)
    remaining = float(need_reduce)
    inserted = 0.0
    inserted_count = 0
    min_cut = max(0.6, float(min_cutaway_seconds))
    target_cut = max(min_cut, float(cutaway_seconds))
    side = max(0.0, float(min_side_seconds))
    gap = max(0.0, float(min_gap_seconds))
    search_w = max(0.0, float(search_window_seconds))
    search_step = max(0.1, float(search_step_seconds))

    new_segments: list[dict[str, Any]] = []
    for seg in base_segments:
        cam = str(seg.get("camera") or "")
        audio_cam = str(seg.get("audio_camera") or cam)
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        seg_len = seg_end - seg_start
        if seg_len <= 0.01:
            continue

        if cam != dominant_cam or remaining <= 0:
            new_segments.append(
                {
                    "camera": cam,
                    "audio_camera": audio_cam,
                    "start": seg_start,
                    "end": seg_end,
                }
            )
            continue

        cur = seg_start
        target_t = seg_start + gap
        slot_misses = 0
        while remaining > 0:
            cut_dur = target_cut if remaining >= target_cut else max(min_cut, remaining)
            lo = cur + side
            hi = seg_end - cut_dur - side
            if hi <= lo:
                break
            if target_t < lo:
                target_t = lo
            if target_t > hi:
                target_t = hi

            if search_w <= 0:
                probes = [target_t]
            else:
                probes = list(np.arange(target_t - search_w, target_t + search_w + 1e-6, search_step))
                if not probes:
                    probes = [target_t]

            best_start: float | None = None
            best_other_cam: str | None = None
            best_other_score = float("-inf")
            for probe_t in probes:
                cut_start = max(lo, min(float(probe_t), hi))
                cut_mid = cut_start + cut_dur * 0.5
                dom_score = _camera_score_at(dominant_obj, cut_mid)
                other_cam = None
                other_score = float("-inf")
                for cam_obj in cameras:
                    if cam_obj.name == dominant_cam:
                        continue
                    score = _camera_score_at(cam_obj, cut_mid)
                    # Speaker-aware guardrail:
                    # avoid forcing cutaway when dominant mic is clearly leading.
                    if score < min_other_score:
                        continue
                    if (dom_score - score) > max_dominant_advantage_db:
                        continue
                    if score > other_score:
                        other_score = score
                        other_cam = cam_obj.name
                if other_cam is None:
                    continue
                if other_score > best_other_score:
                    best_other_score = other_score
                    best_other_cam = other_cam
                    best_start = cut_start

            if best_start is None or best_other_cam is None:
                slot_misses += 1
                target_t += max(gap, cut_dur)
                if target_t > hi or slot_misses >= 8:
                    break
                continue

            cut_end = best_start + cut_dur
            if best_start - cur > 0.05:
                new_segments.append(
                    {
                        "camera": dominant_cam,
                        "audio_camera": audio_cam,
                        "start": cur,
                        "end": best_start,
                    }
                )
            new_segments.append(
                {
                    "camera": best_other_cam,
                    "audio_camera": audio_cam,
                    "start": best_start,
                    "end": cut_end,
                }
            )
            inserted += cut_dur
            inserted_count += 1
            remaining -= cut_dur
            cur = cut_end
            target_t = cur + gap
            slot_misses = 0

        if seg_end - cur > 0.05:
            new_segments.append(
                {
                    "camera": dominant_cam,
                    "audio_camera": audio_cam,
                    "start": cur,
                    "end": seg_end,
                }
            )
    balanced = _coalesce_segments(new_segments, join_epsilon=0.005)
    per_cam_after, total_after = _camera_duration_stats(balanced)
    dom_after = dominant_cam
    share_after = 0.0
    if total_after > 0 and dom_after in per_cam_after:
        share_after = per_cam_after[dom_after] / total_after

    info = {
        "enabled": True,
        "applied": inserted_count > 0,
        "dominant_camera": dominant_cam,
        "share_before": round(float(dominant_share), 4),
        "share_after": round(float(share_after), 4),
        "max_dominant_share": float(max_dominant_share),
        "inserted_seconds": round(float(inserted), 3),
        "inserted_count": int(inserted_count),
        "need_reduce_seconds": round(float(need_reduce), 3),
        "remaining_seconds": round(max(0.0, float(remaining)), 3),
        "max_dominant_advantage_db": float(max_dominant_advantage_db),
        "min_other_score": float(min_other_score),
    }
    return balanced, info


@dataclass(frozen=True)
class OffsetModel:
    ref: str | None
    other: str | None
    offset_start: float
    slope_per_sec: float | None = None
    intercept_seconds: float | None = None

    def offset_at(self, t_ref: float | np.ndarray) -> float | np.ndarray:
        if self.slope_per_sec is not None and self.intercept_seconds is not None:
            return self.intercept_seconds + self.slope_per_sec * t_ref
        return self.offset_start

    def offset_range(self, t0: float, t1: float) -> tuple[float, float]:
        o0 = float(self.offset_at(t0))
        o1 = float(self.offset_at(t1))
        return (min(o0, o1), max(o0, o1))


def load_offset_model(path: str | Path) -> OffsetModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    offset_start = float(data.get("offset_start", 0.0))
    fit = data.get("fit_drift") or {}
    slope = fit.get("slope_per_sec")
    intercept = fit.get("intercept_seconds")
    slope_f = float(slope) if slope is not None else None
    intercept_f = float(intercept) if intercept is not None else None
    return OffsetModel(
        ref=data.get("ref"),
        other=data.get("other"),
        offset_start=offset_start,
        slope_per_sec=slope_f,
        intercept_seconds=intercept_f,
    )


def _load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    p = Path(path)
    with wave.open(str(p), "rb") as wf:
        if wf.getnchannels() != 1:
            raise RuntimeError("WAV должен быть mono.")
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    audio /= 32768.0
    return audio, sr


def _energy_envelope(
    sig: np.ndarray, sr: int, *, frame_ms: float = 200.0, hop_ms: float = 100.0
) -> tuple[np.ndarray, int]:
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    n_frames = max(1, 1 + (len(sig) - frame) // hop) if len(sig) >= frame else 1
    env = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        end = start + frame
        chunk = sig[start:end]
        if len(chunk) == 0:
            env[i] = 0.0
        else:
            env[i] = float(np.sqrt(np.mean(chunk * chunk)))
    env_sr = int(round(sr / hop))
    return env, env_sr


def _env_to_db(env: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(env, 1e-8))


def _interp_env_db(env_db: np.ndarray, env_sr: int, t_local: float) -> float:
    idx = t_local * env_sr
    if idx < 0 or idx >= len(env_db) - 1:
        return float("-inf")
    i0 = int(math.floor(idx))
    frac = float(idx - i0)
    return float(env_db[i0] * (1.0 - frac) + env_db[i0 + 1] * frac)


def _sample_audio(audio: np.ndarray, sr: int, t_local: np.ndarray) -> np.ndarray:
    idx = t_local * float(sr)
    i0 = np.floor(idx).astype(np.int64)
    i1 = i0 + 1
    frac = idx - i0
    out = np.zeros_like(t_local, dtype=np.float32)
    mask = (i0 >= 0) & (i1 < audio.shape[0])
    if np.any(mask):
        i0m = i0[mask]
        i1m = i1[mask]
        fracm = frac[mask].astype(np.float32)
        out[mask] = audio[i0m] * (1.0 - fracm) + audio[i1m] * fracm
    return out


@dataclass
class CameraAudio:
    name: str
    source_path: str
    video_path: str
    offset_model: OffsetModel | None
    video_filter: str | None
    audio_path: Path
    audio: np.ndarray
    sr: int
    audio_start: float
    env_db: np.ndarray
    env_sr: int
    threshold_db: float
    level_shift_db: float = 0.0
    bias_db: float = 0.0


def _prepare_cameras(
    cfg: AppConfig,
    *,
    ref_video: str,
    other_videos: list[str],
    offsets: list[OffsetModel],
    ref_channel: str | int | None,
    other_channels: list[str | int | None],
    ref_bias_db: float,
    other_bias_db: list[float],
    ref_vf: str | None,
    other_vf: list[str],
    start_seconds: float,
    end_seconds: float,
    frame_ms: float,
    hop_ms: float,
    speech_margin_db: float,
    fast_seek_only: bool,
    level_normalize: bool,
    level_max_db: float,
    level_target: str,
    audio_sample_rate: int = 48000,
    pad_seconds: float = 0.5,
) -> list[CameraAudio]:
    storage = get_storage_paths(cfg)
    audio_dir = storage.audio / "multicam"
    ensure_dirs(audio_dir)

    cameras: list[tuple[str, str, OffsetModel | None, str | int | None, float, str | None]] = [
        ("ref", ref_video, None, ref_channel, float(ref_bias_db), ref_vf)
    ]
    if other_channels and len(other_channels) != len(other_videos):
        raise RuntimeError("Количество --other-channel должно совпадать с количеством --other.")
    if not other_channels:
        other_channels = [None] * len(other_videos)
    if other_bias_db and len(other_bias_db) != len(other_videos):
        raise RuntimeError("Количество --other-bias-db должно совпадать с количеством --other.")
    if not other_bias_db:
        other_bias_db = [0.0] * len(other_videos)
    if other_vf and len(other_vf) != len(other_videos):
        raise RuntimeError("Количество --other-vf должно совпадать с количеством --other.")
    if not other_vf:
        other_vf = [None] * len(other_videos)
    for vid, off, ch, bias, vf in zip(
        other_videos, offsets, other_channels, other_bias_db, other_vf, strict=True
    ):
        cameras.append(("other", vid, off, ch, float(bias), vf))

    prepared: list[CameraAudio] = []
    for _role, src_path, offset_model, channel, bias_db, vf in cameras:
        resolved = resolve_video_source(str(src_path))
        name = _video_stem(src_path)

        duration = probe_duration_seconds(resolved)
        if offset_model is None:
            off_min, off_max = 0.0, 0.0
        else:
            off_min, off_max = offset_model.offset_range(start_seconds, end_seconds)

        t_cam_min = start_seconds + off_min
        t_cam_max = end_seconds + off_max
        cam_start = max(0.0, t_cam_min - pad_seconds)
        cam_end = t_cam_max + pad_seconds
        if duration is not None:
            if cam_start >= float(duration) - 1e-3:
                raise RuntimeError(
                    f"Камера {name}: запрошенный диапазон вне длительности источника "
                    f"(cam_start={cam_start:.2f}s, duration={float(duration):.2f}s). "
                    "Проверьте оффсет и границы клипа."
                )
            cam_end = min(cam_end, duration)
        cam_dur = cam_end - cam_start if cam_end > cam_start else None

        if audio_sample_rate <= 0:
            raise RuntimeError("audio_sample_rate должен быть > 0.")
        suffix = f"_s{start_seconds:g}_e{end_seconds:g}"
        audio_path = audio_dir / f"{name}{suffix}_sr{int(audio_sample_rate)}.wav"
        reuse = False
        try:
            if audio_path.exists() and audio_path.stat().st_size > 1024:
                reuse = True
        except OSError:
            reuse = False

        if not reuse:
            extract_wav_mono(
                resolved,
                audio_path,
                sample_rate=int(audio_sample_rate),
                start_seconds=cam_start,
                duration_seconds=cam_dur,
                accurate_seek=not fast_seek_only,
                channel=channel,
            )
        audio, sr = _load_wav_mono(audio_path)
        env, env_sr = _energy_envelope(audio, sr, frame_ms=frame_ms, hop_ms=hop_ms)
        env_db = _env_to_db(env)
        noise_floor = float(np.percentile(env_db, 20))
        threshold = noise_floor + speech_margin_db

        prepared.append(
            CameraAudio(
                name=name,
                source_path=str(src_path),
                video_path=str(resolved),
                offset_model=offset_model,
                video_filter=vf,
                audio_path=audio_path,
                audio=audio,
                sr=sr,
                audio_start=cam_start,
                env_db=env_db,
                env_sr=env_sr,
                threshold_db=threshold,
                level_shift_db=0.0,
                bias_db=float(bias_db),
            )
        )

    if level_normalize and prepared:
        speech_medians: list[float] = []
        for cam in prepared:
            speech_mask = cam.env_db >= cam.threshold_db
            if np.any(speech_mask):
                speech_medians.append(float(np.median(cam.env_db[speech_mask])))
            else:
                speech_medians.append(float(np.percentile(cam.env_db, 95)))
        if level_target == "max":
            target = float(max(speech_medians))
        elif level_target == "min":
            target = float(min(speech_medians))
        else:
            target = float(np.median(speech_medians))
        max_db = float(level_max_db)
        for cam, med in zip(prepared, speech_medians, strict=True):
            shift = target - float(med)
            if max_db > 0:
                shift = max(-max_db, min(max_db, shift))
            cam.level_shift_db = shift

    return prepared


def _camera_db_at(camera: CameraAudio, t_ref: float) -> float:
    if camera.offset_model is None:
        t_cam = t_ref
    else:
        t_cam = t_ref + float(camera.offset_model.offset_at(t_ref))
    t_local = t_cam - camera.audio_start
    return _interp_env_db(camera.env_db, camera.env_sr, t_local)


def _camera_score_at(camera: CameraAudio, t_ref: float) -> float:
    """
    Normalize per-mic loudness by its own noise floor threshold.
    Positive score => likely speech on that mic.
    """
    return (_camera_db_at(camera, t_ref) + camera.level_shift_db + camera.bias_db) - camera.threshold_db


def _window_samples(camera: CameraAudio, t_ref: float, frame_samples: int) -> np.ndarray:
    if camera.offset_model is None:
        t_cam = t_ref
    else:
        t_cam = t_ref + float(camera.offset_model.offset_at(t_ref))
    t_local = t_cam - camera.audio_start
    start = int(round(t_local * camera.sr))
    end = start + frame_samples
    if start < 0 or end <= 0 or start >= camera.audio.shape[0]:
        return np.zeros(frame_samples, dtype=np.float32)
    win = camera.audio[max(0, start) : min(end, camera.audio.shape[0])]
    if win.shape[0] < frame_samples:
        pad = np.zeros(frame_samples - win.shape[0], dtype=np.float32)
        win = np.concatenate([win, pad], axis=0)
    return win.astype(np.float32)


def _residual_scores(cameras: list[CameraAudio], t_ref: float, frame_samples: int) -> list[float]:
    if len(cameras) != 2:
        return [_camera_score_at(cam, t_ref) for cam in cameras]
    a = _window_samples(cameras[0], t_ref, frame_samples)
    b = _window_samples(cameras[1], t_ref, frame_samples)
    eps = 1e-8
    denom_b = float(np.dot(b, b)) + eps
    denom_a = float(np.dot(a, a)) + eps
    alpha = float(np.dot(a, b)) / denom_b
    beta = float(np.dot(b, a)) / denom_a
    ra = a - alpha * b
    rb = b - beta * a
    rms_a = float(np.sqrt(np.mean(ra * ra) + eps))
    rms_b = float(np.sqrt(np.mean(rb * rb) + eps))
    db_a = 20.0 * math.log10(max(rms_a, 1e-8)) + cameras[0].level_shift_db + cameras[0].bias_db
    db_b = 20.0 * math.log10(max(rms_b, 1e-8)) + cameras[1].level_shift_db + cameras[1].bias_db
    return [db_a, db_b]


def build_shotlist(
    cfg: AppConfig,
    *,
    ref_video: str,
    other_videos: list[str],
    offsets: list[OffsetModel],
    ref_channel: str | int | None,
    other_channels: list[str | int | None],
    ref_bias_db: float = 0.0,
    other_bias_db: list[float] | None = None,
    ref_vf: str | None = None,
    other_vf: list[str] | None = None,
    start_seconds: float,
    end_seconds: float,
    frame_ms: float = 200.0,
    hop_seconds: float = 0.1,
    speech_margin_db: float = 8.0,
    switch_margin_db: float = 4.0,
    min_shot_seconds: float | None = None,
    fast_seek_only: bool = False,
    level_normalize: bool = False,
    level_max_db: float = 6.0,
    level_target: str = "median",
    dominance_mode: str = "energy",
    speaker_refs_path: str | None = None,
    speaker_ref_dir: str | None = None,
    speaker_map: dict[str, str] | None = None,
    speaker_window_seconds: float = 1.5,
    speaker_lead_seconds: float = 0.0,
    speaker_min_similarity: float = 0.45,
    speaker_margin: float = 0.1,
    speaker_device: str = "cpu",
    reaction_every_seconds: float = 0.0,
    reaction_duration_seconds: float = 2.4,
    audio_sample_rate: int = 48000,
) -> tuple[list[dict[str, Any]], list[CameraAudio], dict[str, Any], dict[str, Any] | None]:
    hop_ms = hop_seconds * 1000.0
    cameras = _prepare_cameras(
        cfg,
        ref_video=ref_video,
        other_videos=other_videos,
        offsets=offsets,
        ref_channel=ref_channel,
        other_channels=other_channels,
        ref_bias_db=ref_bias_db,
        other_bias_db=other_bias_db or [],
        ref_vf=ref_vf,
        other_vf=other_vf or [],
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        speech_margin_db=speech_margin_db,
        fast_seek_only=fast_seek_only,
        level_normalize=level_normalize,
        level_max_db=level_max_db,
        level_target=level_target,
        audio_sample_rate=audio_sample_rate,
    )

    if not cameras:
        raise RuntimeError("Не удалось подготовить камеры.")

    if min_shot_seconds is None:
        min_shot_seconds = float(cfg.video.min_shot_duration)

    speaker_timeline: dict[str, Any] | None = None
    speaker_steps: list[dict[str, Any]] = []
    speaker_refs: dict[str, np.ndarray] = {}
    speaker_cam_map: dict[str, str] = {}
    speaker_encoder = None
    speaker_window_samples = 0

    if dominance_mode == "speaker":
        if speaker_window_seconds <= 0:
            raise RuntimeError("speaker_window_seconds должен быть > 0.")
        if speaker_lead_seconds < 0:
            raise RuntimeError("speaker_lead_seconds должен быть >= 0.")

        speaker_encoder = load_encoder(device=speaker_device)
        samples: list[Any] | None = None
        if speaker_refs_path:
            samples = load_speaker_samples(speaker_refs_path)
            speaker_refs = build_reference_embeddings(
                cfg,
                samples,
                device=speaker_device,
                encoder=speaker_encoder,
            )
        else:
            ref_dir = (
                Path(speaker_ref_dir)
                if speaker_ref_dir
                else get_storage_paths(cfg).audio / "speaker_refs"
            )
            speaker_refs = build_reference_embeddings_from_wavs(
                ref_dir,
                device=speaker_device,
                encoder=speaker_encoder,
            )

        labels = sorted(speaker_refs.keys())
        inferred = _infer_speaker_map_from_samples(samples or [])
        speaker_cam_map = _resolve_speaker_map(
            cfg,
            labels=labels,
            cameras=cameras,
            speaker_map=speaker_map or {},
            inferred_map=inferred,
        )
        missing = [lbl for lbl in labels if lbl not in speaker_cam_map]
        if missing:
            raise RuntimeError(f"Нет маппинга speaker->camera для: {', '.join(missing)}")

        speaker_window_samples = max(1, int(round(cameras[0].sr * speaker_window_seconds)))
        cam_name_to_idx = {cam.name: idx for idx, cam in enumerate(cameras)}
        cam_idx_to_label = {
            cam_name_to_idx[cam_name]: label
            for label, cam_name in speaker_cam_map.items()
            if cam_name in cam_name_to_idx
        }
        mapped_cam_indices = sorted(cam_idx_to_label.keys())

    frame_samples = max(1, int(round(cameras[0].sr * (frame_ms / 1000.0))))
    n_steps = max(1, int(math.ceil((end_seconds - start_seconds) / hop_seconds)))
    current = 0
    last_switch = start_seconds
    per_step: list[int] = []

    for i in range(n_steps):
        t_ref = start_seconds + i * hop_seconds
        speech_scores = [_camera_score_at(cam, t_ref) for cam in cameras]
        if dominance_mode == "speaker":
            mapped_scores = [speech_scores[idx] for idx in mapped_cam_indices]
            if not mapped_scores or max(mapped_scores) < 0:
                per_step.append(current)
                speaker_steps.append(
                    {
                        "time": t_ref,
                        "best_label": None,
                        "best_sim": None,
                        "second_sim": None,
                        "chosen_camera": cameras[current].name,
                        "chosen_label": cam_idx_to_label.get(current),
                        "reason": "silence",
                    }
                )
                continue

            best_cam_idx: int | None = None
            best_label: str | None = None
            best_sim = -1.0
            second_sim = -1.0

            for cam_idx, cam in enumerate(cameras):
                label = cam_idx_to_label.get(cam_idx)
                if label is None:
                    continue
                if speech_scores[cam_idx] < 0:
                    continue
                t_center = t_ref + speaker_lead_seconds - speaker_window_seconds * 0.5
                win = _window_samples(cam, t_center, speaker_window_samples)
                emb = embed_audio(speaker_encoder, win, cam.sr)
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                ref = speaker_refs.get(label)
                if ref is None:
                    continue
                sim = float(np.dot(emb, ref))
                if sim > best_sim:
                    second_sim = best_sim
                    best_sim = sim
                    best_cam_idx = cam_idx
                    best_label = label
                elif sim > second_sim:
                    second_sim = sim

            reason = "hold"
            if best_cam_idx is None:
                reason = "no_candidates"
                per_step.append(current)
            elif best_sim < speaker_min_similarity:
                reason = "low_similarity"
                per_step.append(current)
            elif (best_sim - second_sim) < speaker_margin:
                reason = "margin"
                per_step.append(current)
            elif best_cam_idx == current:
                reason = "stay"
                per_step.append(current)
            elif (t_ref - last_switch) < min_shot_seconds:
                reason = "min_shot"
                per_step.append(current)
            else:
                current = best_cam_idx
                last_switch = t_ref
                reason = "switch"
                per_step.append(current)

            best_sim_out = None if best_cam_idx is None else round(best_sim, 4)
            second_sim_out = None if best_cam_idx is None else round(second_sim, 4)
            speaker_steps.append(
                {
                    "time": t_ref,
                    "best_label": best_label,
                    "best_sim": best_sim_out,
                    "second_sim": second_sim_out,
                    "chosen_camera": cameras[current].name,
                    "chosen_label": cam_idx_to_label.get(current),
                    "reason": reason,
                }
            )
            continue

        if dominance_mode == "residual":
            scores = _residual_scores(cameras, t_ref, frame_samples)
        else:
            scores = speech_scores
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]

        if max(speech_scores) < 0:
            per_step.append(current)
            continue

        if best_idx == current:
            per_step.append(current)
            continue

        curr_score = scores[current]
        if (t_ref - last_switch) >= min_shot_seconds and (best_score - curr_score) >= switch_margin_db:
            current = best_idx
            last_switch = t_ref
        per_step.append(current)

    # Build segments
    segments: list[dict[str, Any]] = []
    seg_start = start_seconds
    seg_cam = per_step[0]
    for i in range(1, n_steps):
        if per_step[i] != seg_cam:
            seg_end = min(end_seconds, start_seconds + i * hop_seconds)
            segments.append(
                {
                    "camera": cameras[seg_cam].name,
                    "audio_camera": cameras[seg_cam].name,
                    "start": seg_start,
                    "end": seg_end,
                }
            )
            seg_start = seg_end
            seg_cam = per_step[i]
    segments.append(
        {
            "camera": cameras[seg_cam].name,
            "audio_camera": cameras[seg_cam].name,
            "start": seg_start,
            "end": end_seconds,
        }
    )

    if dominance_mode == "speaker":
        speaker_timeline = {
            "schema_version": "1.0",
            "meta": {
                "labels": sorted(speaker_refs.keys()),
                "speaker_map": speaker_cam_map,
                "speaker_window_seconds": speaker_window_seconds,
                "speaker_lead_seconds": speaker_lead_seconds,
                "speaker_min_similarity": speaker_min_similarity,
                "speaker_margin": speaker_margin,
                "speaker_device": speaker_device,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "hop_seconds": hop_seconds,
            },
            "steps": speaker_steps,
        }

    meta = {
        "ref": ref_video,
        "cameras": [
            {
                "name": cam.name,
                "source": cam.source_path,
                "threshold_db": cam.threshold_db,
                "level_shift_db": cam.level_shift_db,
                "bias_db": cam.bias_db,
            }
            for cam in cameras
        ],
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "hop_seconds": hop_seconds,
        "frame_ms": frame_ms,
        "speech_margin_db": speech_margin_db,
        "switch_margin_db": switch_margin_db,
        "min_shot_seconds": min_shot_seconds,
        "level_normalize": level_normalize,
        "level_max_db": level_max_db,
        "level_target": level_target,
        "audio_sample_rate": audio_sample_rate,
        "dominance_mode": dominance_mode,
        "speaker_mode": {
            "speaker_window_seconds": speaker_window_seconds,
            "speaker_lead_seconds": speaker_lead_seconds,
            "speaker_min_similarity": speaker_min_similarity,
            "speaker_margin": speaker_margin,
            "speaker_device": speaker_device,
            "speaker_map": speaker_cam_map,
        }
        if dominance_mode == "speaker"
        else None,
        "reaction_mode": None,
    }

    return segments, cameras, meta, speaker_timeline


def make_gated_mix(
    *,
    cameras: list[CameraAudio],
    segments: list[dict[str, Any]],
    start_seconds: float,
    end_seconds: float,
    duck_db: float = 18.0,
    crossfade_seconds: float = 0.08,
    hard_gate: bool = False,
    out_path: str | Path,
    chunk_seconds: float = 10.0,
) -> Path:
    if not cameras:
        raise RuntimeError("Нет камер для микса.")
    sr = cameras[0].sr
    for cam in cameras[1:]:
        if cam.sr != sr:
            raise RuntimeError("Частоты дискретизации камер не совпадают.")

    duck_gain = float(10 ** (-duck_db / 20.0))
    inactive_gain = 0.0 if hard_gate else duck_gain
    total_samples = int(round((end_seconds - start_seconds) * sr))
    if total_samples <= 0:
        raise RuntimeError("Некорректный диапазон для микса.")

    # Precompute segment sample ranges
    seg_ranges: list[tuple[int, int, int]] = []
    name_to_idx = {cam.name: idx for idx, cam in enumerate(cameras)}
    for seg in segments:
        s = int(round((float(seg["start"]) - start_seconds) * sr))
        e = int(round((float(seg["end"]) - start_seconds) * sr))
        cam_name = seg.get("audio_camera") or seg.get("camera")
        cam_idx = name_to_idx.get(cam_name, 0)
        seg_ranges.append((max(0, s), max(0, e), cam_idx))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_size = int(round(chunk_seconds * sr))
    if chunk_size <= 0:
        chunk_size = sr

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)

        seg_ptr = 0
        last_active = 0
        for chunk_start in range(0, total_samples, chunk_size):
            chunk_n = min(chunk_size, total_samples - chunk_start)
            t_ref = start_seconds + (chunk_start + np.arange(chunk_n)) / float(sr)

            active = np.full(chunk_n, last_active, dtype=np.int32)
            # Fill active camera indices for this chunk
            while seg_ptr < len(seg_ranges) and seg_ranges[seg_ptr][1] <= chunk_start:
                seg_ptr += 1
            local_ptr = seg_ptr
            while local_ptr < len(seg_ranges):
                s, e, cam_idx = seg_ranges[local_ptr]
                if s >= chunk_start + chunk_n:
                    break
                lo = max(0, s - chunk_start)
                hi = min(chunk_n, e - chunk_start)
                if hi > lo:
                    active[lo:hi] = cam_idx
                local_ptr += 1
            if chunk_n > 0:
                last_active = int(active[-1])

            mix = np.zeros(chunk_n, dtype=np.float32)
            fade_samples = int(round(crossfade_seconds * sr))
            if fade_samples < 1:
                fade_samples = 0

            gains: list[np.ndarray] | None = None
            if fade_samples > 0:
                gains = [np.full(chunk_n, inactive_gain, dtype=np.float32) for _ in cameras]
                for cam_idx in range(len(cameras)):
                    gains[cam_idx][active == cam_idx] = 1.0
                change_idx = np.where(active[1:] != active[:-1])[0] + 1
                for idx in change_idx:
                    prev_cam = int(active[idx - 1])
                    next_cam = int(active[idx])
                    if prev_cam == next_cam:
                        continue
                    fade = min(fade_samples, chunk_n - idx)
                    if fade <= 1:
                        continue
                    ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
                    gains[prev_cam][idx : idx + fade] = (1.0 - ramp) * 1.0 + ramp * inactive_gain
                    gains[next_cam][idx : idx + fade] = (1.0 - ramp) * inactive_gain + ramp * 1.0

            for cam_idx, cam in enumerate(cameras):
                if cam.offset_model is None:
                    offsets = 0.0
                else:
                    offsets = cam.offset_model.offset_at(t_ref)
                t_cam = t_ref + offsets
                t_local = t_cam - cam.audio_start
                samples = _sample_audio(cam.audio, sr, t_local)
                if cam.level_shift_db:
                    samples = samples * float(10 ** (cam.level_shift_db / 20.0))

                if gains is None:
                    gain = np.full(chunk_n, inactive_gain, dtype=np.float32)
                    gain[active == cam_idx] = 1.0
                else:
                    gain = gains[cam_idx]
                mix += samples * gain

            mix = np.clip(mix, -1.0, 1.0)
            pcm = (mix * 32767.0).astype(np.int16)
            wf.writeframes(pcm.tobytes())

    return out_path


def render_multicam(
    cfg: AppConfig,
    *,
    cameras: list[CameraAudio],
    segments: list[dict[str, Any]],
    start_seconds: float,
    end_seconds: float,
    audio_mix_path: Path,
    audio_from_sources: bool = False,
    segment_audio_sr: int | None = None,
    segment_audio_channels: int = 2,
    audio_encoder: str = "aac",
    output_path: str | Path,
    fast_seek_only: bool,
    subtitle_path: Path | None = None,
    title_text: str | None = None,
    title_duration: float = 1.6,
    title_font: str | None = None,
    title_fontsize: int = 64,
    title_style: str = "card",
    cache_segments: bool = False,
    cache_padding_seconds: float = 2.0,
    cache_vf: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    storage = get_storage_paths(cfg)
    seg_root = storage.output / "multicam_segments"
    run_key = f"{output_path}|{start_seconds:.3f}|{end_seconds:.3f}|{os.getpid()}|{time.time_ns()}"
    run_hash = hashlib.sha1(run_key.encode("utf-8")).hexdigest()[:10]
    seg_dir = seg_root / f"{Path(output_path).stem}_{run_hash}"
    ensure_dirs(seg_dir)

    output_path = Path(output_path)

    cache_root = cache_dir or (storage.output / "multicam_cache")
    if cache_segments:
        ensure_dirs(cache_root)

    name_to_cam = {cam.name: cam for cam in cameras}
    segment_files: list[Path] = []
    audio_parts: list[tuple[str, float, float]] = []

    cache_map: dict[str, tuple[Path, float]] = {}
    if cache_segments:
        ranges: dict[str, tuple[float, float]] = {}
        for seg in segments:
            cam = name_to_cam.get(seg["camera"])
            if cam is None:
                continue
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
            seg_dur = max(0.0, seg_end - seg_start)
            if seg_dur < 0.05:
                continue
            if cam.offset_model is None:
                start_cam = seg_start
            else:
                start_cam = seg_start + float(cam.offset_model.offset_at(seg_start))
            if start_cam < 0:
                start_cam = 0.0
            end_cam = start_cam + seg_dur
            if cam.name in ranges:
                lo, hi = ranges[cam.name]
                ranges[cam.name] = (min(lo, start_cam), max(hi, end_cam))
            else:
                ranges[cam.name] = (start_cam, end_cam)

        for cam_name, (lo, hi) in ranges.items():
            cam = name_to_cam[cam_name]
            pad = max(0.0, float(cache_padding_seconds))
            cache_start = max(0.0, lo - pad)
            cache_end = hi + pad
            cache_dur = max(0.01, cache_end - cache_start)
            effective_vf = cache_vf if cache_vf is not None else cam.video_filter
            if effective_vf:
                vf_hash = hashlib.sha1(effective_vf.encode("utf-8")).hexdigest()[:8]
            else:
                vf_hash = "nofx"
            cache_name = f"{cam.name}_cache_{vf_hash}_s{cache_start:.2f}_d{cache_dur:.2f}.mp4"
            cache_path = cache_root / cache_name
            if not cache_path.exists() or cache_path.stat().st_size < 1024:
                cut_clip(
                    cfg=cfg,
                    video_path=cam.video_path,
                    start=cache_start,
                    end=cache_start + cache_dur,
                    output_path=cache_path,
                    accurate_seek=not fast_seek_only,
                    video_filters=cache_vf or cam.video_filter,
                    include_audio=False,
                )
            cache_map[cam.name] = (cache_path, cache_start)

    for idx, seg in enumerate(segments, start=1):
        cam = name_to_cam.get(seg["camera"])
        if cam is None:
            continue
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        seg_dur = max(0.0, seg_end - seg_start)
        if seg_dur < 0.05:
            continue

        if cam.offset_model is None:
            start_cam = seg_start
        else:
            start_cam = seg_start + float(cam.offset_model.offset_at(seg_start))
        if start_cam < 0:
            start_cam = 0.0

        out_seg = seg_dir / f"{cam.name}_seg_{idx:04d}.mp4"

        if cache_segments:
            cache_path, cache_start = cache_map[cam.name]
            local_start = max(0.0, start_cam - cache_start)
            cut_clip(
                cfg=cfg,
                video_path=cache_path,
                start=local_start,
                end=local_start + seg_dur,
                output_path=out_seg,
                accurate_seek=True,
                video_filters=None,
                include_audio=False,
            )
        else:
            cut_clip(
                cfg=cfg,
                video_path=cam.video_path,
                start=start_cam,
                end=start_cam + seg_dur,
                output_path=out_seg,
                accurate_seek=not fast_seek_only,
                video_filters=cam.video_filter,
                include_audio=False,
            )
        segment_files.append(out_seg)
        audio_cam_name = seg.get("audio_camera") or seg.get("camera")
        audio_cam = name_to_cam.get(audio_cam_name) or cam
        if audio_cam.offset_model is None:
            audio_start_cam = seg_start
        else:
            audio_start_cam = seg_start + float(audio_cam.offset_model.offset_at(seg_start))
        if audio_start_cam < 0:
            audio_start_cam = 0.0
        audio_parts.append((audio_cam.video_path, audio_start_cam, seg_dur))

    if not segment_files:
        raise RuntimeError("Не удалось создать сегменты для рендера.")

    concat_list = seg_dir / "concat.txt"
    concat_list.write_text(
        "\n".join([f"file '{p}'" for p in segment_files]),
        encoding="utf-8",
    )

    from .utils import run_ffmpeg

    vcfg = cfg.video
    crf = {"low": 28, "medium": 23, "high": 18}.get(vcfg.output_quality, 18)

    tmp_video = seg_dir / "concat_video.mp4"
    run_ffmpeg(
        [
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-fflags",
            "+genpts",
            "-an",
            "-c:v",
            vcfg.output_codec,
            "-crf",
            str(crf),
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(tmp_video),
        ]
    )

    audio_source_path = audio_mix_path
    if audio_from_sources:
        audio_sr = int(segment_audio_sr or 48000)
        audio_channels = int(segment_audio_channels or 2)
        audio_source_path = seg_dir / f"{output_path.stem}_sources_audio.wav"
        # If all audio comes from one camera, extract one continuous chunk.
        # This avoids per-segment joins and keeps lip-sync stable on that camera.
        audio_cam_names: set[str] = set()
        for seg in segments:
            cam_name = str(seg.get("audio_camera") or seg.get("camera") or "")
            if cam_name and cam_name in name_to_cam:
                audio_cam_names.add(cam_name)

        built_continuous = False
        if len(audio_cam_names) == 1 and _segments_cover_range(
            segments,
            start_seconds=float(start_seconds),
            end_seconds=float(end_seconds),
        ):
            only_audio_cam_name = next(iter(audio_cam_names))
            only_audio_cam = name_to_cam.get(only_audio_cam_name)
            if only_audio_cam is not None:
                if only_audio_cam.offset_model is None:
                    audio_start_cam = float(start_seconds)
                    audio_end_cam = float(end_seconds)
                else:
                    audio_start_cam = float(start_seconds) + float(
                        only_audio_cam.offset_model.offset_at(float(start_seconds))
                    )
                    audio_end_cam = float(end_seconds) + float(
                        only_audio_cam.offset_model.offset_at(float(end_seconds))
                    )
                if audio_start_cam < 0:
                    audio_start_cam = 0.0
                audio_dur = float(audio_end_cam) - float(audio_start_cam)
                if audio_dur > 0.01:
                    _build_segment_audio_concat(
                        audio_parts=[(only_audio_cam.video_path, float(audio_start_cam), float(audio_dur))],
                        out_path=audio_source_path,
                        audio_sr=audio_sr,
                        audio_channels=audio_channels,
                        accurate_seek=not fast_seek_only,
                    )
                    built_continuous = True

        if not built_continuous:
            _build_segment_audio_concat(
                audio_parts=audio_parts,
                out_path=audio_source_path,
                audio_sr=audio_sr,
                audio_channels=audio_channels,
                accurate_seek=not fast_seek_only,
            )

    # Segment-by-segment cutting can accumulate tiny frame-rounding error in video
    # timeline. Keep video duration aligned with audio content duration to avoid
    # progressive lip-sync drift on long/fragmented clips.
    video_retime_factor: float | None = None
    tmp_video_dur = probe_duration_seconds(tmp_video)
    audio_content_dur = probe_duration_seconds(audio_source_path)
    if tmp_video_dur and audio_content_dur and tmp_video_dur > 0.0 and audio_content_dur > 0.0:
        diff = float(audio_content_dur) - float(tmp_video_dur)
        if abs(diff) >= 0.02:
            factor = float(audio_content_dur) / float(tmp_video_dur)
            # Guardrails: apply only tiny correction (normal case is very close to 1.0).
            if 0.98 <= factor <= 1.02:
                video_retime_factor = factor
    title_lines: list[str] = []
    if title_text:
        raw = _normalize_title_typography(str(title_text))
        split_lines = [" ".join(line.strip().split()) for line in raw.splitlines()]
        split_lines = [line for line in split_lines if line]
        if split_lines:
            title_lines = split_lines
        else:
            one_line = " ".join(raw.strip().split())
            if one_line:
                title_lines = [one_line]
    title_clean = "\n".join(title_lines)
    title_style_key = (title_style or "card").strip().lower()

    subtitle_path_effective = subtitle_path
    if (
        subtitle_path is not None
        and video_retime_factor is not None
        and abs(float(video_retime_factor) - 1.0) >= 1e-6
    ):
        subtitle_path_effective = seg_dir / f"{output_path.stem}_subs_retimed.ass"
        _retime_ass_dialogues(
            source_path=Path(subtitle_path),
            out_path=subtitle_path_effective,
            factor=float(video_retime_factor),
        )

    vf_parts: list[str] = []
    if video_retime_factor is not None:
        vf_parts.append(f"setpts={video_retime_factor:.10f}*PTS")

    if title_clean:
        title_dir = output_path.parent / "title_cards"
        title_dir.mkdir(parents=True, exist_ok=True)
        title_file = title_dir / "title.txt"
        title_len = max(0.0, float(title_duration))
        font_choice = title_font or _default_title_font()
        fontfile = _escape_filter_path(font_choice) if font_choice else None
        font_arg = f":fontfile='{fontfile}'" if fontfile else ""
        if title_style_key == "card":
            wrapped = _wrap_title_text(" ".join(title_lines), max_chars=28, max_lines=3)
            title_file.write_text(wrapped, encoding="utf-8")
            textfile = _escape_filter_path(title_file)
            vf_parts.append(f"tpad=start_duration={title_len:.3f}:start_mode=add:color=black")
            vf_parts.append(
                (
                    "drawtext="
                    f"{font_arg}"
                    f":textfile='{textfile}'"
                    f":fontcolor=white:fontsize={int(title_fontsize)}"
                    ":line_spacing=10"
                    ":x=(w-text_w)/2:y=(h-text_h)/2"
                    f":enable='between(t,0,{title_len:.3f})'"
                )
            )
        elif title_style_key == "overlay_center_dark":
            # Dark centered top panel: high contrast, no yellow accents.
            panel_x = "iw*0.137"
            panel_w = "iw*0.726"
            panel_y = "ih*0.06875"
            panel_h = "ih*0.1302"
            overlay_fontsize = max(16, int(round(float(title_fontsize) * 0.9)))
            panel_px = 1080.0 * 0.726
            panel_pad_px = max(18, int(round(float(overlay_fontsize) * 0.55)))

            manual_lines = [line.strip() for line in title_lines if line.strip()]
            overlay_pages: list[list[str]] = []
            auto_overlay_text: str | None = None

            if len(manual_lines) >= 2:
                overlay_pages = [manual_lines[:2]]
            else:
                auto_overlay_text = " ".join(manual_lines).strip()
                if auto_overlay_text:
                    fit_sizes: list[int] = []
                    for candidate_fs in (
                        overlay_fontsize,
                        max(16, overlay_fontsize - 2),
                        max(16, overlay_fontsize - 4),
                    ):
                        if candidate_fs not in fit_sizes:
                            fit_sizes.append(candidate_fs)

                    best_pages: list[list[str]] | None = None
                    best_fontsize = overlay_fontsize
                    for candidate_fs in fit_sizes:
                        candidate_pad = max(18, int(round(float(candidate_fs) * 0.55)))
                        pages_try = _overlay_title_pages(
                            auto_overlay_text,
                            fontsize=candidate_fs,
                            panel_width_px=panel_px,
                            side_padding_px=float(candidate_pad),
                            lines_per_page=2,
                        )
                        if not pages_try:
                            continue
                        if best_pages is None or len(pages_try) < len(best_pages):
                            best_pages = pages_try
                            best_fontsize = candidate_fs
                        elif len(pages_try) == len(best_pages) and candidate_fs > best_fontsize:
                            best_pages = pages_try
                            best_fontsize = candidate_fs

                    if best_pages:
                        overlay_pages = best_pages
                        overlay_fontsize = best_fontsize
                        panel_pad_px = max(18, int(round(float(overlay_fontsize) * 0.55)))

            if not overlay_pages:
                fallback_text = " ".join(title_lines).strip()
                overlay_pages = [[fallback_text]] if fallback_text else [[""]]

            # Ensure lines fit into panel by reducing font to a readable minimum.
            while overlay_fontsize > 16:
                max_line_px = 0.0
                for page in overlay_pages:
                    for line in page[:2]:
                        max_line_px = max(max_line_px, _estimate_overlay_title_width_px(line, overlay_fontsize))
                if max_line_px <= (panel_px - 2.0 * float(panel_pad_px)):
                    break
                overlay_fontsize = max(16, overlay_fontsize - 1)
                panel_pad_px = max(18, int(round(float(overlay_fontsize) * 0.55)))

            if auto_overlay_text:
                pages_reflow = _overlay_title_pages(
                    auto_overlay_text,
                    fontsize=overlay_fontsize,
                    panel_width_px=panel_px,
                    side_padding_px=float(panel_pad_px),
                    lines_per_page=2,
                )
                if pages_reflow:
                    overlay_pages = pages_reflow

            vf_parts.append(
                "drawbox="
                f"x={panel_x}:y={panel_y}:w={panel_w}:h={panel_h}:"
                "color=black@0.70:t=fill"
                f":enable='between(t,0,{title_len:.3f})'"
            )
            vf_parts.append(
                "drawbox="
                f"x={panel_x}:y={panel_y}:w={panel_w}:h={panel_h}:"
                "color=white@0.22:t=2"
                f":enable='between(t,0,{title_len:.3f})'"
            )

            fs = int(overlay_fontsize)
            center_y = "h*0.06875+(h*0.1302)/2"
            y_single = "h*0.06875+((h*0.1302)-text_h)/2"
            y1 = f"{center_y}-({fs}*0.72)"
            y2 = f"{center_y}+({fs}*0.14)"
            common = (
                f":fontcolor=white:fontsize={fs}"
                ":borderw=2:bordercolor=black@0.60"
                ":shadowx=0:shadowy=3"
                ":x=(w-text_w)/2"
            )

            def _append_overlay_page(page_lines: list[str], *, start_t: float, end_t: float, page_idx: int) -> None:
                lines = [ln for ln in page_lines[:2] if str(ln).strip()]
                if not lines:
                    return
                if end_t <= start_t:
                    return
                if len(lines) == 1:
                    page_file = title_dir / f"title_page{page_idx}_line1.txt"
                    page_file.write_text(lines[0], encoding="utf-8")
                    textfile = _escape_filter_path(page_file)
                    vf_parts.append(
                        (
                            "drawtext="
                            f"{font_arg}"
                            f":textfile='{textfile}'"
                            f"{common}"
                            f":y={y_single}"
                            f":enable='between(t,{start_t:.3f},{end_t:.3f})'"
                        )
                    )
                    return

                line1_file = title_dir / f"title_page{page_idx}_line1.txt"
                line2_file = title_dir / f"title_page{page_idx}_line2.txt"
                line1_file.write_text(lines[0], encoding="utf-8")
                line2_file.write_text(lines[1], encoding="utf-8")
                textfile1 = _escape_filter_path(line1_file)
                textfile2 = _escape_filter_path(line2_file)
                vf_parts.append(
                    (
                        "drawtext="
                        f"{font_arg}"
                        f":textfile='{textfile1}'"
                        f"{common}"
                        f":y={y1}"
                        f":enable='between(t,{start_t:.3f},{end_t:.3f})'"
                    )
                )
                vf_parts.append(
                    (
                        "drawtext="
                        f"{font_arg}"
                        f":textfile='{textfile2}'"
                        f"{common}"
                        f":y={y2}"
                        f":enable='between(t,{start_t:.3f},{end_t:.3f})'"
                    )
                )

            if len(overlay_pages) <= 1:
                _append_overlay_page(overlay_pages[0], start_t=0.0, end_t=title_len, page_idx=0)
            else:
                weights = [max(1.0, float(sum(len(x) for x in page))) for page in overlay_pages]
                total_weight = sum(weights) or 1.0
                cursor = 0.0
                for idx, page in enumerate(overlay_pages):
                    if idx == len(overlay_pages) - 1:
                        end_t = title_len
                    else:
                        end_t = min(title_len, cursor + title_len * (weights[idx] / total_weight))
                    if end_t <= cursor:
                        end_t = min(title_len, cursor + 0.05)
                    _append_overlay_page(page, start_t=cursor, end_t=end_t, page_idx=idx)
                    cursor = end_t
        else:
            raise RuntimeError(
                f"Неподдерживаемый title-style '{title_style}'. Используй: card, overlay_center_dark."
            )

    if subtitle_path_effective:
        vf_parts.append(f"subtitles='{_escape_filter_path(subtitle_path_effective)}'")

    vf = ",".join(vf_parts) if vf_parts else None

    audio_sr = _probe_wav_sr(audio_source_path) or 16000
    audio_channels = _probe_wav_channels(audio_source_path) or 1
    audio_out_sr = None
    if audio_sr:
        audio_out_sr = 48000 if audio_sr < 32000 else audio_sr
    audio_bitrate = "192k"
    if audio_out_sr is not None:
        if audio_out_sr <= 24000:
            audio_bitrate = "64k"
        elif audio_out_sr <= 32000:
            audio_bitrate = "96k"

    audio_codec, audio_is_lossy = _resolve_audio_encoder(audio_encoder)
    audio_codec_args = ["-c:a", audio_codec]
    if audio_is_lossy:
        audio_codec_args += ["-b:a", audio_bitrate]
    if audio_out_sr is not None:
        audio_codec_args += ["-ar", str(audio_out_sr)]
    audio_fallback = {"aac_at": "aac", "alac_at": "alac"}.get(audio_codec)

    args = [
        "-y",
        "-i",
        str(tmp_video),
        "-i",
        str(audio_source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    if vf:
        args += [
            "-vf",
            vf,
            "-c:v",
            vcfg.output_codec,
            "-crf",
            str(crf),
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        args += ["-c:v", "copy"]

    if title_clean and title_style_key == "card":
        delay_ms = int(round(max(0.0, float(title_duration)) * 1000.0))
        if delay_ms > 0:
            delays = "|".join([str(delay_ms)] * max(1, int(audio_channels)))
            args += ["-af", f"adelay={delays}"]

    args += audio_codec_args
    args += [str(output_path)]
    args_fallback = _replace_audio_codec_arg(args, audio_fallback) if audio_fallback else None

    try:
        run_ffmpeg(args)
    except RuntimeError:
        if args_fallback is not None:
            run_ffmpeg(args_fallback)
        else:
            raise
    else:
        if args_fallback is not None and not _audio_stream_decodes(output_path):
            run_ffmpeg(args_fallback)

    return output_path


def _build_subtitles_from_rendered_audio(
    cfg: AppConfig,
    *,
    video_path: Path,
    transcript_path: Path,
    subtitle_path: Path,
    subtitle_max_chars: int,
    subtitle_max_lines: int,
    subtitle_preset: str,
) -> Path:
    """
    Build subtitles from the already-rendered clip audio timeline.
    This avoids desync when subtitle transcript timeline differs from final audio.
    """
    ensure_dirs(transcript_path.parent, subtitle_path.parent)
    audio_tmp = transcript_path.with_suffix(".wav")
    ensure_dirs(audio_tmp.parent)

    extract_wav_16k_mono(str(video_path), audio_tmp)

    # Local import avoids hard dependency during non-subtitle runs.
    from .transcriber import Transcriber

    transcriber = Transcriber(cfg)
    transcript = transcriber.transcribe_audio(
        audio_path=audio_tmp,
        video_path=str(video_path),
        word_timestamps=True,
    )
    write_json(transcript_path, transcript.model_dump(mode="json"))

    video_dur = probe_duration_seconds(video_path)
    if video_dur is None or float(video_dur) <= 0.0:
        raise RuntimeError("Не удалось определить длительность видео для генерации субтитров.")

    build_ass_subtitles(
        transcript_path,
        start_seconds=0.0,
        end_seconds=float(video_dur),
        out_path=subtitle_path,
        max_chars=subtitle_max_chars,
        max_lines=subtitle_max_lines,
        style=subtitle_style_from_preset(subtitle_preset),
    )
    return subtitle_path


def _burn_ass_on_video_copy_audio(
    cfg: AppConfig,
    *,
    input_video: Path,
    subtitle_path: Path,
    output_video: Path,
) -> Path:
    from .utils import run_ffmpeg

    vcfg = cfg.video
    crf = {"low": 28, "medium": 23, "high": 18}.get(vcfg.output_quality, 18)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(input_video),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-vf",
            f"subtitles='{_escape_filter_path(subtitle_path)}'",
            "-c:v",
            vcfg.output_codec,
            "-crf",
            str(crf),
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(output_video),
        ]
    )
    return output_video


def cmd_multicam(
    cfg: AppConfig,
    *,
    ref: str | None,
    others: list[str],
    offsets_paths: list[str],
    ref_channel: str | int | None,
    other_channels: list[str | int | None],
    ref_bias_db: float,
    other_bias_db: list[float],
    ref_vf: str | None,
    other_vf: list[str],
    start_seconds: float,
    end_seconds: float | None,
    duration_seconds: float | None,
    hop_seconds: float,
    frame_ms: float,
    speech_margin_db: float,
    switch_margin_db: float,
    duck_db: float,
    plan_only: bool,
    output: str | None,
    fast_seek_only: bool,
    level_normalize: bool,
    level_max_db: float,
    level_target: str,
    min_shot_seconds: float | None,
    dominance_mode: str,
    speaker_refs_path: str | None,
    speaker_ref_dir: str | None,
    speaker_map_entries: list[str],
    speaker_window_seconds: float,
    speaker_lead_seconds: float,
    speaker_min_similarity: float,
    speaker_margin: float,
    speaker_device: str,
    reaction_every_seconds: float,
    reaction_duration_seconds: float,
    reaction_min_score: float,
    reaction_force_cutaway: bool,
    reaction_search_window_seconds: float,
    reaction_search_step_seconds: float,
    balance_max_dominant_share: float,
    balance_cutaway_seconds: float,
    balance_min_gap_seconds: float,
    audio_crossfade_seconds: float,
    hard_gate: bool,
    audio_sample_rate: int,
    audio_from_sources: bool,
    audio_encoder: str,
    audio_force_camera: str | None,
    speaker_main_camera: str | None,
    subtitle_transcript: str | None,
    subtitle_max_chars: int,
    subtitle_max_lines: int,
    subtitle_preset: str,
    subtitle_align_mode: str,
    pause_accel_profile: str,
    title_text: str | None,
    title_auto: bool,
    title_seconds: float,
    title_font: str | None,
    title_fontsize: int,
    title_style: str,
    cache_segments: bool,
    cache_padding_seconds: float,
    cache_vf: str | None,
) -> dict[str, Path]:
    if not offsets_paths:
        raise RuntimeError("Нужно указать хотя бы один файл offsets.json.")
    if audio_force_camera and not audio_from_sources:
        # Forced external camera audio expects per-segment extraction from sources.
        audio_from_sources = True

    offsets = [load_offset_model(p) for p in offsets_paths]

    if ref is None:
        ref = offsets[0].ref
        if ref is None:
            raise RuntimeError("Не удалось определить ref из offsets.json. Укажи --ref.")
        for off in offsets[1:]:
            if off.ref and off.ref != ref:
                raise RuntimeError("Все offsets.json должны иметь одинаковый ref.")

    if not others:
        others = []
        for off in offsets:
            if not off.other:
                raise RuntimeError("В offsets.json нет поля other. Укажи --other.")
            others.append(off.other)

    if len(others) != len(offsets):
        raise RuntimeError("Количество --other должно совпадать с количеством --offsets.")

    subtitle_align_mode_key = (subtitle_align_mode or "transcript").strip().lower()
    if subtitle_align_mode_key not in {"transcript", "rendered-audio"}:
        raise RuntimeError("Неподдерживаемый --subtitle-align-mode. Используй: transcript, rendered-audio.")

    ref_resolved = resolve_video_source(ref)
    if end_seconds is None:
        if duration_seconds is not None:
            end_seconds = start_seconds + duration_seconds
        else:
            dur = probe_duration_seconds(ref_resolved)
            if dur is None:
                raise RuntimeError("Не удалось определить длительность ref. Укажи --end-seconds.")
            end_seconds = dur

    speaker_map = _parse_speaker_map_entries(speaker_map_entries)

    segments, cameras, meta, speaker_timeline = build_shotlist(
        cfg,
        ref_video=ref,
        other_videos=others,
        offsets=offsets,
        ref_channel=ref_channel,
        other_channels=other_channels,
        ref_bias_db=ref_bias_db,
        other_bias_db=other_bias_db,
        ref_vf=ref_vf,
        other_vf=other_vf,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        frame_ms=frame_ms,
        hop_seconds=hop_seconds,
        speech_margin_db=speech_margin_db,
        switch_margin_db=switch_margin_db,
        min_shot_seconds=min_shot_seconds,
        fast_seek_only=fast_seek_only,
        level_normalize=level_normalize,
        level_max_db=level_max_db,
        level_target=level_target,
        dominance_mode=dominance_mode,
        speaker_refs_path=speaker_refs_path,
        speaker_ref_dir=speaker_ref_dir,
        speaker_map=speaker_map,
        speaker_window_seconds=speaker_window_seconds,
        speaker_lead_seconds=speaker_lead_seconds,
        speaker_min_similarity=speaker_min_similarity,
        speaker_margin=speaker_margin,
        speaker_device=speaker_device,
        reaction_every_seconds=reaction_every_seconds,
        reaction_duration_seconds=reaction_duration_seconds,
    )

    cam_names = {cam.name for cam in cameras}
    forced_audio_cam: str | None = None
    if audio_force_camera:
        forced_audio_cam = _normalize_camera_name(audio_force_camera, cam_names)
        if forced_audio_cam is None:
            raise RuntimeError(
                f"Unknown audio camera '{audio_force_camera}'. Available: {', '.join(sorted(cam_names))}"
            )
        for seg in segments:
            seg["audio_camera"] = forced_audio_cam

    main_cam_raw = speaker_main_camera
    main_cam: str | None = None
    if main_cam_raw:
        main_cam = _normalize_camera_name(main_cam_raw, cam_names)
        if main_cam is None:
            raise RuntimeError(
                f"Unknown speaker main camera '{main_cam_raw}'. Available: {', '.join(sorted(cam_names))}"
            )
        segments = _pin_video_to_camera(segments, camera_name=main_cam)
        if reaction_every_seconds <= 0:
            reaction_every_seconds = 10.0
        if reaction_duration_seconds < 2.0:
            reaction_duration_seconds = 2.4

    if reaction_every_seconds > 0 and reaction_duration_seconds > 0:
        segments = _insert_reaction_segments(
            segments,
            cameras,
            reaction_every_seconds=float(reaction_every_seconds),
            reaction_duration_seconds=float(reaction_duration_seconds),
            reaction_min_score=float(reaction_min_score),
            reaction_force_cutaway=bool(reaction_force_cutaway),
            reaction_search_window_seconds=float(reaction_search_window_seconds),
            reaction_search_step_seconds=float(reaction_search_step_seconds),
        )
    segments = _coalesce_segments(segments)

    balance_info: dict[str, Any] | None = None
    if float(balance_max_dominant_share) < 0.999:
        segments, balance_info = _balance_camera_focus(
            segments,
            cameras,
            max_dominant_share=float(balance_max_dominant_share),
            cutaway_seconds=float(balance_cutaway_seconds),
            min_gap_seconds=float(balance_min_gap_seconds),
            min_side_seconds=0.8,
            search_window_seconds=1.2,
            search_step_seconds=0.4,
            min_cutaway_seconds=1.6,
        )

    pause_cut_intervals: list[tuple[float, float]] = []
    pause_profile_key = (pause_accel_profile or "none").strip().lower()
    if _get_pause_accel_profile(pause_profile_key) is not None:
        if not subtitle_transcript:
            raise RuntimeError(
                "Для ускорения пауз нужен --subtitle-transcript (по нему определяются интервалы речи)."
            )
        if not audio_from_sources:
            raise RuntimeError("Для ускорения пауз включи --audio-from-sources для стабильного lip-sync.")
        pause_cut_intervals = _build_pause_cut_intervals(
            subtitle_transcript,
            start_seconds=float(start_seconds),
            end_seconds=float(end_seconds),
            profile_name=pause_profile_key,
        )
        if pause_cut_intervals:
            segments = _apply_cut_intervals_to_segments(
                segments,
                cut_intervals=pause_cut_intervals,
                min_piece_seconds=0.05,
            )
            if not segments:
                raise RuntimeError("После ускорения пауз не осталось видеосегментов для рендера.")

    post_min_target = min(1.2, max(0.2, float(meta.get("min_shot_seconds", 1.2))))
    segments, post_min_info = _enforce_min_segment_duration(
        segments,
        min_seconds=post_min_target,
        adjacency_epsilon=0.06,
    )

    meta["reaction_mode"] = (
        {
            "reaction_every_seconds": float(reaction_every_seconds),
            "reaction_duration_seconds": float(reaction_duration_seconds),
            "reaction_min_score": float(reaction_min_score),
            "reaction_force_cutaway": bool(reaction_force_cutaway),
            "reaction_search_window_seconds": float(reaction_search_window_seconds),
            "reaction_search_step_seconds": float(reaction_search_step_seconds),
        }
        if reaction_every_seconds > 0 and reaction_duration_seconds > 0
        else None
    )
    meta["speaker_primary_video_camera"] = main_cam
    meta["camera_balance"] = balance_info
    meta["post_min_shot"] = post_min_info
    removed_pause_seconds = sum((end - start) for start, end in pause_cut_intervals)
    meta["pause_accel"] = (
        {
            "profile": pause_profile_key,
            "cuts_count": len(pause_cut_intervals),
            "removed_seconds": round(removed_pause_seconds, 3),
            "source_duration_seconds": round(float(end_seconds) - float(start_seconds), 3),
            "output_duration_estimate_seconds": round(
                max(0.0, float(end_seconds) - float(start_seconds) - removed_pause_seconds),
                3,
            ),
            "cuts": [
                {
                    "start": round(float(c_start), 3),
                    "end": round(float(c_end), 3),
                    "duration": round(float(c_end - c_start), 3),
                }
                for c_start, c_end in pause_cut_intervals
            ],
        }
        if _get_pause_accel_profile(pause_profile_key) is not None
        else None
    )

    resolved_title_text = title_text
    if title_auto and subtitle_transcript:
        auto_title = _auto_title_from_transcript(
            subtitle_transcript,
            start_seconds=float(start_seconds),
            end_seconds=float(end_seconds),
        )
        if auto_title:
            resolved_title_text = auto_title
    if resolved_title_text:
        resolved_title_text = _polish_title_text(resolved_title_text)
    effective_title_seconds = float(title_seconds)
    if resolved_title_text and (title_style or "card").strip().lower() == "overlay_center_dark":
        effective_title_seconds = _effective_overlay_title_seconds(
            resolved_title_text,
            float(title_seconds),
        )
    meta["title_text"] = resolved_title_text
    meta["title_style"] = title_style
    meta["title_seconds_effective"] = effective_title_seconds

    storage = get_storage_paths(cfg)
    ensure_dirs(storage.output)
    suffix = f"_s{start_seconds:g}_e{end_seconds:g}"
    stem = _video_stem(ref)

    shotlist_path = storage.output / f"{stem}_shotlist{suffix}.json"
    write_json(
        shotlist_path,
        {
            "schema_version": "1.0",
            "meta": meta,
            "segments": segments,
        },
    )

    mix_path = storage.output / f"{stem}_mix{suffix}.wav"
    make_gated_mix(
        cameras=cameras,
        segments=segments,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        duck_db=duck_db,
        crossfade_seconds=audio_crossfade_seconds,
        hard_gate=hard_gate,
        out_path=mix_path,
    )

    outputs = {"shotlist": shotlist_path, "mix": mix_path}

    if speaker_timeline is not None:
        timeline_path = storage.output / f"{stem}_speaker_timeline{suffix}.json"
        write_json(timeline_path, speaker_timeline)
        outputs["speaker_timeline"] = timeline_path

    if not plan_only:
        subtitle_path = None
        subtitle_from_rendered_audio = bool(subtitle_transcript) and subtitle_align_mode_key == "rendered-audio"
        if subtitle_transcript and not subtitle_from_rendered_audio:
            subtitle_path = storage.output / f"{stem}_subs{suffix}.ass"
            subtitle_offset = (
                float(effective_title_seconds)
                if resolved_title_text and (title_style or "card").strip().lower() == "card"
                else 0.0
            )
            build_ass_subtitles(
                subtitle_transcript,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                out_path=subtitle_path,
                max_chars=subtitle_max_chars,
                max_lines=subtitle_max_lines,
                time_offset=subtitle_offset,
                cut_intervals=pause_cut_intervals,
                style=subtitle_style_from_preset(subtitle_preset),
            )
        if output:
            out_video = Path(output)
        else:
            default_ext = _default_multicam_extension(audio_encoder)
            out_video = storage.output / f"{stem}_multicam{suffix}{default_ext}"
        render_output_path = out_video
        if subtitle_from_rendered_audio:
            render_output_path = out_video.parent / f"{out_video.stem}_nosubs_tmp{out_video.suffix}"
        video_path = render_multicam(
            cfg,
            cameras=cameras,
            segments=segments,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            audio_mix_path=mix_path,
            audio_from_sources=audio_from_sources,
            segment_audio_sr=audio_sample_rate,
            segment_audio_channels=2,
            audio_encoder=audio_encoder,
            output_path=render_output_path,
            fast_seek_only=fast_seek_only,
            subtitle_path=subtitle_path,
            title_text=resolved_title_text,
            title_duration=effective_title_seconds,
            title_font=title_font,
            title_fontsize=title_fontsize,
            title_style=title_style,
            cache_segments=cache_segments,
            cache_padding_seconds=cache_padding_seconds,
            cache_vf=cache_vf,
        )
        if subtitle_from_rendered_audio:
            subtitle_transcript_local = storage.output / f"{out_video.stem}_subs_audio_timeline_transcript.json"
            subtitle_path_local = storage.output / f"{out_video.stem}_subs_audio_timeline.ass"
            _build_subtitles_from_rendered_audio(
                cfg,
                video_path=video_path,
                transcript_path=subtitle_transcript_local,
                subtitle_path=subtitle_path_local,
                subtitle_max_chars=subtitle_max_chars,
                subtitle_max_lines=subtitle_max_lines,
                subtitle_preset=subtitle_preset,
            )
            _burn_ass_on_video_copy_audio(
                cfg,
                input_video=video_path,
                subtitle_path=subtitle_path_local,
                output_video=out_video,
            )
            if video_path != out_video:
                video_path.unlink(missing_ok=True)
            outputs["subtitle"] = subtitle_path_local
            outputs["subtitle_transcript_local"] = subtitle_transcript_local
            outputs["video"] = out_video
        else:
            outputs["video"] = video_path

    return outputs
