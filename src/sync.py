from __future__ import annotations

import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from .audio_extractor import extract_wav_16k_mono, slice_wav
from .utils import ensure_dirs, get_storage_paths, probe_duration_seconds, run_ffmpeg
from .yadisk import YandexDiskError, get_download_url


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


def _xcorr_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    cc = np.fft.irfft(fa * np.conj(fb), nfft)
    # Match numpy.correlate(a,b,'full') order
    cc = np.concatenate((cc[-(len(b) - 1) :], cc[: len(a)]))
    return cc


def _energy_envelope(sig: np.ndarray, sr: int, *, frame_ms: float = 20.0, hop_ms: float = 10.0) -> tuple[np.ndarray, int]:
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
    # Normalize
    if env.size:
        env = env - float(np.mean(env))
    env_sr = int(round(sr / hop))
    return env, env_sr


def _estimate_offset_xcorr(
    a: np.ndarray, b: np.ndarray, sr: int, max_lag_seconds: float
) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        raise RuntimeError("Пустой сигнал для синхронизации.")

    # Normalize
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))

    # Make same length (trim to min)
    n = min(len(a), len(b))
    if n <= 0:
        raise RuntimeError("Недостаточно данных для синхронизации.")
    a = a[:n]
    b = b[:n]

    cc = _xcorr_fft(a, b)
    center = len(b) - 1
    max_lag = int(max_lag_seconds * sr)
    lo = max(0, center - max_lag)
    hi = min(len(cc), center + max_lag + 1)
    idx = int(np.argmax(cc[lo:hi])) + lo
    lag = idx - center

    # When b is delayed vs a, lag is negative; offset should be positive.
    offset = -lag / float(sr)

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    confidence = float(cc[idx] / denom)
    return offset, confidence


def _estimate_offset_gcc_phat(
    a: np.ndarray, b: np.ndarray, sr: int, max_lag_seconds: float, interp: int = 8
) -> tuple[float, float]:
    # GCC-PHAT
    n = a.size + b.size
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    R = A * np.conj(B)
    denom = np.abs(R)
    denom[denom < 1e-9] = 1e-9
    R /= denom
    cc = np.fft.irfft(R, n=interp * nfft)
    max_shift = int(interp * max_lag_seconds * sr)
    if max_shift <= 0:
        max_shift = len(cc) // 2
    max_shift = min(max_shift, (len(cc) // 2) - 1)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc))) - max_shift
    offset = -shift / float(interp * sr)
    # Confidence: peak vs median
    peak = float(np.max(np.abs(cc)))
    med = float(np.median(np.abs(cc)) + 1e-9)
    confidence = peak / med
    return offset, confidence


def estimate_offset_seconds(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    *,
    max_lag_seconds: float = 10.0,
) -> tuple[float, float, str]:
    # Try GCC-PHAT on raw signals
    offset_gcc, conf_gcc = _estimate_offset_gcc_phat(a, b, sr, max_lag_seconds=max_lag_seconds)

    # Try energy-envelope cross-correlation (more robust to mic mismatch)
    env_a, env_sr_a = _energy_envelope(a, sr)
    env_b, env_sr_b = _energy_envelope(b, sr)
    offset_env, conf_env = _estimate_offset_xcorr(
        env_a, env_b, env_sr_a, max_lag_seconds=max_lag_seconds
    )

    # Choose the best by confidence (note: gcc confidence is ratio, env is normalized corr)
    if conf_gcc >= conf_env:
        return offset_gcc, conf_gcc, "gcc_phat"
    return offset_env, conf_env, "envelope_xcorr"


def resolve_video_source(path: str) -> str:
    if path.startswith("yadisk:"):
        yd_path = path[len("yadisk:") :].strip()
        if yd_path and not yd_path.startswith("disk:"):
            yd_path = f"disk:{yd_path}" if yd_path.startswith("/") else f"disk:/{yd_path}"
        return get_download_url(yd_path)
    return path


def _source_stem_and_ext(path: str) -> tuple[str, str]:
    name: str
    if path.startswith("yadisk:"):
        yd_path = path[len("yadisk:") :].strip()
        if yd_path.startswith("disk:"):
            yd_path = yd_path[len("disk:") :]
        name = Path(yd_path).name
    elif path.startswith("http"):
        parsed = urlparse(path)
        q = parse_qs(parsed.query)
        if "filename" in q and q["filename"]:
            name = unquote(q["filename"][0])
        else:
            name = Path(unquote(parsed.path)).name
    else:
        name = Path(path).name
    stem = Path(name).stem or "video"
    ext = Path(name).suffix or ".mov"
    return stem, ext


def _cache_key(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()


def _cache_wav_path(path: str, cache_dir: Path) -> Path:
    stem, _ = _source_stem_and_ext(path)
    digest = _cache_key(path)[:8]
    return cache_dir / f"{stem}_{digest}_16k.wav"


def _load_cache_index(cache_dir: Path) -> dict[str, Any] | None:
    index_path = cache_dir / "cache_index.json"
    if not index_path.exists():
        return None
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_cached_chunk(
    cache_index: dict[str, Any], cache_key: str, start_seconds: float, duration_seconds: float
) -> tuple[Path, float] | None:
    sources = cache_index.get("sources") or {}
    entry = sources.get(cache_key) or {}
    chunks = entry.get("chunks") or []
    for ch in chunks:
        ch_start = float(ch.get("start") or 0.0)
        ch_dur = float(ch.get("duration") or 0.0)
        if ch_start <= start_seconds and (ch_start + ch_dur) >= (start_seconds + duration_seconds):
            return Path(str(ch.get("path"))), start_seconds - ch_start
    return None


def cache_full_audio(
    *,
    cfg,
    ref_path: str,
    other_path: str,
    cache_dir: str | Path | None = None,
    fast_seek_only: bool = False,
    force: bool = False,
    chunk_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    storage = get_storage_paths(cfg)
    cache_root = Path(cache_dir) if cache_dir else (storage.sync / "cache")
    ensure_dirs(cache_root)

    ref_cache = _cache_wav_path(ref_path, cache_root)
    other_cache = _cache_wav_path(other_path, cache_root)
    index: dict[str, Any] = _load_cache_index(cache_root) or {"version": 1, "sources": {}}
    if "sources" not in index or not isinstance(index.get("sources"), dict):
        index["sources"] = {}
    index_path = cache_root / "cache_index.json"

    def _write_index() -> None:
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    def _maybe_resolve(src: str) -> str:
        if src.startswith("yadisk:"):
            return resolve_video_source(src)
        return src

    def _extract_if_needed(src: str, out_path: Path) -> None:
        if out_path.exists() and not force:
            return
        src = _maybe_resolve(src)
        extract_wav_16k_mono(
            src,
            out_path,
            start_seconds=None,
            duration_seconds=None,
            accurate_seek=not fast_seek_only,
        )

    ref_src = ref_path
    other_src = other_path

    def _cache_chunks(
        *, src: str, stem_path: str, total_dur: float | None, key: str
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        t = 0.0
        max_unknown_dur = 12 * 3600.0
        while True:
            if total_dur is not None and t >= total_dur:
                break
            if total_dur is None and t >= max_unknown_dur:
                break
            if total_dur is None:
                dur = float(chunk_seconds)
            else:
                dur = min(float(chunk_seconds), max(0.0, total_dur - t))
            out_path = cache_root / f"{stem_path}_s{int(t)}_d{int(dur)}.wav"
            if not out_path.exists() or force:
                try:
                    src_resolved = _maybe_resolve(src)
                    extract_wav_16k_mono(
                        src_resolved,
                        out_path,
                        start_seconds=t,
                        duration_seconds=dur,
                        accurate_seek=not fast_seek_only,
                    )
                except Exception:
                    if total_dur is None:
                        break
                    raise
            actual_dur = probe_duration_seconds(out_path)
            if actual_dur is None or actual_dur < 1.0:
                if total_dur is None and out_path.exists():
                    out_path.unlink(missing_ok=True)
                break
            chunks.append({"start": float(t), "duration": float(dur), "path": str(out_path)})
            index["sources"][key] = {
                "path": ref_path if key == _cache_key(ref_path) else other_path,
                "full_wav": None,
                "chunks": chunks,
            }
            _write_index()
            t += float(chunk_seconds)
        return chunks

    def _duration_or_none(src: str) -> float | None:
        if duration_seconds is not None:
            return float(duration_seconds)
        dur = probe_duration_seconds(_maybe_resolve(src))
        if dur is None:
            return None
        return float(dur)

    if chunk_seconds is not None:
        key_ref = _cache_key(ref_path)
        key_other = _cache_key(other_path)
        dur_ref = _duration_or_none(ref_src)
        dur_other = _duration_or_none(other_src)
        stem_ref, _ = _source_stem_and_ext(ref_path)
        stem_other, _ = _source_stem_and_ext(other_path)
        ref_chunks = _cache_chunks(
            src=ref_src, stem_path=f"{stem_ref}_{key_ref[:8]}", total_dur=dur_ref, key=key_ref
        )
        other_chunks = _cache_chunks(
            src=other_src,
            stem_path=f"{stem_other}_{key_other[:8]}",
            total_dur=dur_other,
            key=key_other,
        )
        index["sources"][key_ref] = {
            "path": ref_path,
            "full_wav": None,
            "chunks": ref_chunks,
        }
        index["sources"][key_other] = {
            "path": other_path,
            "full_wav": None,
            "chunks": other_chunks,
        }
    else:
        _extract_if_needed(ref_src, ref_cache)
        _extract_if_needed(other_src, other_cache)
        key_ref = _cache_key(ref_path)
        key_other = _cache_key(other_path)
        index["sources"][key_ref] = {
            "path": ref_path,
            "full_wav": str(ref_cache),
            "chunks": [],
        }
        index["sources"][key_other] = {
            "path": other_path,
            "full_wav": str(other_cache),
            "chunks": [],
        }
    _write_index()

    return {
        "cache_dir": str(cache_root),
        "ref_cache": str(ref_cache) if ref_cache.exists() else None,
        "other_cache": str(other_cache) if other_cache.exists() else None,
        "fast_seek_only": bool(fast_seek_only),
        "force": bool(force),
        "chunk_seconds": float(chunk_seconds) if chunk_seconds is not None else None,
        "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
        "cache_index": str(index_path),
    }


def _extract_window_audio(
    *,
    src: str,
    out_path: Path,
    start_seconds: float,
    duration_seconds: float,
    fast_seek_only: bool,
    cache_wav: Path | None = None,
    cache_index: dict[str, Any] | None = None,
    cache_key: str | None = None,
) -> None:
    if cache_wav is not None and cache_wav.exists():
        slice_wav(
            cache_wav,
            out_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        return
    if cache_index is not None and cache_key is not None:
        found = _find_cached_chunk(cache_index, cache_key, start_seconds, duration_seconds)
        if found is not None:
            chunk_path, chunk_offset = found
            slice_wav(
                chunk_path,
                out_path,
                start_seconds=chunk_offset,
                duration_seconds=duration_seconds,
            )
            return
    if isinstance(src, str) and src.startswith("yadisk:"):
        src = resolve_video_source(src)
    extract_wav_16k_mono(
        src,
        out_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        accurate_seek=not fast_seek_only,
    )


def _score_for_method(confidence: float, method: str) -> float:
    if method == "gcc_phat":
        return float(confidence)
    # envelope_xcorr is in [-1,1]; scale for comparison
    return float(confidence) * 10.0


def _best_offset_over_search(
    *,
    ref_src: str,
    other_src: str,
    search_start: float,
    search_duration: float,
    window_seconds: float,
    step_seconds: float,
    max_lag_seconds: float,
    min_score: float,
    storage_sync: Path,
    fast_seek_only: bool,
    cache_ref_wav: Path | None = None,
    cache_other_wav: Path | None = None,
    cache_index: dict[str, Any] | None = None,
    cache_key_ref: str | None = None,
    cache_key_other: str | None = None,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    t = max(0.0, search_start)
    end = t + max(0.0, search_duration)
    ensure_dirs(storage_sync)

    ref_tmp = storage_sync / "sync_ref_tmp.wav"
    other_tmp = storage_sync / "sync_other_tmp.wav"

    def _try_extract_and_load(
        src: str,
        out_path: Path,
        t: float,
        window: float,
        cache_wav: Path | None,
        cache_key: str | None,
    ) -> tuple[np.ndarray, int] | None:
        try:
            _extract_window_audio(
                src=src,
                out_path=out_path,
                start_seconds=t,
                duration_seconds=window,
                fast_seek_only=fast_seek_only,
                cache_wav=cache_wav,
                cache_index=cache_index,
                cache_key=cache_key,
            )
            return _load_wav_mono(out_path)
        except Exception:
            return None

    while t <= end:
        ref_loaded = _try_extract_and_load(
            ref_src, ref_tmp, t, window_seconds, cache_ref_wav, cache_key_ref
        )
        other_loaded = _try_extract_and_load(
            other_src, other_tmp, t, window_seconds, cache_other_wav, cache_key_other
        )
        if ref_loaded is None or other_loaded is None:
            t += step_seconds
            continue
        a, sr_a = ref_loaded
        b, sr_b = other_loaded
        if sr_a != sr_b:
            t += step_seconds
            continue
        offset, conf, method = estimate_offset_seconds(a, b, sr_a, max_lag_seconds=max_lag_seconds)
        score = _score_for_method(conf, method)
        if best is None or score > float(best.get("score", 0.0)):
            best = {
                "window_start": t,
                "offset": offset,
                "confidence": conf,
                "method": method,
                "score": score,
            }
        t += step_seconds

    if best is None:
        raise RuntimeError("Не удалось подобрать окно для синхронизации.")

    if float(best.get("score", 0.0)) < min_score:
        best["warning"] = (
            "Низкая уверенность синхронизации. "
            "Попробуйте увеличить окно или указать start_seconds вручную."
        )
    return best


def _collect_offset_points(
    *,
    ref_src: str,
    other_src: str,
    fit_start: float,
    fit_duration: float,
    window_seconds: float,
    step_seconds: float,
    max_lag_seconds: float,
    min_score: float,
    storage_sync: Path,
    fast_seek_only: bool,
    cache_ref_wav: Path | None = None,
    cache_other_wav: Path | None = None,
    cache_index: dict[str, Any] | None = None,
    cache_key_ref: str | None = None,
    cache_key_other: str | None = None,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    t = max(0.0, fit_start)
    end = t + max(0.0, fit_duration)
    ensure_dirs(storage_sync)

    ref_tmp = storage_sync / "sync_ref_tmp.wav"
    other_tmp = storage_sync / "sync_other_tmp.wav"

    def _try_extract_and_load(
        src: str,
        out_path: Path,
        t: float,
        window: float,
        cache_wav: Path | None,
        cache_key: str | None,
    ) -> tuple[np.ndarray, int] | None:
        try:
            _extract_window_audio(
                src=src,
                out_path=out_path,
                start_seconds=t,
                duration_seconds=window,
                fast_seek_only=fast_seek_only,
                cache_wav=cache_wav,
                cache_index=cache_index,
                cache_key=cache_key,
            )
            return _load_wav_mono(out_path)
        except Exception:
            return None

    while t <= end:
        ref_loaded = _try_extract_and_load(
            ref_src, ref_tmp, t, window_seconds, cache_ref_wav, cache_key_ref
        )
        other_loaded = _try_extract_and_load(
            other_src, other_tmp, t, window_seconds, cache_other_wav, cache_key_other
        )
        if ref_loaded is None or other_loaded is None:
            t += step_seconds
            continue
        a, sr_a = ref_loaded
        b, sr_b = other_loaded
        if sr_a != sr_b:
            t += step_seconds
            continue
        offset, conf, method = estimate_offset_seconds(a, b, sr_a, max_lag_seconds=max_lag_seconds)
        score = _score_for_method(conf, method)
        if score >= min_score:
            points.append(
                {
                    "window_start": t,
                    "offset": float(offset),
                    "confidence": float(conf),
                    "method": str(method),
                    "score": float(score),
                }
            )
        t += step_seconds

    return points


def _fit_drift(points: list[dict[str, Any]]) -> tuple[float, float, float]:
    if len(points) < 2:
        raise RuntimeError("Недостаточно точек для оценки дрейфа (нужно минимум 2).")
    x = np.array([p["window_start"] for p in points], dtype=np.float64)
    y = np.array([p["offset"] for p in points], dtype=np.float64)
    w = np.array([max(float(p.get("score", 0.0)), 1e-6) for p in points], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1, w=w)
    y_pred = slope * x + intercept
    mse = float(np.average((y - y_pred) ** 2, weights=w))
    return float(slope), float(intercept), mse


def _filter_outlier_points(
    points: list[dict[str, Any]], *, max_dev_seconds: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_dev_seconds <= 0 or len(points) < 4:
        return points, []
    offsets = np.array([p["offset"] for p in points], dtype=np.float64)
    median = float(np.median(offsets))
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for p in points:
        if abs(float(p["offset"]) - median) <= max_dev_seconds:
            kept.append(p)
        else:
            dropped.append(p)
    return kept, dropped


def _normalize_rms(x: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
    return x / rms


def _residual_rms(a: np.ndarray, b: np.ndarray, sr: int, offset_seconds: float) -> float:
    shift = int(round(offset_seconds * sr))
    if shift > 0:
        b_shift = b[shift:]
        a_crop = a[: len(b_shift)]
    elif shift < 0:
        b_shift = b[: len(b) + shift]
        a_crop = a[-shift : -shift + len(b_shift)]
    else:
        b_shift = b
        a_crop = a[: len(b)]
    if len(b_shift) == 0:
        return float("inf")
    a_n = _normalize_rms(a_crop)
    b_n = _normalize_rms(b_shift)
    res = a_n - b_n
    return float(np.sqrt(np.mean(res * res)))


def _refine_offset(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    *,
    center_offset: float,
    span_seconds: float,
    step_seconds: float,
) -> tuple[float, float]:
    if step_seconds <= 0:
        raise RuntimeError("refine_step_seconds должен быть > 0.")
    span = abs(float(span_seconds))
    step = float(step_seconds)
    best_offset = center_offset
    best_score = _residual_rms(a, b, sr, best_offset)
    t = center_offset - span
    end = center_offset + span + 1e-9
    while t <= end:
        score = _residual_rms(a, b, sr, t)
        if score < best_score:
            best_score = score
            best_offset = float(t)
        t += step
    return best_offset, best_score


def sync_offset(
    *,
    cfg,
    ref_path: str,
    other_path: str,
    window_seconds: float = 60.0,
    start_seconds: float = 0.0,
    max_lag_seconds: float = 10.0,
    use_end_window: bool = False,
    auto_search: bool = False,
    search_start: float = 0.0,
    search_duration: float = 600.0,
    step_seconds: float = 30.0,
    min_score: float = 3.0,
    fast_seek_only: bool = False,
    fit_drift: bool = False,
    fit_start: float | None = None,
    fit_duration: float | None = None,
    fit_step_seconds: float | None = None,
    fit_min_points: int = 4,
    fit_min_score: float | None = None,
    fit_outlier_seconds: float = 3.0,
    refine_offset: bool = False,
    refine_span_seconds: float = 0.6,
    refine_step_seconds: float = 0.02,
    cache_dir: str | Path | None = None,
) -> Path:
    storage = get_storage_paths(cfg)
    ensure_dirs(storage.sync, storage.audio)

    ref_src = ref_path
    other_src = other_path
    cache_root: Path | None = None
    cache_ref_wav: Path | None = None
    cache_other_wav: Path | None = None
    cache_index: dict[str, Any] | None = None
    cache_key_ref: str | None = None
    cache_key_other: str | None = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        ref_candidate = _cache_wav_path(ref_path, cache_root)
        other_candidate = _cache_wav_path(other_path, cache_root)
        if ref_candidate.exists():
            cache_ref_wav = ref_candidate
        if other_candidate.exists():
            cache_other_wav = other_candidate
        cache_index = _load_cache_index(cache_root)
        if cache_index is not None:
            cache_key_ref = _cache_key(ref_path)
            cache_key_other = _cache_key(other_path)

    if auto_search:
        best = _best_offset_over_search(
            ref_src=ref_src,
            other_src=other_src,
            search_start=search_start,
            search_duration=search_duration,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            max_lag_seconds=max_lag_seconds,
            min_score=min_score,
            storage_sync=storage.sync,
            fast_seek_only=fast_seek_only,
            cache_ref_wav=cache_ref_wav,
            cache_other_wav=cache_other_wav,
            cache_index=cache_index,
            cache_key_ref=cache_key_ref,
            cache_key_other=cache_key_other,
        )
        start_seconds = float(best["window_start"])
        offset_start = float(best["offset"])
        conf_start = float(best["confidence"])
        method_start = str(best["method"])
    else:
        ref_wav = storage.sync / "sync_ref_start.wav"
        other_wav = storage.sync / "sync_other_start.wav"

        _extract_window_audio(
            src=ref_src,
            out_path=ref_wav,
            start_seconds=start_seconds,
            duration_seconds=window_seconds,
            fast_seek_only=fast_seek_only,
            cache_wav=cache_ref_wav,
            cache_index=cache_index,
            cache_key=cache_key_ref,
        )
        _extract_window_audio(
            src=other_src,
            out_path=other_wav,
            start_seconds=start_seconds,
            duration_seconds=window_seconds,
            fast_seek_only=fast_seek_only,
            cache_wav=cache_other_wav,
            cache_index=cache_index,
            cache_key=cache_key_other,
        )

        a, sr_a = _load_wav_mono(ref_wav)
        b, sr_b = _load_wav_mono(other_wav)
        if sr_a != sr_b:
            raise RuntimeError("Частоты дискретизации не совпадают.")

        offset_start, conf_start, method_start = estimate_offset_seconds(
            a, b, sr_a, max_lag_seconds=max_lag_seconds
        )

    result: dict[str, Any] = {
        "ref": ref_path,
        "other": other_path,
        "start_seconds": start_seconds,
        "window_seconds": window_seconds,
        "max_lag_seconds": max_lag_seconds,
        "offset_start": offset_start,
        "confidence_start": conf_start,
        "method_start": method_start,
        "auto_search": auto_search,
    }
    if auto_search:
        result.update(
            {
                "search_start": search_start,
                "search_duration": search_duration,
                "step_seconds": step_seconds,
                "min_score": min_score,
                "fast_seek_only": fast_seek_only,
            }
        )

    if cache_ref_wav or cache_other_wav:
        result["cache"] = {
            "cache_dir": str(cache_root) if cache_root is not None else None,
            "ref_cache": str(cache_ref_wav) if cache_ref_wav is not None else None,
            "other_cache": str(cache_other_wav) if cache_other_wav is not None else None,
        }

    if fit_drift:
        if fit_start is None:
            fit_start = start_seconds
        if fit_duration is None:
            fit_duration = search_duration
        if fit_step_seconds is None:
            fit_step_seconds = step_seconds
        if fit_min_score is None:
            fit_min_score = min_score
        points = _collect_offset_points(
            ref_src=ref_src,
            other_src=other_src,
            fit_start=fit_start,
            fit_duration=fit_duration,
            window_seconds=window_seconds,
            step_seconds=fit_step_seconds,
            max_lag_seconds=max_lag_seconds,
            min_score=fit_min_score,
            storage_sync=storage.sync,
            fast_seek_only=fast_seek_only,
            cache_ref_wav=cache_ref_wav,
            cache_other_wav=cache_other_wav,
            cache_index=cache_index,
            cache_key_ref=cache_key_ref,
            cache_key_other=cache_key_other,
        )
        points_used, points_dropped = _filter_outlier_points(
            points, max_dev_seconds=fit_outlier_seconds
        )
        if len(points_used) < fit_min_points:
            raise RuntimeError(
                f"Недостаточно точек для дрейфа: {len(points_used)} (нужно минимум {fit_min_points})."
            )
        slope, intercept, mse = _fit_drift(points_used)
        drift_per_hour = slope * 3600.0
        ppm = slope * 1_000_000.0
        result["fit_drift"] = {
            "fit_start": fit_start,
            "fit_duration": fit_duration,
            "fit_step_seconds": fit_step_seconds,
            "fit_min_score": fit_min_score,
            "fit_min_points": fit_min_points,
            "fit_outlier_seconds": fit_outlier_seconds,
            "points": points_used,
            "points_dropped": points_dropped,
            "slope_per_sec": slope,
            "intercept_seconds": intercept,
            "mse": mse,
            "drift_per_hour": drift_per_hour,
            "ppm": ppm,
        }
        # Expose top-level drift for convenience.
        result["drift_per_hour"] = drift_per_hour

    # Save the exact window audio used for the chosen offset
    def _extract_window_wav(
        src: str, out_path: Path, cache_wav: Path | None, cache_key: str | None
    ) -> None:
        try:
            _extract_window_audio(
                src=src,
                out_path=out_path,
                start_seconds=start_seconds,
                duration_seconds=window_seconds,
                fast_seek_only=fast_seek_only,
                cache_wav=cache_wav,
                cache_index=cache_index,
                cache_key=cache_key,
            )
            _load_wav_mono(out_path)
            return
        except Exception as e:
            raise RuntimeError("Не удалось извлечь корректный WAV для проверки синхронизации.") from e

    ref_best = storage.sync / "sync_ref_best.wav"
    other_best = storage.sync / "sync_other_best.wav"
    _extract_window_wav(ref_src, ref_best, cache_ref_wav, cache_key_ref)
    _extract_window_wav(other_src, other_best, cache_other_wav, cache_key_other)
    result["ref_wav"] = str(ref_best)
    result["other_wav"] = str(other_best)

    # Recompute offset from the exact saved windows to keep JSON + mix consistent.
    a_best, sr_best_a = _load_wav_mono(ref_best)
    b_best, sr_best_b = _load_wav_mono(other_best)
    if sr_best_a != sr_best_b:
        raise RuntimeError("Частоты дискретизации не совпадают.")
    offset_start, conf_start, method_start = estimate_offset_seconds(
        a_best, b_best, sr_best_a, max_lag_seconds=max_lag_seconds
    )
    if refine_offset:
        refined, refined_score = _refine_offset(
            a_best,
            b_best,
            sr_best_a,
            center_offset=offset_start,
            span_seconds=refine_span_seconds,
            step_seconds=refine_step_seconds,
        )
        result["offset_start_raw"] = float(offset_start)
        result["confidence_start_raw"] = float(conf_start)
        result["method_start_raw"] = str(method_start)
        result["refine_span_seconds"] = float(refine_span_seconds)
        result["refine_step_seconds"] = float(refine_step_seconds)
        result["refine_score"] = float(refined_score)
        offset_start = float(refined)
        method_start = "refine_rms"
    result["offset_start"] = offset_start
    result["confidence_start"] = conf_start
    result["method_start"] = method_start

    if use_end_window:
        dur_ref = probe_duration_seconds(ref_src)
        dur_other = probe_duration_seconds(other_src)
        if dur_ref is not None and dur_other is not None:
            end_start = min(dur_ref, dur_other) - window_seconds - 1.0
            if end_start > 0:
                ref_wav_end = storage.sync / "sync_ref_end.wav"
                other_wav_end = storage.sync / "sync_other_end.wav"
                extract_wav_16k_mono(
                    ref_src,
                    ref_wav_end,
                    start_seconds=end_start,
                    duration_seconds=window_seconds,
                    accurate_seek=True,
                )
                extract_wav_16k_mono(
                    other_src,
                    other_wav_end,
                    start_seconds=end_start,
                    duration_seconds=window_seconds,
                    accurate_seek=True,
                )
                a2, sr2 = _load_wav_mono(ref_wav_end)
                b2, sr3 = _load_wav_mono(other_wav_end)
                if sr2 == sr3:
                    offset_end, conf_end, method_end = estimate_offset_seconds(
                        a2, b2, sr2, max_lag_seconds=max_lag_seconds
                    )
                    result.update(
                        {
                            "end_start_seconds": end_start,
                            "offset_end": offset_end,
                            "confidence_end": conf_end,
                            "method_end": method_end,
                        }
                    )
                    # Drift per hour (seconds offset change per hour)
                    dt = end_start - start_seconds
                    if dt > 0:
                        drift_per_sec = (offset_end - offset_start) / dt
                        result["drift_per_hour"] = drift_per_sec * 3600.0

    out_path = storage.sync / "offsets.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def make_sync_mix(
    *,
    cfg,
    ref_wav: str | Path,
    other_wav: str | Path,
    offset_seconds: float,
    out_name: str = "sync_mix_check.wav",
) -> Path:
    storage = get_storage_paths(cfg)
    ensure_dirs(storage.sync)
    out_path = storage.sync / out_name

    # If offset is positive, other is delayed. We shift it forward with negative itsoffset.
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(ref_wav),
        "-itsoffset",
        str(-float(offset_seconds)),
        "-i",
        str(other_wav),
        "-filter_complex",
        "amix=inputs=2:normalize=0",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return out_path


def apply_sync_trim(
    *,
    cfg,
    ref_path: str,
    other_path: str,
    offset_seconds: float,
    output_dir: str | Path | None = None,
    reencode: bool = False,
    fast_seek_only: bool = False,
    verify: bool = True,
    verify_window_seconds: float = 20.0,
    verify_max_lag_seconds: float = 2.0,
    verify_tolerance_seconds: float = 0.15,
    verify_start_seconds: float = 0.0,
    make_checks: bool = True,
    copy_unchanged: bool = False,
    audio_only: bool = False,
    audio_window_seconds: float | None = None,
    auto_correct: bool = False,
    auto_correct_iters: int = 2,
    cache_dir: str | Path | None = None,
) -> Path:
    storage = get_storage_paths(cfg)
    out_dir = Path(output_dir) if output_dir else (storage.sync / "aligned")
    ensure_dirs(out_dir)

    ref_src = ref_path
    other_src = other_path

    cache_ref_wav: Path | None = None
    cache_other_wav: Path | None = None
    cache_index: dict[str, Any] | None = None
    cache_key_ref: str | None = None
    cache_key_other: str | None = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        ref_candidate = _cache_wav_path(ref_path, cache_root)
        other_candidate = _cache_wav_path(other_path, cache_root)
        if ref_candidate.exists():
            cache_ref_wav = ref_candidate
        if other_candidate.exists():
            cache_other_wav = other_candidate
        cache_index = _load_cache_index(cache_root)
        if cache_index is not None:
            cache_key_ref = _cache_key(ref_path)
            cache_key_other = _cache_key(other_path)

    ref_stem, ref_ext = _source_stem_and_ext(ref_path)
    other_stem, other_ext = _source_stem_and_ext(other_path)

    ref_out = out_dir / f"{ref_stem}_aligned{ref_ext}"
    other_out = out_dir / f"{other_stem}_aligned{other_ext}"
    ref_audio = out_dir / f"{ref_stem}_aligned.wav"
    other_audio = out_dir / f"{other_stem}_aligned.wav"

    def _trim_media(src: str, out_path: Path, start_seconds: float) -> None:
        if src.startswith("yadisk:"):
            src = resolve_video_source(src)
        args: list[str] = ["-y"]
        if not reencode:
            # Fast trim with stream copy (keyframe-accurate).
            if start_seconds > 0:
                args += ["-ss", str(start_seconds)]
            args += ["-i", src, "-c", "copy", str(out_path)]
        else:
            # Accurate trim with re-encode (unless fast_seek_only).
            if start_seconds > 0 and fast_seek_only:
                args += ["-ss", str(start_seconds)]
            args += ["-i", src]
            if start_seconds > 0 and not fast_seek_only:
                args += ["-ss", str(start_seconds)]
            args += [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(out_path),
            ]
        run_ffmpeg(args)

    def _run_once(current_offset: float) -> tuple[str, str, dict[str, Any] | None, float, float]:
        ref_trim = float(current_offset) if current_offset > 0 else 0.0
        other_trim = float(-current_offset) if current_offset < 0 else 0.0

        if audio_only:
            dur = audio_window_seconds if audio_window_seconds is not None else verify_window_seconds
            _extract_window_audio(
                src=ref_src,
                out_path=ref_audio,
                start_seconds=ref_trim,
                duration_seconds=dur,
                fast_seek_only=fast_seek_only,
                cache_wav=cache_ref_wav,
                cache_index=cache_index,
                cache_key=cache_key_ref,
            )
            _extract_window_audio(
                src=other_src,
                out_path=other_audio,
                start_seconds=other_trim,
                duration_seconds=dur,
                fast_seek_only=fast_seek_only,
                cache_wav=cache_other_wav,
                cache_index=cache_index,
                cache_key=cache_key_other,
            )
            ref_out_path = str(ref_audio)
            other_out_path = str(other_audio)
        else:
            if ref_trim <= 1e-9 and not reencode and not copy_unchanged:
                ref_out_path = str(ref_path)
            else:
                _trim_media(ref_src, ref_out, ref_trim)
                ref_out_path = str(ref_out)

            if other_trim <= 1e-9 and not reencode and not copy_unchanged:
                other_out_path = str(other_path)
            else:
                _trim_media(other_src, other_out, other_trim)
                other_out_path = str(other_out)

        verify_data: dict[str, Any] | None = None
        if verify:
            ref_wav = out_dir / f"{ref_stem}_aligned_check.wav"
            other_wav = out_dir / f"{other_stem}_aligned_check.wav"
            if audio_only:
                _extract_window_audio(
                    src=ref_src,
                    out_path=ref_wav,
                    start_seconds=ref_trim + float(verify_start_seconds),
                    duration_seconds=verify_window_seconds,
                    fast_seek_only=False,
                    cache_wav=cache_ref_wav,
                    cache_index=cache_index,
                    cache_key=cache_key_ref,
                )
                _extract_window_audio(
                    src=other_src,
                    out_path=other_wav,
                    start_seconds=other_trim + float(verify_start_seconds),
                    duration_seconds=verify_window_seconds,
                    fast_seek_only=False,
                    cache_wav=cache_other_wav,
                    cache_index=cache_index,
                    cache_key=cache_key_other,
                )
            else:
                extract_wav_16k_mono(
                    ref_out_path,
                    ref_wav,
                    start_seconds=float(verify_start_seconds),
                    duration_seconds=verify_window_seconds,
                    accurate_seek=True,
                )
                extract_wav_16k_mono(
                    other_out_path,
                    other_wav,
                    start_seconds=float(verify_start_seconds),
                    duration_seconds=verify_window_seconds,
                    accurate_seek=True,
                )
            a, sr_a = _load_wav_mono(ref_wav)
            b, sr_b = _load_wav_mono(other_wav)
            if sr_a != sr_b:
                raise RuntimeError("Частоты дискретизации не совпадают для проверки.")
            v_offset, v_conf, v_method = estimate_offset_seconds(
                a, b, sr_a, max_lag_seconds=verify_max_lag_seconds
            )
            verify_data = {
                "start_seconds": float(verify_start_seconds),
                "window_seconds": float(verify_window_seconds),
                "max_lag_seconds": float(verify_max_lag_seconds),
                "offset_seconds": float(v_offset),
                "confidence": float(v_conf),
                "method": str(v_method),
                "tolerance_seconds": float(verify_tolerance_seconds),
                "ok": abs(float(v_offset)) <= float(verify_tolerance_seconds),
                "ref_check_wav": str(ref_wav),
                "other_check_wav": str(other_wav),
            }

            if make_checks:
                lr_out = out_dir / "aligned_lr_check.wav"
                null_out = out_dir / "aligned_null_check.wav"
                run_ffmpeg(
                    [
                        "-y",
                        "-i",
                        str(ref_wav),
                        "-i",
                        str(other_wav),
                        "-filter_complex",
                        "[0:a]dynaudnorm=f=150:g=15[a0];"
                        "[1:a]dynaudnorm=f=150:g=15[a1];"
                        "[a0][a1]amerge=inputs=2,pan=stereo|c0=c0|c1=c1",
                        "-c:a",
                        "pcm_s16le",
                        str(lr_out),
                    ]
                )
                run_ffmpeg(
                    [
                        "-y",
                        "-i",
                        str(ref_wav),
                        "-i",
                        str(other_wav),
                        "-filter_complex",
                        "[0:a]dynaudnorm=f=150:g=15[a0];"
                        "[1:a]dynaudnorm=f=150:g=15,volume=-1[a1];"
                        "[a0][a1]amix=inputs=2:normalize=0",
                        "-c:a",
                        "pcm_s16le",
                        str(null_out),
                    ]
                )
                verify_data["lr_check_wav"] = str(lr_out)
                verify_data["null_check_wav"] = str(null_out)

        return ref_out_path, other_out_path, verify_data, ref_trim, other_trim

    current_offset = float(offset_seconds)
    corrections: list[dict[str, Any]] = []
    ref_out_path = ""
    other_out_path = ""
    verify_data: dict[str, Any] | None = None
    ref_trim = 0.0
    other_trim = 0.0

    max_iters = max(1, int(auto_correct_iters)) if auto_correct else 1
    for i in range(max_iters):
        ref_out_path, other_out_path, verify_data, ref_trim, other_trim = _run_once(current_offset)
        if not verify or verify_data is None:
            break
        if verify_data.get("ok"):
            break
        if not auto_correct:
            break
        residual = float(verify_data.get("offset_seconds", 0.0))
        next_offset = current_offset - residual
        corrections.append(
            {
                "iteration": i + 1,
                "offset_before": float(current_offset),
                "residual_offset": float(residual),
                "offset_after": float(next_offset),
            }
        )
        current_offset = next_offset

    result: dict[str, Any] = {
        "ref": ref_path,
        "other": other_path,
        "offset_seconds": float(current_offset),
        "ref_trim_seconds": ref_trim,
        "other_trim_seconds": other_trim,
        "ref_out": ref_out_path,
        "other_out": other_out_path,
        "reencode": bool(reencode),
        "fast_seek_only": bool(fast_seek_only),
        "copy_unchanged": bool(copy_unchanged),
        "audio_only": bool(audio_only),
        "audio_window_seconds": audio_window_seconds,
        "auto_correct": bool(auto_correct),
        "auto_correct_iters": int(auto_correct_iters),
        "auto_correct_steps": corrections,
        "cache": {
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "ref_cache": str(cache_ref_wav) if cache_ref_wav is not None else None,
            "other_cache": str(cache_other_wav) if cache_other_wav is not None else None,
        },
    }

    if verify_data is not None:
        result["verify"] = verify_data

    report_path = out_dir / "aligned_report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path
