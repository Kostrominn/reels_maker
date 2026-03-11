#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path("/Users/mac/Documents/reels_maker")


def run_cmd(
    cmd: list[str],
    *,
    log_file: Path,
    attempts: int = 2,
    pause_s: float = 4.0,
    env: dict[str, str] | None = None,
) -> tuple[bool, int, int, float]:
    t0 = time.time()
    rc = -1
    for attempt in range(1, attempts + 1):
        with log_file.open("a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(cmd)}\n")
            log.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    stdout=log,
                    stderr=log,
                    text=True,
                    check=False,
                    env=env,
                )
                rc = int(proc.returncode)
            except Exception as exc:  # noqa: BLE001
                log.write(f"runner exception attempt={attempt}: {exc}\n")
                rc = -1
            log.write(f"attempt {attempt}/{attempts} rc={rc}\n")
            log.flush()
        if rc == 0:
            return True, rc, attempt, time.time() - t0
        if attempt < attempts:
            time.sleep(pause_s)
    return False, rc, attempts, time.time() - t0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_yadisk_arg(src: str) -> str:
    yd_path = src[len("yadisk:") :].strip()
    if yd_path and not yd_path.startswith("disk:"):
        yd_path = f"disk:{yd_path}" if yd_path.startswith("/") else f"disk:/{yd_path}"
    return yd_path


def _resolve_source_for_probe(src: str) -> str:
    if not src.startswith("yadisk:"):
        return src
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "main.py"),
        "yadisk-url",
        _normalize_yadisk_arg(src),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to resolve Yandex URL for source: {src}. stderr={proc.stderr.strip()[:240]}"
        )
    raw = "".join(line.strip() for line in proc.stdout.splitlines()).replace(" ", "")
    idx = raw.find("http")
    if idx >= 0:
        raw = raw[idx:]
    if not raw.startswith("http"):
        raise RuntimeError(f"Unexpected yadisk-url output for {src}: {proc.stdout[:240]!r}")
    return raw


def _probe_duration_seconds(src: str) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            src,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for source: {src}. stderr={proc.stderr.strip()[:240]}")
    out = (proc.stdout or "").strip()
    try:
        dur = float(out)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse ffprobe duration for source: {src}. out={out!r}") from exc
    if dur <= 0:
        raise RuntimeError(f"Invalid non-positive duration for source: {src}. duration={dur}")
    return float(dur)


def _offset_mapping_coeffs(offsets_path: Path) -> tuple[float, float, dict[str, Any]]:
    data = read_json(offsets_path)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid offsets format: {offsets_path}")
    offset_start = float(data.get("offset_start", 0.0))
    fit = data.get("fit_drift")
    slope = None
    intercept = None
    if isinstance(fit, dict):
        if fit.get("slope_per_sec") is not None:
            slope = float(fit.get("slope_per_sec"))
        if fit.get("intercept_seconds") is not None:
            intercept = float(fit.get("intercept_seconds"))
    if slope is not None and intercept is not None:
        a = 1.0 + slope
        b = intercept
    else:
        a = 1.0
        b = offset_start
    return (
        float(a),
        float(b),
        {
            "offset_start": offset_start,
            "fit_slope_per_sec": slope,
            "fit_intercept_seconds": intercept,
            "mapping_a": float(a),
            "mapping_b": float(b),
        },
    )


def _solve_ref_overlap_window(
    *,
    ref_duration: float,
    other_duration: float,
    map_a: float,
    map_b: float,
    edge_guard_seconds: float = 0.25,
) -> tuple[float, float] | None:
    # other_time(t_ref) = map_a * t_ref + map_b
    lo = float("-inf")
    hi = float("inf")
    eps = 1e-9
    if abs(map_a) <= eps:
        if not (0.0 <= map_b <= other_duration):
            return None
    elif map_a > 0:
        lo = max(lo, -map_b / map_a)
        hi = min(hi, (other_duration - map_b) / map_a)
    else:
        lo = max(lo, (other_duration - map_b) / map_a)
        hi = min(hi, -map_b / map_a)

    lo = max(0.0, float(lo))
    hi = min(float(ref_duration), float(hi))
    if hi <= lo:
        return None

    guard = max(0.0, float(edge_guard_seconds))
    hi_guarded = hi - guard
    if hi_guarded <= lo:
        hi_guarded = hi
    if hi_guarded <= lo:
        return None
    return (round(lo, 3), round(hi_guarded, 3))


def _safe_title(text: str, index: int) -> str:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return f"Фрагмент {index}"
    max_chars = 68
    if len(clean) <= max_chars:
        return clean
    cut = clean[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    cut = cut.rstrip(" ,.;:-")
    return f"{cut}…"


def _retime_to_abs(transcript_src: Path, transcript_abs: Path, start_shift: float) -> None:
    data = read_json(transcript_src)
    for seg in data.get("segments", []):
        if not isinstance(seg, dict):
            continue
        try:
            seg["start"] = round(float(seg.get("start", 0.0)) + float(start_shift), 3)
            seg["end"] = round(float(seg.get("end", 0.0)) + float(start_shift), 3)
        except Exception:
            continue
        words = seg.get("words")
        if isinstance(words, list):
            for w in words:
                if not isinstance(w, dict):
                    continue
                if "start" in w and w["start"] is not None:
                    w["start"] = round(float(w["start"]) + float(start_shift), 3)
                if "end" in w and w["end"] is not None:
                    w["end"] = round(float(w["end"]) + float(start_shift), 3)
    write_json(transcript_abs, data)


def _ranges_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not (float(a["end"]) <= float(b["start"]) + 1.0 or float(a["start"]) >= float(b["end"]) - 1.0)


def _candidate_key(c: dict[str, Any]) -> tuple[float, float]:
    return (round(float(c.get("start", 0.0)), 2), round(float(c.get("end", 0.0)), 2))


def _normalize_candidates(candidates: list[dict[str, Any]], max_clips: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        try:
            s = float(c.get("start"))
            e = float(c.get("end"))
        except Exception:
            continue
        if e <= s + 4.0:
            continue
        c2 = dict(c)
        c2["start"] = round(s, 2)
        c2["end"] = round(e, 2)
        out.append(c2)
        if len(out) >= max_clips:
            break
    return out


def _load_fallback_function() -> Callable[[Path, int], list[dict[str, Any]]]:
    script = ROOT / "scripts" / "run_reels_pair_full.py"
    spec = importlib.util.spec_from_file_location("run_reels_pair_full", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "fallback_candidates_from_transcript", None)
    if not callable(fn):
        raise RuntimeError("fallback_candidates_from_transcript not found")
    return fn


def _select_candidates(
    clips_path: Path,
    transcript_abs_path: Path,
    *,
    max_clips: int,
    fallback_fn: Callable[[Path, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    clips_data = read_json(clips_path)
    base = clips_data.get("candidates") if isinstance(clips_data, dict) else []
    if not isinstance(base, list):
        base = []

    selected = _normalize_candidates([c for c in base if isinstance(c, dict)], max_clips=max_clips)
    if len(selected) >= max_clips:
        for idx, c in enumerate(selected, start=1):
            c["title"] = _safe_title(str(c.get("title") or ""), idx)
        return selected

    fallback = fallback_fn(transcript_abs_path, max(12, max_clips * 2))
    if not isinstance(fallback, list):
        fallback = []
    fallback = _normalize_candidates([c for c in fallback if isinstance(c, dict)], max_clips=max(12, max_clips * 2))

    # First pass: add non-overlapping fallback windows.
    for f in fallback:
        if len(selected) >= max_clips:
            break
        if all(not _ranges_overlap(f, cur) for cur in selected):
            selected.append(dict(f))

    # Second pass: add unique windows if still short.
    if len(selected) < max_clips:
        seen = {_candidate_key(c) for c in selected}
        for f in fallback:
            if len(selected) >= max_clips:
                break
            k = _candidate_key(f)
            if k in seen:
                continue
            selected.append(dict(f))
            seen.add(k)

    for idx, c in enumerate(selected, start=1):
        c["title"] = _safe_title(str(c.get("title") or ""), idx)
    return selected[:max_clips]


def process_batch(
    *,
    ref: str,
    other: str,
    offsets: str,
    config: str,
    start_seconds: float,
    duration_seconds: float,
    output_base: Path,
    run_tag: str,
    max_clips: int,
    fallback_fn: Callable[[Path, int], list[dict[str, Any]]],
    ref_valid_start: float | None = None,
    ref_valid_end: float | None = None,
) -> dict[str, Any]:
    ref_stem = Path(ref.split("/")[-1]).stem
    other_stem = Path(other.split("/")[-1]).stem
    # Resolve short-lived direct URLs once per batch to reduce flaky Yandex API calls during each clip render.
    render_ref = _resolve_source_for_probe(ref) if ref.startswith("yadisk:") else ref
    render_other = _resolve_source_for_probe(other) if other.startswith("yadisk:") else other

    run_dir = output_base / f"run_{run_tag}_batch_s{start_seconds:g}_d{duration_seconds:g}_syncfix2"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run_log.txt"
    records_file = run_dir / "records.ndjson"
    summary_file = run_dir / "summary.json"

    if summary_file.exists():
        try:
            existing = read_json(summary_file)
            if str(existing.get("status")) == "ok":
                print(f"[batch s{start_seconds:g}] skip: summary status=ok")
                return existing
        except Exception:
            pass

    env = os.environ.copy()
    env.setdefault("REELS_MAKER_FFMPEG_TIMEOUT", "7200")
    env_multicam = env.copy()
    env_multicam["REELS_MAKER_FFMPEG_TIMEOUT"] = str(
        os.getenv("REELS_MAKER_FFMPEG_TIMEOUT_MULTICAM", "1800") or "1800"
    )
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("NUMEXPR_NUM_THREADS", "4")

    transcript_raw = ROOT / "storage" / "transcripts" / f"{ref_stem}_s{start_seconds:g}_d{duration_seconds:g}.json"
    transcript_abs = ROOT / "storage" / "transcripts" / f"{ref_stem}_s{start_seconds:g}_d{duration_seconds:g}_abs.json"
    clips_abs = ROOT / "storage" / "clips" / f"{transcript_abs.stem}_clips.json"
    clips_render = ROOT / "storage" / "clips" / f"{transcript_abs.stem}_clips_render.json"

    ref_vf = "eq=gamma_r=1.01:gamma_g=1.00:gamma_b=1.04:saturation=1.00:contrast=0.99"
    other_vf = "eq=gamma_r=1.12:gamma_g=1.02:gamma_b=0.88:saturation=1.07:contrast=0.97:brightness=0.005"
    if ref_stem == "IMG_0003" and other_stem == "IMG_0018":
        other_vf = (
            "crop=w=iw/1.35:h=ih/1.35:x=iw-iw/1.35:y=(ih-ih/1.35)/2,"
            "scale=1080:1920,"
            f"{other_vf}"
        )

    ok_count = 0
    err_count = 0
    skip_count = 0
    t0 = time.time()

    with log_file.open("a", encoding="utf-8") as log, records_file.open("a", encoding="utf-8") as rec:
        log.write(f"# start {datetime.now().isoformat()}\n")
        log.write(f"batch_start={start_seconds:g}\nbatch_duration={duration_seconds:g}\n")
        log.write(f"transcript_raw={transcript_raw}\ntranscript_abs={transcript_abs}\n")
        log.flush()

        if not transcript_raw.exists() or transcript_raw.stat().st_size < 256:
            cmd_t = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "main.py"),
                "--config",
                config,
                "transcribe",
                ref,
                "--start-seconds",
                f"{start_seconds}",
                "--duration-seconds",
                f"{duration_seconds}",
            ]
            ok, rc, attempts, sec = run_cmd(cmd_t, log_file=log_file, attempts=2, pause_s=8.0, env=env)
            rec.write(
                json.dumps(
                    {
                        "type": "transcribe",
                        "status": "ok" if ok else "error",
                        "rc": rc,
                        "attempts": attempts,
                        "seconds": round(sec, 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rec.flush()
            if not ok:
                summary = {
                    "finished_at": datetime.now().isoformat(),
                    "status": "error_transcribe",
                    "seconds": round(time.time() - t0, 3),
                    "run_dir": str(run_dir),
                    "log": str(log_file),
                }
                write_json(summary_file, summary)
                return summary
        else:
            rec.write(json.dumps({"type": "transcribe", "status": "skip_exists"}, ensure_ascii=False) + "\n")
            rec.flush()

        _retime_to_abs(transcript_raw, transcript_abs, start_shift=start_seconds)
        rec.write(json.dumps({"type": "abs_shift", "status": "ok", "path": str(transcript_abs)}, ensure_ascii=False) + "\n")
        rec.flush()

        if not clips_abs.exists() or clips_abs.stat().st_size < 256:
            cmd_a = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "main.py"),
                "--config",
                config,
                "analyze",
                str(transcript_abs.relative_to(ROOT)),
            ]
            ok, rc, attempts, sec = run_cmd(cmd_a, log_file=log_file, attempts=2, pause_s=8.0, env=env)
            rec.write(
                json.dumps(
                    {
                        "type": "analyze",
                        "status": "ok" if ok else "error",
                        "rc": rc,
                        "attempts": attempts,
                        "seconds": round(sec, 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rec.flush()
            if not ok:
                summary = {
                    "finished_at": datetime.now().isoformat(),
                    "status": "error_analyze",
                    "seconds": round(time.time() - t0, 3),
                    "run_dir": str(run_dir),
                    "log": str(log_file),
                }
                write_json(summary_file, summary)
                return summary
        else:
            rec.write(json.dumps({"type": "analyze", "status": "skip_exists"}, ensure_ascii=False) + "\n")
            rec.flush()

        candidates = _select_candidates(
            clips_abs,
            transcript_abs,
            max_clips=max_clips,
            fallback_fn=fallback_fn,
        )
        raw_candidates_count = len(candidates)
        batch_start = float(start_seconds)
        batch_end = float(start_seconds) + float(duration_seconds)
        filtered_candidates: list[dict[str, Any]] = []
        for cand in candidates:
            try:
                c_start = float(cand.get("start"))
                c_end = float(cand.get("end"))
            except Exception:
                continue
            if c_start < batch_start - 1e-6 or c_end > batch_end + 1e-6:
                continue
            if ref_valid_start is not None and c_start < float(ref_valid_start) - 1e-6:
                continue
            if ref_valid_end is not None and c_end > float(ref_valid_end) + 1e-6:
                continue
            filtered_candidates.append(cand)
        candidates = filtered_candidates

        write_json(clips_render, {"transcript_path": str(transcript_abs), "candidates": candidates})
        rec.write(
            json.dumps(
                {
                    "type": "candidates",
                    "status": "ok",
                    "count_raw": raw_candidates_count,
                    "count": len(candidates),
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "ref_valid_start": ref_valid_start,
                    "ref_valid_end": ref_valid_end,
                    "clips_render": str(clips_render),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        rec.flush()

        if not candidates:
            summary = {
                "finished_at": datetime.now().isoformat(),
                "status": "ok_no_candidates",
                "seconds": round(time.time() - t0, 3),
                "run_dir": str(run_dir),
                "log": str(log_file),
                "records": str(records_file),
            }
            write_json(summary_file, summary)
            return summary

        for idx, cand in enumerate(candidates, start=1):
            s = float(cand["start"])
            e = float(cand["end"])
            title = str(cand.get("title") or "").strip() or f"Клип {idx}"
            out_name = (
                f"{ref_stem}_multicam_s{s:.2f}_e{e:.2f}"
                "_final_newdesign_pauseM_speakerA_cut16x2p2_seq_syncfix2.mov"
            )
            out_path = run_dir / out_name

            if out_path.exists() and out_path.stat().st_size > 1024:
                skip_count += 1
                rec.write(
                    json.dumps(
                        {
                            "type": "clip_render",
                            "index": idx,
                            "start": s,
                            "end": e,
                            "title": title,
                            "status": "skip_exists",
                            "output": str(out_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rec.flush()
                print(f"[batch s{start_seconds:g}] [{idx}] skip exists")
                continue

            cmd = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "main.py"),
                "--config",
                config,
                "multicam",
                "--ref",
                render_ref,
                "--other",
                render_other,
                "--offsets",
                offsets,
                "--dominance-mode",
                "speaker",
                "--speaker-map",
                f"andrey={ref_stem}",
                "--speaker-map",
                f"me={other_stem}",
                "--hop-seconds",
                "0.4",
                "--frame-ms",
                "200",
                "--min-shot-seconds",
                "2.0",
                "--audio-crossfade-seconds",
                "0",
                "--hard-gate",
                "--audio-from-sources",
                "--audio-force-camera",
                other_stem,
                "--audio-encoder",
                "alac",
                "--reaction-every-seconds",
                "16.0",
                "--reaction-duration-seconds",
                "2.2",
                "--pause-accel-profile",
                "medium",
                "--subtitle-transcript",
                str(transcript_abs.relative_to(ROOT)),
                "--subtitle-max-chars",
                "22",
                "--subtitle-max-lines",
                "2",
                "--subtitle-preset",
                "readable",
                "--subtitle-align-mode",
                "rendered-audio",
                "--title-style",
                "overlay_center_dark",
                "--title-seconds",
                "1.6",
                "--title-font",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "--title-fontsize",
                "56",
                "--ref-vf",
                ref_vf,
                "--other-vf",
                other_vf,
                "--cache-segments",
                "--cache-padding-seconds",
                "2.0",
                "--fast-seek-only",
                "--start-seconds",
                f"{s}",
                "--end-seconds",
                f"{e}",
                "--title-text",
                title,
                "--output",
                str(out_path.relative_to(ROOT)),
            ]

            print(f"[batch s{start_seconds:g}] [{idx}] render {s:.2f}-{e:.2f}")
            ok, rc, attempts, sec = run_cmd(cmd, log_file=log_file, attempts=3, pause_s=6.0, env=env_multicam)
            rec.write(
                json.dumps(
                    {
                        "type": "clip_render",
                        "index": idx,
                        "start": s,
                        "end": e,
                        "title": title,
                        "status": "ok" if ok else "error",
                        "rc": rc,
                        "attempts": attempts,
                        "seconds": round(sec, 3),
                        "output": str(out_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rec.flush()
            if ok:
                ok_count += 1
            else:
                err_count += 1

    summary = {
        "finished_at": datetime.now().isoformat(),
        "status": "ok" if err_count == 0 else "partial",
        "seconds": round(time.time() - t0, 3),
        "ref": ref,
        "other": other,
        "offsets": offsets,
        "config": config,
        "batch_start": start_seconds,
        "batch_duration": duration_seconds,
        "render_ok": ok_count,
        "render_error": err_count,
        "render_skip_exists": skip_count,
        "run_dir": str(run_dir),
        "log": str(log_file),
        "records": str(records_file),
        "transcript": str(transcript_abs),
        "clips": str(clips_render),
    }
    write_json(summary_file, summary)
    print(f"[batch s{start_seconds:g}] summary {summary['status']} ok={ok_count} err={err_count} skip={skip_count}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous 10-min batch runner for Vova pair")
    parser.add_argument("--ref", default="yadisk:disk:/Интервью/Вова/IMG_0003.mov")
    parser.add_argument("--other", default="yadisk:disk:/Интервью/Вова/IMG_0018.mov")
    parser.add_argument("--offsets", default="storage/sync/offsets_vova.json")
    parser.add_argument("--config", default="config.lowmem.yaml")
    parser.add_argument("--start-seconds", type=float, default=1500.0)
    parser.add_argument("--end-seconds", type=float, default=7520.15)
    parser.add_argument("--chunk-seconds", type=float, default=600.0)
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--run-tag", default="20260220")
    parser.add_argument("--output-base", default="storage/output/final_reels_vova")
    args = parser.parse_args()

    fallback_fn = _load_fallback_function()
    output_base = (ROOT / args.output_base).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    offsets_path = Path(args.offsets)
    if not offsets_path.is_absolute():
        offsets_path = (ROOT / offsets_path).resolve()

    ref_probe_src = _resolve_source_for_probe(args.ref)
    other_probe_src = _resolve_source_for_probe(args.other)
    ref_duration = _probe_duration_seconds(ref_probe_src)
    other_duration = _probe_duration_seconds(other_probe_src)
    map_a, map_b, mapping_meta = _offset_mapping_coeffs(offsets_path)
    overlap_window = _solve_ref_overlap_window(
        ref_duration=ref_duration,
        other_duration=other_duration,
        map_a=map_a,
        map_b=map_b,
        edge_guard_seconds=0.25,
    )
    if overlap_window is None:
        ref_valid_start = None
        ref_valid_end = None
        effective_start = float(args.start_seconds)
        effective_end = float(args.start_seconds)
        print("No valid ref/other overlap for current offsets and source durations.")
    else:
        ref_valid_start, ref_valid_end = overlap_window
        effective_start = max(float(args.start_seconds), float(ref_valid_start))
        effective_end = min(float(args.end_seconds), float(ref_valid_end))
        print(
            "effective reference window for this pair: "
            f"{ref_valid_start:.2f}s..{ref_valid_end:.2f}s "
            f"(requested {float(args.start_seconds):.2f}s..{float(args.end_seconds):.2f}s)"
        )

    t = float(effective_start)
    end = float(effective_end)
    chunk = float(args.chunk_seconds)

    if chunk <= 0:
        raise SystemExit("chunk-seconds must be > 0")

    all_summaries: list[dict[str, Any]] = []
    while t < end - 1e-6:
        dur = min(chunk, end - t)
        if dur <= 8.0:
            break
        print(f"=== batch start={t:g} dur={dur:g} ===")
        summary = process_batch(
            ref=args.ref,
            other=args.other,
            offsets=args.offsets,
            config=args.config,
            start_seconds=round(t, 2),
            duration_seconds=round(dur, 2),
            output_base=output_base,
            run_tag=args.run_tag,
            max_clips=int(args.max_clips),
            fallback_fn=fallback_fn,
            ref_valid_start=ref_valid_start,
            ref_valid_end=ref_valid_end,
        )
        all_summaries.append(summary)
        t += chunk

    master = output_base / f"autobatch_{args.run_tag}_from_{args.start_seconds:g}.json"
    write_json(
        master,
        {
            "finished_at": datetime.now().isoformat(),
            "ref": args.ref,
            "other": args.other,
            "offsets": args.offsets,
            "config": args.config,
            "start_seconds": args.start_seconds,
            "end_seconds": args.end_seconds,
            "effective_start_seconds": effective_start,
            "effective_end_seconds": effective_end,
            "chunk_seconds": args.chunk_seconds,
            "ref_duration_seconds": ref_duration,
            "other_duration_seconds": other_duration,
            "offset_mapping": mapping_meta,
            "ref_overlap_window": (
                {"start": ref_valid_start, "end": ref_valid_end}
                if overlap_window is not None
                else None
            ),
            "batches": all_summaries,
        },
    )
    print(f"saved master summary -> {master}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
