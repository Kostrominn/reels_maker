#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path("/Users/mac/Documents/reels_maker")


def run_cmd(
    cmd: list[str],
    *,
    log_file: Path,
    attempts: int = 2,
    pause_s: float = 6.0,
) -> tuple[bool, int, int]:
    env = os.environ.copy()
    last_rc = -1
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
                last_rc = int(proc.returncode)
            except Exception as exc:  # noqa: BLE001
                log.write(f"runner exception attempt={attempt}: {exc}\n")
                log.flush()
                last_rc = -1
        if last_rc == 0:
            return True, last_rc, attempt
        with log_file.open("a", encoding="utf-8") as log:
            log.write(f"attempt {attempt}/{attempts} failed rc={last_rc}\n")
            log.flush()
        if attempt < attempts:
            time.sleep(pause_s)
    return False, last_rc, attempts


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def video_stem(src: str) -> str:
    if src.startswith("yadisk:"):
        p = src[len("yadisk:") :]
        if p.startswith("disk:"):
            p = p[len("disk:") :]
        return Path(p).stem
    return Path(src).stem


def _normalize_yadisk_arg(src: str) -> str:
    yd_path = src[len("yadisk:") :].strip()
    if yd_path and not yd_path.startswith("disk:"):
        yd_path = f"disk:{yd_path}" if yd_path.startswith("/") else f"disk:/{yd_path}"
    return yd_path


def _resolve_source_for_render(src: str) -> str:
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
        raise RuntimeError(f"Failed to resolve Yandex URL for {src}: {proc.stderr.strip()[:240]}")
    raw = "".join(line.strip() for line in proc.stdout.splitlines()).replace(" ", "")
    idx = raw.find("http")
    if idx >= 0:
        raw = raw[idx:]
    if not raw.startswith("http"):
        raise RuntimeError(f"Unexpected yadisk-url output for {src}: {proc.stdout[:240]!r}")
    return raw


def _safe_title_from_text(text: str, index: int) -> str:
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


def fallback_candidates_from_transcript(transcript_path: Path, max_clips: int) -> list[dict[str, Any]]:
    try:
        data = read_json(transcript_path)
    except Exception:  # noqa: BLE001
        return []

    raw_segments = data.get("segments") if isinstance(data, dict) else None
    if not isinstance(raw_segments, list):
        return []

    segs: list[dict[str, Any]] = []
    for s in raw_segments:
        if not isinstance(s, dict):
            continue
        try:
            st = float(s.get("start"))
            en = float(s.get("end"))
        except Exception:
            continue
        txt = str(s.get("text") or "").strip()
        if not txt or en <= st:
            continue
        segs.append({"start": st, "end": en, "text": txt})

    if not segs:
        return []

    # Split into contiguous speech groups by pauses.
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [segs[0]]
    for seg in segs[1:]:
        gap = float(seg["start"]) - float(current[-1]["end"])
        if gap <= 2.4:
            current.append(seg)
        else:
            groups.append(current)
            current = [seg]
    groups.append(current)

    windows: list[dict[str, Any]] = []
    min_len = 16.0
    target_len = 40.0
    max_len = 70.0

    for g in groups:
        if not g:
            continue
        i = 0
        n = len(g)
        while i < n:
            st = float(g[i]["start"])
            j = i
            while j + 1 < n and (float(g[j + 1]["end"]) - st) <= target_len:
                j += 1
            # if too short, try extending up to hard max
            while j + 1 < n and (float(g[j]["end"]) - st) < min_len and (float(g[j + 1]["end"]) - st) <= max_len:
                j += 1
            en = float(g[j]["end"])
            dur = en - st
            if dur >= min_len:
                txt = " ".join(str(x["text"]) for x in g[i : j + 1])
                windows.append(
                    {
                        "start": st,
                        "end": en,
                        "duration": dur,
                        "text": txt,
                        "score": abs(target_len - dur),
                    }
                )
            i = j + 1

    if not windows:
        return []

    # Prefer windows close to target duration; then enforce non-overlap.
    windows.sort(key=lambda w: (w["score"], w["start"]))
    chosen: list[dict[str, Any]] = []
    for w in windows:
        overlap = False
        for c in chosen:
            if not (w["end"] <= c["start"] + 1.0 or w["start"] >= c["end"] - 1.0):
                overlap = True
                break
        if overlap:
            continue
        chosen.append(w)
        if len(chosen) >= max(1, max_clips):
            break

    out: list[dict[str, Any]] = []
    for idx, w in enumerate(sorted(chosen, key=lambda x: x["start"]), start=1):
        out.append(
            {
                "start": round(float(w["start"]), 2),
                "end": round(float(w["end"]), 2),
                "title": _safe_title_from_text(str(w["text"]), idx),
                "reason": "fallback_from_transcript_windows",
                "score": round(1.0 / (1.0 + float(w["score"])), 3),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-file reels runner for a camera pair.")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--other", required=True)
    parser.add_argument("--offsets", required=True, help="Path relative to repo root")
    parser.add_argument("--output-dir", required=True, help="Path relative to repo root")
    parser.add_argument("--config", default="config.lowmem.yaml")
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--speaker-map-andrey", default="andrey")
    parser.add_argument("--speaker-map-me", default="me")
    parser.add_argument(
        "--subtitle-align-mode",
        choices=["transcript", "rendered-audio"],
        default="rendered-audio",
    )
    args = parser.parse_args()

    # Remote Yandex reads can be slow; keep ffmpeg timeout high to avoid reset loops.
    os.environ.setdefault("REELS_MAKER_FFMPEG_TIMEOUT", "3600")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run_log.txt"
    records_file = out_dir / "records.ndjson"
    summary_file = out_dir / "summary.json"

    ref_stem = video_stem(args.ref)
    other_stem = video_stem(args.other)
    render_ref = _resolve_source_for_render(args.ref)
    render_other = _resolve_source_for_render(args.other)
    transcript_path = ROOT / "storage" / "transcripts" / f"{ref_stem}.json"
    clips_path = ROOT / "storage" / "clips" / f"{ref_stem}_clips.json"

    t0 = time.time()
    ok_count = 0
    err_count = 0
    skip_count = 0

    with log_file.open("a", encoding="utf-8") as log, records_file.open("a", encoding="utf-8") as rec:
        log.write(f"# start {datetime.now().isoformat()}\n")
        log.write(
            f"ref={args.ref}\nother={args.other}\noffsets={args.offsets}\nconfig={args.config}\n"
            f"output_dir={out_dir}\n"
        )
        log.flush()

        if not transcript_path.exists() or transcript_path.stat().st_size < 256:
            cmd = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "main.py"),
                "--config",
                args.config,
                "transcribe",
                args.ref,
            ]
            ok, rc, attempts = run_cmd(cmd, log_file=log_file, attempts=2, pause_s=8.0)
            rec.write(
                json.dumps(
                    {"type": "transcribe", "status": "ok" if ok else "error", "rc": rc, "attempts": attempts},
                    ensure_ascii=False,
                )
                + "\n"
            )
            rec.flush()
            if not ok:
                summary_file.write_text(
                    json.dumps(
                        {
                            "finished_at": datetime.now().isoformat(),
                            "status": "error_transcribe",
                            "seconds": round(time.time() - t0, 3),
                            "run_dir": str(out_dir),
                            "log": str(log_file),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return 1
        else:
            rec.write(json.dumps({"type": "transcribe", "status": "skip_exists"}, ensure_ascii=False) + "\n")
            rec.flush()

        if not clips_path.exists() or clips_path.stat().st_size < 256:
            cmd = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "main.py"),
                "--config",
                args.config,
                "analyze",
                str(transcript_path.relative_to(ROOT)),
            ]
            ok, rc, attempts = run_cmd(cmd, log_file=log_file, attempts=2, pause_s=8.0)
            rec.write(
                json.dumps(
                    {"type": "analyze", "status": "ok" if ok else "error", "rc": rc, "attempts": attempts},
                    ensure_ascii=False,
                )
                + "\n"
            )
            rec.flush()
            if not ok:
                summary_file.write_text(
                    json.dumps(
                        {
                            "finished_at": datetime.now().isoformat(),
                            "status": "error_analyze",
                            "seconds": round(time.time() - t0, 3),
                            "run_dir": str(out_dir),
                            "log": str(log_file),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return 1
        else:
            rec.write(json.dumps({"type": "analyze", "status": "skip_exists"}, ensure_ascii=False) + "\n")
            rec.flush()

        clips = read_json(clips_path)
        candidates = clips.get("candidates") if isinstance(clips, dict) else None
        if not isinstance(candidates, list):
            candidates = []
        if not candidates:
            fallback = fallback_candidates_from_transcript(transcript_path, int(args.max_clips))
            if fallback:
                candidates = fallback
                rec.write(
                    json.dumps(
                        {
                            "type": "fallback_candidates",
                            "status": "ok",
                            "count": len(fallback),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rec.flush()
                with log_file.open("a", encoding="utf-8") as log2:
                    log2.write(f"LLM returned 0 candidates, using fallback={len(fallback)}\n")
            else:
                summary_file.write_text(
                    json.dumps(
                        {
                            "finished_at": datetime.now().isoformat(),
                            "status": "ok_no_candidates",
                            "seconds": round(time.time() - t0, 3),
                            "run_dir": str(out_dir),
                            "log": str(log_file),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return 0

        title_font = "/System/Library/Fonts/Supplemental/Arial.ttf"
        ref_vf = "eq=gamma_r=1.01:gamma_g=1.00:gamma_b=1.04:saturation=1.00:contrast=0.99"
        other_vf = "eq=gamma_r=1.12:gamma_g=1.02:gamma_b=0.88:saturation=1.07:contrast=0.97:brightness=0.005"
        # Vova pair: camera IMG_0018 is right-shifted; apply soft reframe for more balanced composition.
        if ref_stem == "IMG_0003" and other_stem == "IMG_0018":
            other_vf = (
                "crop=w=iw/1.35:h=ih/1.35:x=iw-iw/1.35:y=(ih-ih/1.35)/2,"
                "scale=1080:1920,"
                f"{other_vf}"
            )
        speaker_map_andrey = f"{args.speaker_map_andrey}={ref_stem}"
        speaker_map_me = f"{args.speaker_map_me}={other_stem}"

        for idx, cand in enumerate(candidates[: int(args.max_clips)], start=1):
            if not isinstance(cand, dict):
                continue
            try:
                c_start = float(cand.get("start"))
                c_end = float(cand.get("end"))
            except Exception:
                continue
            if c_end <= c_start + 4.0:
                continue
            title = str(cand.get("title") or "").strip() or f"Клип {idx}"
            out_name = (
                f"{ref_stem}_multicam_s{c_start:.2f}_e{c_end:.2f}"
                "_final_newdesign_pauseM_speakerA_cut16x2p2_seq.mov"
            )
            out_path = out_dir / out_name
            if out_path.exists() and out_path.stat().st_size > 1024:
                skip_count += 1
                rec.write(
                    json.dumps(
                        {
                            "type": "clip_render",
                            "index": idx,
                            "start": c_start,
                            "end": c_end,
                            "title": title,
                            "status": "skip_exists",
                            "output": str(out_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rec.flush()
                continue

            cmd = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "main.py"),
                "--config",
                args.config,
                "multicam",
                "--ref",
                render_ref,
                "--other",
                render_other,
                "--offsets",
                args.offsets,
                "--dominance-mode",
                "speaker",
                "--speaker-map",
                speaker_map_andrey,
                "--speaker-map",
                speaker_map_me,
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
                str(transcript_path.relative_to(ROOT)),
                "--subtitle-max-chars",
                "22",
                "--subtitle-max-lines",
                "2",
                "--subtitle-preset",
                "readable",
                "--subtitle-align-mode",
                args.subtitle_align_mode,
                "--title-style",
                "overlay_center_dark",
                "--title-seconds",
                "1.6",
                "--title-font",
                title_font,
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
                f"{c_start}",
                "--end-seconds",
                f"{c_end}",
                "--title-text",
                title,
                "--output",
                str(out_path.relative_to(ROOT)),
            ]

            t_clip = time.time()
            ok, rc, attempts = run_cmd(cmd, log_file=log_file, attempts=3, pause_s=6.0)
            rec.write(
                json.dumps(
                    {
                        "type": "clip_render",
                        "index": idx,
                        "start": c_start,
                        "end": c_end,
                        "title": title,
                        "status": "ok" if ok else "error",
                        "rc": rc,
                        "attempts": attempts,
                        "seconds": round(time.time() - t_clip, 3),
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

    summary_file.write_text(
        json.dumps(
            {
                "finished_at": datetime.now().isoformat(),
                "status": "ok" if err_count == 0 else "partial",
                "seconds": round(time.time() - t0, 3),
                "ref": args.ref,
                "other": args.other,
                "offsets": args.offsets,
                "config": args.config,
                "render_ok": ok_count,
                "render_error": err_count,
                "render_skip_exists": skip_count,
                "run_dir": str(out_dir),
                "log": str(log_file),
                "records": str(records_file),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
