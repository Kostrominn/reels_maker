from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_analyzer import LlmAnalyzer
from src.models import ClipCandidate, Transcript
from src.utils import load_config, read_json, write_json


def _duration(c: ClipCandidate) -> float:
    return max(0.0, float(c.end) - float(c.start))


def _overlap(a: ClipCandidate, b: ClipCandidate) -> float:
    s = max(float(a.start), float(b.start))
    e = min(float(a.end), float(b.end))
    return max(0.0, e - s)


def _iou(a: ClipCandidate, b: ClipCandidate) -> float:
    inter = _overlap(a, b)
    if inter <= 0:
        return 0.0
    union = _duration(a) + _duration(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class Match:
    g_idx: int
    q_idx: int
    overlap: float
    iou: float
    start_abs_err: float
    end_abs_err: float


def _greedy_match(gpt: list[ClipCandidate], qwen: list[ClipCandidate]) -> list[Match]:
    pairs: list[tuple[float, int, int]] = []
    for i, g in enumerate(gpt):
        for j, q in enumerate(qwen):
            inter = _overlap(g, q)
            if inter <= 0:
                continue
            pairs.append((inter, i, j))
    pairs.sort(reverse=True)

    used_g: set[int] = set()
    used_q: set[int] = set()
    out: list[Match] = []
    for inter, gi, qj in pairs:
        if gi in used_g or qj in used_q:
            continue
        g = gpt[gi]
        q = qwen[qj]
        out.append(
            Match(
                g_idx=gi,
                q_idx=qj,
                overlap=inter,
                iou=_iou(g, q),
                start_abs_err=abs(float(g.start) - float(q.start)),
                end_abs_err=abs(float(g.end) - float(q.end)),
            )
        )
        used_g.add(gi)
        used_q.add(qj)
    return out


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _to_candidates(items: list[dict[str, Any]]) -> list[ClipCandidate]:
    out: list[ClipCandidate] = []
    for it in items:
        try:
            out.append(ClipCandidate.model_validate(it))
        except Exception:
            continue
    return out


def _candidate_stats(items: list[ClipCandidate]) -> dict[str, float]:
    durs = [_duration(c) for c in items]
    return {
        "count": float(len(items)),
        "avg_duration": _avg(durs),
        "min_duration": min(durs) if durs else 0.0,
        "max_duration": max(durs) if durs else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenRouter/Open-source model vs saved GPT baselines.")
    parser.add_argument("--config", type=str, default="config.openrouter.yaml")
    parser.add_argument("--cases", type=str, nargs="+", required=True, help="Transcript stems (without .json)")
    parser.add_argument("--out-json", type=str, default="storage/clips/llm_qwen_eval.json")
    parser.add_argument("--out-tsv", type=str, default="storage/clips/llm_qwen_eval.tsv")
    parser.add_argument("--save-prefix", type=str, default="oss_qwen72b_eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    analyzer = LlmAnalyzer(cfg)

    transcripts_dir = ROOT / "storage" / "transcripts"
    clips_dir = ROOT / "storage" / "clips"

    rows: list[dict[str, Any]] = []

    for stem in args.cases:
        t_path = transcripts_dir / f"{stem}.json"
        gpt_path = clips_dir / f"{stem}_clips.json"
        if not t_path.exists():
            rows.append({"stem": stem, "status": "missing_transcript"})
            continue
        if not gpt_path.exists():
            rows.append({"stem": stem, "status": "missing_gpt_baseline"})
            continue

        gpt_data = read_json(gpt_path)
        if (gpt_data or {}).get("llm_model") != "gpt-5.2":
            rows.append(
                {
                    "stem": stem,
                    "status": "gpt_baseline_not_gpt5_2",
                    "baseline_model": (gpt_data or {}).get("llm_model"),
                }
            )
            continue

        transcript = Transcript.model_validate(read_json(t_path))
        q_res = analyzer.analyze(transcript)

        # Save model output separately; do not overwrite baseline.
        q_path = clips_dir / f"{stem}_clips_{args.save_prefix}.json"
        q_payload = q_res.model_dump(mode="json")
        q_payload["transcript_path"] = str(t_path.relative_to(ROOT))
        write_json(q_path, q_payload)

        gpt_cands = _to_candidates((gpt_data or {}).get("candidates") or [])
        q_cands = list(q_res.candidates or [])
        matches = _greedy_match(gpt_cands, q_cands)

        gpt_total_dur = sum(_duration(c) for c in gpt_cands)
        q_total_dur = sum(_duration(c) for c in q_cands)
        overlap_total = sum(m.overlap for m in matches)

        row = {
            "stem": stem,
            "status": "ok",
            "gpt_model": (gpt_data or {}).get("llm_model"),
            "qwen_model": q_res.llm_model,
            "gpt": _candidate_stats(gpt_cands),
            "qwen": _candidate_stats(q_cands),
            "matches": len(matches),
            "match_rate_gpt": (len(matches) / len(gpt_cands)) if gpt_cands else 0.0,
            "match_rate_qwen": (len(matches) / len(q_cands)) if q_cands else 0.0,
            "avg_iou": _avg([m.iou for m in matches]),
            "avg_start_abs_err_s": _avg([m.start_abs_err for m in matches]),
            "avg_end_abs_err_s": _avg([m.end_abs_err for m in matches]),
            "timing_ok_5s_rate": (
                sum(1 for m in matches if m.start_abs_err <= 5 and m.end_abs_err <= 5) / len(matches)
                if matches
                else 0.0
            ),
            "overlap_total_s": overlap_total,
            "coverage_of_gpt_by_qwen": (overlap_total / gpt_total_dur) if gpt_total_dur > 0 else 0.0,
            "coverage_of_qwen_by_gpt": (overlap_total / q_total_dur) if q_total_dur > 0 else 0.0,
            "qwen_output_file": str(q_path.relative_to(ROOT)),
        }
        rows.append(row)

    summary_ok = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "cases": args.cases,
        "rows": rows,
        "aggregate": {
            "ok_cases": len(summary_ok),
            "total_cases": len(rows),
            "avg_match_rate_gpt": _avg([float(r["match_rate_gpt"]) for r in summary_ok]),
            "avg_match_rate_qwen": _avg([float(r["match_rate_qwen"]) for r in summary_ok]),
            "avg_iou": _avg([float(r["avg_iou"]) for r in summary_ok]),
            "avg_start_abs_err_s": _avg([float(r["avg_start_abs_err_s"]) for r in summary_ok]),
            "avg_end_abs_err_s": _avg([float(r["avg_end_abs_err_s"]) for r in summary_ok]),
            "avg_timing_ok_5s_rate": _avg([float(r["timing_ok_5s_rate"]) for r in summary_ok]),
            "avg_coverage_of_gpt_by_qwen": _avg([float(r["coverage_of_gpt_by_qwen"]) for r in summary_ok]),
            "avg_coverage_of_qwen_by_gpt": _avg([float(r["coverage_of_qwen_by_gpt"]) for r in summary_ok]),
        },
    }

    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    out_tsv = ROOT / args.out_tsv
    header = [
        "stem",
        "status",
        "gpt_count",
        "qwen_count",
        "matches",
        "match_rate_gpt",
        "match_rate_qwen",
        "avg_iou",
        "avg_start_abs_err_s",
        "avg_end_abs_err_s",
        "timing_ok_5s_rate",
        "coverage_of_gpt_by_qwen",
        "coverage_of_qwen_by_gpt",
    ]
    lines = ["\t".join(header)]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(
                "\t".join(
                    [
                        str(r.get("stem", "")),
                        str(r.get("status", "")),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
            )
            continue
        lines.append(
            "\t".join(
                [
                    str(r["stem"]),
                    "ok",
                    str(int(r["gpt"]["count"])),
                    str(int(r["qwen"]["count"])),
                    str(r["matches"]),
                    f"{float(r['match_rate_gpt']):.4f}",
                    f"{float(r['match_rate_qwen']):.4f}",
                    f"{float(r['avg_iou']):.4f}",
                    f"{float(r['avg_start_abs_err_s']):.2f}",
                    f"{float(r['avg_end_abs_err_s']):.2f}",
                    f"{float(r['timing_ok_5s_rate']):.4f}",
                    f"{float(r['coverage_of_gpt_by_qwen']):.4f}",
                    f"{float(r['coverage_of_qwen_by_gpt']):.4f}",
                ]
            )
        )
    out_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_tsv}")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
