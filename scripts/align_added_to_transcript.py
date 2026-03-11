#!/usr/bin/env python3
"""Align edited script blocks to transcript timeline and emit semantic segments JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import Transcript
from src.utils import read_json


MARKER_RE = re.compile(r"\b(вырез|удали|убер|стоп|перезапис)\w*\b", re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")


@dataclass
class Block:
    label: str
    text: str
    norm: str
    tokens: set[str]
    words: int


@dataclass
class Seg:
    start: float
    end: float
    text: str
    norm: str
    tokens: set[str]
    words: int


def normalize_text(text: str) -> str:
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[^a-zа-я0-9\s]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> list[str]:
    return [w.lower().replace("ё", "е") for w in WORD_RE.findall(text)]


def parse_added_blocks(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    parts = re.split(r"\n\*\*Текст:\*\*\n\n", raw)
    blocks: list[str] = []
    for p in parts[1:]:
        # stop before next marker if any
        chunk = re.split(r"\n\*\*Текст:\*\*\n\n", p, maxsplit=1)[0]
        txt = " ".join(chunk.strip().split())
        if txt:
            blocks.append(txt)
    return blocks


def parse_labels(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    labels = [m.group(1).strip() for m in re.finditer(r"^###\s+\d+\.\s+([^\n]+)$", raw, flags=re.M)]
    return labels


def build_blocks(labels: list[str], texts: list[str]) -> list[Block]:
    n = min(len(labels), len(texts))
    out: list[Block] = []
    for i in range(n):
        label = labels[i]
        text = texts[i]
        norm = normalize_text(text)
        toks = set(tokenize(norm))
        out.append(Block(label=label, text=text, norm=norm, tokens=toks, words=len(norm.split())))
    return out


def build_segments(transcript: Transcript) -> list[Seg]:
    out: list[Seg] = []
    for s in transcript.segments:
        text = " ".join(str(s.text).split())
        if not text:
            continue
        # drop short explicit "marker" phrases like "вырезать", "стоп"
        if MARKER_RE.search(text) and len(text.split()) <= 5:
            continue
        norm = normalize_text(text)
        toks = set(tokenize(norm))
        out.append(Seg(start=float(s.start), end=float(s.end), text=text, norm=norm, tokens=toks, words=len(norm.split())))
    return out


def f1_token(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return (2.0 * inter) / (len(a) + len(b))


def similarity(block: Block, window_text: str, window_tokens: set[str], has_marker: bool) -> float:
    tok_score = f1_token(block.tokens, window_tokens)
    # Ratio on clipped strings to keep runtime bounded.
    seq_score = SequenceMatcher(None, block.norm[:3000], window_text[:3000]).ratio()
    score = 0.62 * tok_score + 0.38 * seq_score
    if has_marker:
        score -= 0.08
    return max(0.0, score)


def align_blocks(blocks: list[Block], segs: list[Seg]) -> list[dict[str, float | str]]:
    if not blocks or not segs:
        return []

    avg_words_per_seg = sum(s.words for s in segs) / max(1, len(segs))
    ptr = 0
    n = len(segs)
    out: list[dict[str, float | str]] = []

    for bi, block in enumerate(blocks, start=1):
        expected = max(2, int(round(block.words / max(1.0, avg_words_per_seg))))
        max_span = min(90, expected * 4 + 8)
        min_span = max(1, expected // 3)

        best: tuple[float, int, int] | None = None
        # Keep enough room for remaining blocks.
        max_start = max(ptr, n - (len(blocks) - bi + 1))
        for i in range(ptr, max_start + 1):
            merged_tokens: set[str] = set()
            texts: list[str] = []
            has_marker = False
            for j in range(i, min(n, i + max_span)):
                s = segs[j]
                merged_tokens.update(s.tokens)
                texts.append(s.norm)
                if MARKER_RE.search(s.text):
                    has_marker = True
                k = j - i + 1
                if k < min_span:
                    continue
                duration = segs[j].end - segs[i].start
                if duration < 6 or duration > 140:
                    continue
                window_text = " ".join(texts)
                score = similarity(block, window_text, merged_tokens, has_marker)
                if best is None or score > best[0]:
                    best = (score, i, j)

        if best is None:
            # Fallback: take next few segments.
            i = min(ptr, n - 1)
            j = min(n - 1, i + max(2, expected))
            score = 0.0
        else:
            score, i, j = best

        out.append(
            {
                "label": re.sub(r"[^a-z0-9_]+", "_", block.label.lower()).strip("_") or f"block_{bi:02d}",
                "start": round(segs[i].start, 2),
                "end": round(segs[j].end, 2),
                "score": round(float(score), 4),
                "block_idx": bi,
                "src_label": block.label,
            }
        )
        ptr = min(j + 1, n - 1)

    # Ensure monotonic non-overlap with tiny safety gap.
    cleaned: list[dict[str, float | str]] = []
    prev_end = -1.0
    for row in out:
        s = float(row["start"])
        e = float(row["end"])
        if s < prev_end:
            s = prev_end
        if e <= s:
            e = s + 3.0
        row["start"] = round(s, 2)
        row["end"] = round(e, 2)
        cleaned.append(row)
        prev_end = e
    return cleaned


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Align edited script blocks to transcript")
    p.add_argument("--transcript", default="/Users/mac/Documents/reels_maker/storage/transcripts/IMG_5311.json")
    p.add_argument("--added-md", default="/Users/mac/Documents/reels_maker/highlights/reels_transcripts_for_rerecord-added.md")
    p.add_argument("--base-md", default="/Users/mac/Documents/reels_maker/highlights/reels_transcripts_for_rerecord.md")
    p.add_argument("--video-name", default="IMG_5311.MOV")
    p.add_argument("--segments-out", default="/Users/mac/Documents/reels_maker/highlights/semantic_segments_img5311_auto.json")
    p.add_argument("--report-out", default="/Users/mac/Documents/reels_maker/highlights/semantic_segments_img5311_auto_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    transcript = Transcript.model_validate(read_json(args.transcript))
    texts = parse_added_blocks(Path(args.added_md))
    labels = parse_labels(Path(args.base_md))
    blocks = build_blocks(labels, texts)
    segs = build_segments(transcript)
    aligned = align_blocks(blocks, segs)

    semantic = {
        args.video_name: [
            {"label": str(r["label"]), "start": float(r["start"]), "end": float(r["end"])}
            for r in aligned
        ]
    }

    Path(args.segments_out).write_text(json.dumps(semantic, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.report_out).write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.segments_out)
    print(args.report_out)
    print(f"blocks: {len(aligned)}")


if __name__ == "__main__":
    main()
