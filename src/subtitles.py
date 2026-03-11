from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from .models import Transcript, TranscriptSegment
from .utils import read_json


@dataclass(frozen=True)
class SubtitleStyle:
    font_name: str = "Arial"
    font_size: int = 32
    primary_color: str = "&H00F8F8F8"
    outline_color: str = "&H00000000"
    back_color: str = "&H00000000"
    bold: int = 0
    italic: int = 0
    outline: int = 2
    shadow: int = 1
    border_style: int = 1
    alignment: int = 2  # bottom-center
    margin_l: int = 50
    margin_r: int = 50
    margin_v: int = 50


def subtitle_style_from_preset(name: str | None) -> SubtitleStyle:
    key = (name or "current").strip().lower()
    if key == "current":
        return SubtitleStyle()
    if key == "readable":
        return SubtitleStyle(
            font_name="Arial",
            font_size=34,
            bold=1,
            outline=3,
            shadow=0,
            margin_v=76,
        )
    if key == "safe":
        return SubtitleStyle(
            font_name="Arial",
            font_size=34,
            bold=1,
            outline=3,
            shadow=0,
            margin_v=110,
        )
    raise ValueError(f"Unknown subtitle preset: {name}")


def _ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _sanitize_ass(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "[")
    text = text.replace("}", "]")
    return text


def _wrap_lines(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [text]
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return lines


def _split_pages(lines: list[str], max_lines: int) -> list[list[str]]:
    if not lines:
        return []
    if max_lines <= 0:
        return [lines]
    pages: list[list[str]] = []
    for i in range(0, len(lines), max_lines):
        pages.append(lines[i : i + max_lines])
    return pages


def _merge_segments(
    segments: list[TranscriptSegment],
    *,
    merge_gap: float,
    max_duration: float = 4.0,
) -> list[TranscriptSegment]:
    if not segments:
        return []
    merged: list[TranscriptSegment] = []
    cur = segments[0]
    for seg in segments[1:]:
        gap = seg.start - cur.end
        if gap <= merge_gap and (seg.end - cur.start) <= max_duration:
            cur = TranscriptSegment(
                start=cur.start,
                end=seg.end,
                text=f"{cur.text} {seg.text}".strip(),
                words=None,
            )
        else:
            merged.append(cur)
            cur = seg
    merged.append(cur)
    return merged


def _normalize_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for start, end in sorted(intervals, key=lambda x: x[0]):
        s = float(start)
        e = float(end)
        if e <= s:
            continue
        if not normalized:
            normalized.append((s, e))
            continue
        prev_s, prev_e = normalized[-1]
        if s <= prev_e + 1e-6:
            normalized[-1] = (prev_s, max(prev_e, e))
        else:
            normalized.append((s, e))
    return normalized


def _subtract_interval(
    start: float,
    end: float,
    cuts: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if end <= start:
        return []
    if not cuts:
        return [(start, end)]
    out: list[tuple[float, float]] = []
    cursor = float(start)
    for cut_start, cut_end in cuts:
        if cut_end <= cursor:
            continue
        if cut_start >= end:
            break
        if cut_start > cursor:
            out.append((cursor, min(end, cut_start)))
        cursor = max(cursor, cut_end)
        if cursor >= end:
            break
    if cursor < end:
        out.append((cursor, end))
    return out


def _removed_before(time_value: float, cuts: list[tuple[float, float]]) -> float:
    removed = 0.0
    t = float(time_value)
    for cut_start, cut_end in cuts:
        if cut_end <= t:
            removed += cut_end - cut_start
            continue
        if cut_start < t:
            removed += t - cut_start
        break
    return removed


def _slice_segment_text_for_window(
    seg: TranscriptSegment,
    *,
    clip_start: float,
    clip_end: float,
) -> str:
    raw = " ".join(str(seg.text).split())
    if not raw:
        return ""

    # Prefer word-level alignment when available.
    if seg.words:
        parts: list[str] = []
        for w in seg.words:
            ws = float(w.start)
            we = float(w.end)
            if we <= clip_start or ws >= clip_end:
                continue
            token = " ".join(str(w.word).split())
            if token:
                parts.append(token)
        if parts:
            return " ".join(parts).strip()

    # Fallback for ASR segments without word timestamps:
    # if we cut only a part of a long segment, trim text proportionally.
    seg_start = float(seg.start)
    seg_end = float(seg.end)
    seg_dur = max(1e-6, seg_end - seg_start)
    start_ratio = max(0.0, min(1.0, (clip_start - seg_start) / seg_dur))
    end_ratio = max(0.0, min(1.0, (clip_end - seg_start) / seg_dur))
    if end_ratio <= start_ratio:
        return raw
    if start_ratio <= 0.001 and end_ratio >= 0.999:
        return raw

    words = raw.split()
    if len(words) <= 3:
        return raw

    n = len(words)
    i0 = max(0, min(n - 1, int(math.floor(start_ratio * n))))
    i1 = max(i0 + 1, min(n, int(math.ceil(end_ratio * n))))
    sliced = " ".join(words[i0:i1]).strip()
    return sliced or raw


def _words_per_second(text: str, duration_seconds: float) -> float:
    dur = max(1e-6, float(duration_seconds))
    words = [w for w in str(text).split() if w]
    if not words:
        return 0.0
    return float(len(words)) / dur


def _merge_high_density_segments(
    segments: list[TranscriptSegment],
    *,
    max_words_per_second: float,
    max_duration: float,
    max_gap: float,
) -> list[TranscriptSegment]:
    if not segments:
        return []
    if max_words_per_second <= 0:
        return segments

    merged: list[TranscriptSegment] = []
    i = 0
    while i < len(segments):
        cur = segments[i]
        cur_dur = max(1e-6, float(cur.end) - float(cur.start))
        cur_wps = _words_per_second(cur.text, cur_dur)
        merge_max_duration = float(max_duration)
        if cur_wps > max_words_per_second * 1.6:
            merge_max_duration = max(merge_max_duration, 8.0)

        while i + 1 < len(segments):
            cur_dur = max(1e-6, float(cur.end) - float(cur.start))
            cur_wps = _words_per_second(cur.text, cur_dur)
            if cur_wps <= max_words_per_second:
                break
            nxt = segments[i + 1]
            gap = float(nxt.start) - float(cur.end)
            new_dur = float(nxt.end) - float(cur.start)
            if gap > float(max_gap) or new_dur > merge_max_duration:
                break
            cur = TranscriptSegment(
                start=cur.start,
                end=nxt.end,
                text=f"{cur.text} {nxt.text}".strip(),
                words=None,
            )
            i += 1

        cur_dur = max(1e-6, float(cur.end) - float(cur.start))
        cur_wps = _words_per_second(cur.text, cur_dur)
        if cur_wps > max_words_per_second and merged:
            prev = merged[-1]
            gap_prev = float(cur.start) - float(prev.end)
            merged_dur = float(cur.end) - float(prev.start)
            if gap_prev <= float(max_gap) and merged_dur <= merge_max_duration:
                merged[-1] = TranscriptSegment(
                    start=prev.start,
                    end=cur.end,
                    text=f"{prev.text} {cur.text}".strip(),
                    words=None,
                )
                i += 1
                continue

        merged.append(cur)
        i += 1
    return merged


def _page_word_count(lines: list[str]) -> int:
    return sum(1 for w in " ".join(lines).split() if w)


def _allocate_page_durations(
    pages: list[list[str]],
    *,
    total_duration: float,
    min_duration: float,
) -> list[float]:
    n = len(pages)
    if n <= 0:
        return []
    if n == 1:
        return [float(total_duration)]

    total = max(0.0, float(total_duration))
    floor = max(0.0, float(min_duration))
    if total <= 1e-6:
        return [0.0] * n
    if total <= floor * n:
        # Not enough timeline to satisfy floor for each page.
        return [total / float(n)] * n

    weights = [max(1, _page_word_count(p)) for p in pages]
    durations = [0.0] * n
    fixed: set[int] = set()
    remaining_total = total

    while True:
        free = [idx for idx in range(n) if idx not in fixed]
        if not free:
            break
        free_weight = float(sum(weights[idx] for idx in free))
        if free_weight <= 0:
            for idx in free:
                durations[idx] = remaining_total / float(len(free))
            break
        for idx in free:
            durations[idx] = remaining_total * (float(weights[idx]) / free_weight)

        changed = False
        for idx in free:
            if durations[idx] < floor:
                durations[idx] = floor
                fixed.add(idx)
                remaining_total -= floor
                changed = True
        if remaining_total <= 1e-6:
            break
        if not changed:
            break

    if remaining_total > 1e-6:
        free = [idx for idx in range(n) if idx not in fixed]
        if free:
            free_weight = float(sum(weights[idx] for idx in free))
            if free_weight <= 0:
                add = remaining_total / float(len(free))
                for idx in free:
                    durations[idx] += add
            else:
                for idx in free:
                    durations[idx] = remaining_total * (float(weights[idx]) / free_weight)
        else:
            durations[-1] += remaining_total

    # Small rounding drift fix.
    drift = total - sum(durations)
    durations[-1] += drift
    return durations


def build_ass_subtitles(
    transcript_path: str | Path,
    *,
    start_seconds: float,
    end_seconds: float,
    out_path: str | Path,
    max_chars: int = 22,
    max_lines: int = 2,
    merge_gap: float = 0.08,
    max_duration: float = 2.0,
    min_duration: float = 0.18,
    min_page_duration: float = 2.0,
    max_words_per_second: float = 5.0,
    density_merge_max_duration: float = 6.0,
    density_merge_max_gap: float = 0.30,
    time_offset: float = 0.0,
    cut_intervals: list[tuple[float, float]] | None = None,
    style: SubtitleStyle | None = None,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
) -> Path:
    data = read_json(transcript_path)
    transcript = Transcript.model_validate(data)
    style = style or SubtitleStyle()
    cuts_abs = _normalize_intervals(
        [
            (max(float(start_seconds), float(s)), min(float(end_seconds), float(e)))
            for (s, e) in (cut_intervals or [])
            if float(e) > float(s)
        ]
    )

    segs: list[TranscriptSegment] = []
    for s in transcript.segments:
        if s.end <= start_seconds or s.start >= end_seconds:
            continue
        seg_start = max(start_seconds, float(s.start))
        seg_end = min(end_seconds, float(s.end))
        if seg_end - seg_start <= 0:
            continue
        windows = _subtract_interval(seg_start, seg_end, cuts_abs)
        for piece_start, piece_end in windows:
            if piece_end - piece_start <= 0:
                continue
            clipped_text = _slice_segment_text_for_window(
                s,
                clip_start=piece_start,
                clip_end=piece_end,
            )
            if not clipped_text:
                continue
            segs.append(
                TranscriptSegment(
                    start=piece_start,
                    end=piece_end,
                    text=clipped_text,
                    words=None,
                )
            )

    segs = _merge_segments(segs, merge_gap=merge_gap, max_duration=max_duration)
    segs = _merge_high_density_segments(
        segs,
        max_words_per_second=max_words_per_second,
        max_duration=density_merge_max_duration,
        max_gap=max(merge_gap, density_merge_max_gap),
    )

    lines: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
        "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            "Style: Default,"
            f"{style.font_name},{style.font_size},"
            f"{style.primary_color},&H00000000,{style.outline_color},{style.back_color},"
            f"{style.bold},{style.italic},0,0,100,100,0,0,{style.border_style},{style.outline},{style.shadow},"
            f"{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for seg in segs:
        local_start = (
            float(seg.start - start_seconds)
            - _removed_before(float(seg.start), cuts_abs)
            + float(time_offset)
        )
        local_end = (
            float(seg.end - start_seconds)
            - _removed_before(float(seg.end), cuts_abs)
            + float(time_offset)
        )
        seg_dur = local_end - local_start
        if seg_dur < min_duration:
            continue
        raw_text = _sanitize_ass(seg.text)
        effective_max_chars = max(8, int(max_chars))
        wrapped_lines = _wrap_lines(raw_text, max_chars=effective_max_chars)
        pages = _split_pages(wrapped_lines, max_lines=max_lines)
        if not pages:
            continue
        # If page switching is too fast, first widen wrapping while preserving
        # max_lines, then merge pages conservatively.
        page_min = max(float(min_duration), float(min_page_duration))
        max_chars_cap = max(effective_max_chars, 36)
        while len(pages) > 1 and seg_dur / len(pages) < page_min and effective_max_chars < max_chars_cap:
            effective_max_chars += 2
            wrapped_lines = _wrap_lines(raw_text, max_chars=effective_max_chars)
            pages = _split_pages(wrapped_lines, max_lines=max_lines)

        while len(pages) > 1 and seg_dur / len(pages) < page_min:
            merged_text = " ".join(pages[-2] + pages[-1]).strip()
            merged_lines = _wrap_lines(merged_text, max_chars=effective_max_chars)
            if max_lines > 0 and len(merged_lines) > max_lines:
                local_chars = effective_max_chars
                while len(merged_lines) > max_lines and local_chars < 56:
                    local_chars += 2
                    merged_lines = _wrap_lines(merged_text, max_chars=local_chars)
                if len(merged_lines) > max_lines:
                    head = merged_lines[: max_lines - 1]
                    tail = " ".join(merged_lines[max_lines - 1 :]).strip()
                    merged_lines = head + [tail] if tail else head
            pages[-2] = merged_lines
            pages.pop()

        if len(pages) == 1:
            page_text = "\\N".join(pages[0])
            lines.append(
                f"Dialogue: 0,{_ass_time(local_start)},{_ass_time(local_end)},Default,,0,0,0,,{page_text}"
            )
            continue

        # Weighted pacing (by words/page) improves readability on dense phrases.
        page_durations = _allocate_page_durations(
            pages,
            total_duration=seg_dur,
            min_duration=min_duration,
        )
        if not page_durations or len(page_durations) != len(pages):
            page_dur = seg_dur / float(len(pages))
            page_durations = [page_dur] * len(pages)
        t = local_start
        for i, (page, page_dur) in enumerate(zip(pages, page_durations, strict=True)):
            t_end = local_end if i == len(pages) - 1 else t + float(page_dur)
            if t_end - t < min_duration:
                t_end = min(local_end, t + min_duration)
            if t_end <= t:
                break
            page_text = "\\N".join(page)
            lines.append(
                f"Dialogue: 0,{_ass_time(t)},{_ass_time(t_end)},Default,,0,0,0,,{page_text}"
            )
            t = t_end

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
