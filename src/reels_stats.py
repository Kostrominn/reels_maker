from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


class ReelsStatsError(ValueError):
    pass


STATS_FIELDS = (
    "views_24h",
    "views_72h",
    "nonfollowers_24h",
    "likes_24h",
    "comments_24h",
    "saves_24h",
    "shares_24h",
    "avg_watch_24h",
)

ENGAGEMENT_FIELDS = (
    "likes_24h",
    "comments_24h",
    "saves_24h",
    "shares_24h",
)

TRUTHY = {"1", "y", "yes", "true"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_date_yyyy_mm_dd(raw: str | None) -> date | None:
    txt = str(raw or "").strip()
    if not txt:
        return None
    try:
        return datetime.strptime(txt, "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_float(raw: str | None) -> float | None:
    txt = str(raw or "").strip().replace(",", ".")
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _is_reel_row(row: dict[str, str]) -> bool:
    return bool(str(row.get("file_index") or "").strip())


def _is_posted_row(row: dict[str, str]) -> bool:
    posted = str(row.get("posted") or "").strip().lower()
    status = str(row.get("status") or "").strip().lower()
    return posted in TRUTHY or status == "posted"


def _read_tracker_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Tracker CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ReelsStatsError(f"Tracker CSV has no headers: {path}")
        fieldnames = [str(x) for x in reader.fieldnames]
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({k: str(row.get(k) or "") for k in fieldnames})
    return fieldnames, rows


def _write_tracker_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_excel_friendly_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        # Excel separator hint helps on systems where delimiter autodetect is unreliable.
        f.write("sep=;\r\n")
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";",
            lineterminator="\r\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_excel_unicode_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-16 writes BOM and is reliably detected by older Excel builds.
    with path.open("w", encoding="utf-16", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\r\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _mark_posted_until(rows: list[dict[str, str]], cutoff: date) -> int:
    changed_rows = 0
    for row in rows:
        if not _is_reel_row(row):
            continue
        row_date = _parse_date_yyyy_mm_dd(row.get("date"))
        if row_date is None or row_date > cutoff:
            continue

        changed = False
        posted = str(row.get("posted") or "").strip().lower()
        status = str(row.get("status") or "").strip().lower()
        if posted not in TRUTHY:
            row["posted"] = "yes"
            changed = True
        if status != "posted":
            row["status"] = "posted"
            changed = True
        if changed:
            changed_rows += 1
    return changed_rows


def _metric_coverage(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    filled = sum(1 for row in rows if _parse_float(row.get(field)) is not None)
    return {"filled": filled, "missing": max(0, len(rows) - filled)}


def _row_brief(row: dict[str, str]) -> dict[str, str]:
    return {
        "date": str(row.get("date") or "").strip(),
        "file_index": str(row.get("file_index") or "").strip(),
        "title": str(row.get("title") or "").strip(),
    }


def run_reels_stats(
    *,
    tracker_csv: Path,
    mark_posted_until: str | None,
    write_changes: bool,
    top_n: int,
    summary_json: Path | None,
    excel_csv: Path | None = None,
    excel_tsv: Path | None = None,
) -> dict[str, Any]:
    top_limit = max(1, int(top_n))
    fieldnames, rows = _read_tracker_rows(tracker_csv)

    cutoff_date: date | None = None
    if mark_posted_until:
        cutoff_date = _parse_date_yyyy_mm_dd(mark_posted_until)
        if cutoff_date is None:
            raise ReelsStatsError(
                f"Invalid --mark-posted-until date: {mark_posted_until!r}. Use YYYY-MM-DD."
            )

    rows_marked_posted = 0
    if cutoff_date is not None:
        rows_marked_posted = _mark_posted_until(rows, cutoff_date)
        if write_changes and rows_marked_posted > 0:
            _write_tracker_rows(tracker_csv, fieldnames, rows)

    if excel_csv is not None:
        _write_excel_friendly_csv(excel_csv, fieldnames, rows)
    if excel_tsv is not None:
        _write_excel_unicode_tsv(excel_tsv, fieldnames, rows)

    reel_rows = [row for row in rows if _is_reel_row(row)]
    posted_rows = [row for row in reel_rows if _is_posted_row(row)]
    not_posted_rows = [row for row in reel_rows if not _is_posted_row(row)]

    views_items: list[tuple[float, dict[str, str]]] = []
    for row in posted_rows:
        views = _parse_float(row.get("views_24h"))
        if views is None:
            continue
        views_items.append((views, row))

    views_values = [x[0] for x in views_items]
    views_sum = sum(views_values) if views_values else None
    views_avg = (views_sum / len(views_values)) if views_values else None
    views_median = median(views_values) if views_values else None

    engagement_totals: dict[str, float] = {}
    for field in ENGAGEMENT_FIELDS:
        values = [_parse_float(row.get(field)) for row in posted_rows]
        numbers = [x for x in values if x is not None]
        engagement_totals[field] = float(sum(numbers)) if numbers else 0.0

    views_items.sort(key=lambda x: x[0], reverse=True)
    top_rows = []
    for views, row in views_items[:top_limit]:
        item = _row_brief(row)
        item["views_24h"] = views
        top_rows.append(item)

    missing_views_rows = [_row_brief(row) for row in posted_rows if _parse_float(row.get("views_24h")) is None]

    metric_fill = {field: _metric_coverage(posted_rows, field) for field in STATS_FIELDS}
    posted_without_any_stats = 0
    for row in posted_rows:
        if all(_parse_float(row.get(field)) is None for field in STATS_FIELDS):
            posted_without_any_stats += 1

    summary: dict[str, Any] = {
        "generated_at": _now_iso(),
        "tracker_csv": str(tracker_csv.resolve()),
        "excel_csv": str(excel_csv.resolve()) if excel_csv is not None else None,
        "excel_tsv": str(excel_tsv.resolve()) if excel_tsv is not None else None,
        "mark_posted_until": mark_posted_until,
        "write_changes": write_changes,
        "rows_marked_posted": rows_marked_posted,
        "tracker_rows_total": len(rows),
        "reels_total": len(reel_rows),
        "posted_reels": len(posted_rows),
        "not_posted_reels": len(not_posted_rows),
        "views_24h_filled": len(views_values),
        "views_24h_missing": len(missing_views_rows),
        "views_24h_sum": views_sum,
        "views_24h_avg": views_avg,
        "views_24h_median": views_median,
        "engagement_24h_totals": engagement_totals,
        "metric_fill": metric_fill,
        "posted_without_any_stats": posted_without_any_stats,
        "top_reels_by_views_24h": top_rows,
        "missing_views_24h_rows": missing_views_rows,
    }

    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary
