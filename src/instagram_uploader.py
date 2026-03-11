from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


class InstagramUploadError(RuntimeError):
    pass


@dataclass(frozen=True)
class UploadItem:
    index: int
    filename: str
    path: Path
    camera: str | None = None
    start: float | None = None
    end: float | None = None


@dataclass
class QcResult:
    status: str
    duration_seconds: float | None
    size_bytes: int | None
    video_codec: str | None
    audio_codec: str | None
    width: int | None
    height: int | None
    fps: float | None
    errors: list[str]
    warnings: list[str]

    def notes(self) -> str:
        parts: list[str] = []
        if self.duration_seconds is not None:
            parts.append(f"dur={self.duration_seconds:.2f}s")
        if self.size_bytes is not None:
            parts.append(f"size={self.size_bytes}B")
        if self.video_codec:
            parts.append(f"vcodec={self.video_codec}")
        if self.audio_codec:
            parts.append(f"acodec={self.audio_codec}")
        if self.width and self.height:
            parts.append(f"res={self.width}x{self.height}")
        if self.fps is not None:
            parts.append(f"fps={self.fps:.3f}")
        if self.errors:
            parts.append(f"errors={'; '.join(self.errors)}")
        if self.warnings:
            parts.append(f"warnings={'; '.join(self.warnings)}")
        return " | ".join(parts)


def _truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _to_bool_str(value: bool) -> str:
    return "true" if value else "false"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _safe_fraction(raw: str | None) -> float | None:
    txt = (raw or "").strip()
    if not txt:
        return None
    try:
        return float(Fraction(txt))
    except (ValueError, ZeroDivisionError):
        return None


def _http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    form_data: dict[str, Any] | None = None,
    raw_data: bytes | None = None,
    timeout_seconds: float = 120.0,
) -> Any:
    payload: bytes | None
    if raw_data is not None and form_data is not None:
        raise InstagramUploadError("Both raw_data and form_data are set.")
    if raw_data is not None:
        payload = raw_data
    elif form_data is not None:
        payload = urllib.parse.urlencode(form_data).encode("utf-8")
    else:
        payload = None

    req = urllib.request.Request(url=url, data=payload, method=method.upper())
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    if form_data is not None:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        raw = e.read().decode("utf-8", errors="replace")
        msg = raw
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                err = parsed.get("error")
                if isinstance(err, dict):
                    emsg = str(err.get("message") or "").strip()
                    ecode = err.get("code")
                    etype = str(err.get("type") or "").strip()
                    subcode = err.get("error_subcode")
                    pieces = [f"HTTP {e.code}"]
                    if etype:
                        pieces.append(etype)
                    if ecode is not None:
                        pieces.append(f"code={ecode}")
                    if subcode is not None:
                        pieces.append(f"subcode={subcode}")
                    if emsg:
                        pieces.append(emsg)
                    msg = " | ".join(pieces)
        except json.JSONDecodeError:
            pass
        raise InstagramUploadError(f"Instagram API error: {_truncate(msg)}") from e
    except urllib.error.URLError as e:  # type: ignore[attr-defined]
        raise InstagramUploadError(f"Instagram API network error: {e}") from e

    if not body:
        return {}
    raw_text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise InstagramUploadError(f"Instagram API returned non-JSON body: {_truncate(raw_text)}") from e

    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        err = parsed["error"]
        emsg = str(err.get("message") or "Unknown Graph API error")
        ecode = err.get("code")
        subcode = err.get("error_subcode")
        etype = str(err.get("type") or "")
        pieces = ["Instagram API error"]
        if etype:
            pieces.append(etype)
        if ecode is not None:
            pieces.append(f"code={ecode}")
        if subcode is not None:
            pieces.append(f"subcode={subcode}")
        pieces.append(emsg)
        raise InstagramUploadError(" | ".join(pieces))

    return parsed


class InstagramGraphClient:
    def __init__(
        self,
        *,
        access_token: str,
        ig_user_id: str,
        api_version: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        token = access_token.strip()
        uid = ig_user_id.strip()
        version = api_version.strip() or "v24.0"
        if not token:
            raise InstagramUploadError("INSTAGRAM_ACCESS_TOKEN is missing.")
        if not uid:
            raise InstagramUploadError("INSTAGRAM_IG_USER_ID is missing.")
        self.access_token = token
        self.ig_user_id = uid
        self.api_version = version
        self.timeout_seconds = timeout_seconds
        self.graph_base = f"https://graph.facebook.com/{self.api_version}"

    def create_reel_container(self, *, caption: str, share_to_feed: bool) -> tuple[str, str]:
        url = f"{self.graph_base}/{self.ig_user_id}/media"
        data: dict[str, Any] = {
            "media_type": "REELS",
            "upload_type": "resumable",
            "share_to_feed": _to_bool_str(share_to_feed),
            "access_token": self.access_token,
        }
        if caption.strip():
            data["caption"] = caption.strip()
        resp = _http_json(
            "POST",
            url,
            form_data=data,
            timeout_seconds=self.timeout_seconds,
        )
        if not isinstance(resp, dict):
            raise InstagramUploadError(f"Unexpected create container response: {resp!r}")
        creation_id = str(resp.get("id") or "").strip()
        upload_uri = str(resp.get("uri") or "").strip()
        if not creation_id or not upload_uri:
            raise InstagramUploadError(f"Missing id/uri in create container response: {resp!r}")
        return creation_id, upload_uri

    def upload_binary(self, *, upload_uri: str, video_path: Path) -> None:
        file_size = int(video_path.stat().st_size)
        body = video_path.read_bytes()
        headers = {
            "Authorization": f"OAuth {self.access_token}",
            "offset": "0",
            "file_size": str(file_size),
            "Content-Type": "application/octet-stream",
        }
        resp = _http_json(
            "POST",
            upload_uri,
            headers=headers,
            raw_data=body,
            timeout_seconds=max(self.timeout_seconds, 300.0),
        )
        if isinstance(resp, dict):
            success = resp.get("success")
            if success is False:
                raise InstagramUploadError(f"Upload failed: {resp!r}")

    def get_container_status(self, *, creation_id: str) -> dict[str, Any]:
        fields = "status_code,status,status_reason,error_message"
        url = (
            f"{self.graph_base}/{creation_id}"
            f"?{urllib.parse.urlencode({'fields': fields, 'access_token': self.access_token})}"
        )
        resp = _http_json("GET", url, timeout_seconds=self.timeout_seconds)
        if not isinstance(resp, dict):
            raise InstagramUploadError(f"Unexpected container status response: {resp!r}")
        return resp

    def wait_until_ready(
        self,
        *,
        creation_id: str,
        poll_seconds: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        terminal_ok = {"FINISHED", "PUBLISHED"}
        terminal_error = {"ERROR", "EXPIRED"}
        in_progress = {"IN_PROGRESS", "PROCESSING", "PENDING"}

        deadline = time.time() + timeout_seconds
        last_payload: dict[str, Any] | None = None
        while time.time() < deadline:
            payload = self.get_container_status(creation_id=creation_id)
            last_payload = payload
            status_code = str(payload.get("status_code") or "").strip().upper()
            status = str(payload.get("status") or "").strip().upper()
            effective = status_code or status
            if effective in terminal_ok:
                return payload
            if effective in terminal_error:
                reason = str(payload.get("status_reason") or payload.get("error_message") or "").strip()
                raise InstagramUploadError(
                    f"Container {creation_id} failed with {effective}"
                    + (f": {reason}" if reason else "")
                )
            if effective and effective not in in_progress:
                # Unknown intermediate state: keep polling but include value in the error if timeout.
                pass
            time.sleep(max(1.0, poll_seconds))

        raise InstagramUploadError(
            f"Timed out waiting for container {creation_id} readiness. Last payload: {last_payload!r}"
        )

    def publish_container(self, *, creation_id: str) -> str:
        url = f"{self.graph_base}/{self.ig_user_id}/media_publish"
        resp = _http_json(
            "POST",
            url,
            form_data={"creation_id": creation_id, "access_token": self.access_token},
            timeout_seconds=self.timeout_seconds,
        )
        if not isinstance(resp, dict):
            raise InstagramUploadError(f"Unexpected publish response: {resp!r}")
        media_id = str(resp.get("id") or "").strip()
        if not media_id:
            raise InstagramUploadError(f"Publish response missing media id: {resp!r}")
        return media_id

    def get_permalink(self, *, media_id: str) -> str:
        fields = "permalink,shortcode,media_product_type"
        url = (
            f"{self.graph_base}/{media_id}"
            f"?{urllib.parse.urlencode({'fields': fields, 'access_token': self.access_token})}"
        )
        resp = _http_json("GET", url, timeout_seconds=self.timeout_seconds)
        if isinstance(resp, dict):
            permalink = str(resp.get("permalink") or "").strip()
            if permalink:
                return permalink
            shortcode = str(resp.get("shortcode") or "").strip()
            if shortcode:
                return f"https://www.instagram.com/reel/{shortcode}/"
        return ""


def _read_upload_list(upload_list_csv: Path, upload_pack_dir: Path) -> list[UploadItem]:
    if not upload_list_csv.exists():
        raise FileNotFoundError(f"Upload list not found: {upload_list_csv}")
    items: list[UploadItem] = []
    with upload_list_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_index = str(row.get("index") or "").strip()
            if not raw_index:
                continue
            try:
                index = int(raw_index)
            except ValueError:
                continue
            filename = (
                str(row.get("symlink_name") or "").strip()
                or str(row.get("filename") or "").strip()
                or str(row.get("basename") or "").strip()
            )
            if not filename:
                continue
            abs_path_raw = str(row.get("absolute_path") or "").strip()
            path = upload_pack_dir / filename
            if (not path.exists()) and abs_path_raw:
                abs_path = Path(abs_path_raw)
                if abs_path.exists():
                    path = abs_path
            items.append(
                UploadItem(
                    index=index,
                    filename=filename,
                    path=path,
                    camera=str(row.get("camera") or "").strip() or None,
                    start=_safe_float(row.get("start")),
                    end=_safe_float(row.get("end")),
                )
            )
    items.sort(key=lambda x: x.index)
    return items


def _parse_indices(raw: str | None) -> set[int]:
    expr = (raw or "").strip()
    if not expr:
        return set()
    out: set[int] = set()
    for chunk in expr.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                start, end = end, start
            for i in range(start, end + 1):
                out.add(i)
        else:
            out.add(int(token))
    return out


def _filter_items(
    items: list[UploadItem],
    *,
    index_from: int | None,
    index_to: int | None,
    indices_expr: str | None,
) -> list[UploadItem]:
    selected = _parse_indices(indices_expr)
    out: list[UploadItem] = []
    for item in items:
        if index_from is not None and item.index < index_from:
            continue
        if index_to is not None and item.index > index_to:
            continue
        if selected and item.index not in selected:
            continue
        out.append(item)
    return out


def _probe_media(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise InstagramUploadError("ffprobe not found. Install ffmpeg first.") from e
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise InstagramUploadError(f"ffprobe failed for {path.name}: {_truncate(msg)}")
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as e:
        raise InstagramUploadError(f"ffprobe returned invalid JSON for {path.name}") from e


def inspect_upload_item(item: UploadItem) -> QcResult:
    errors: list[str] = []
    warnings: list[str] = []
    duration: float | None = None
    size: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None

    if not item.path.exists():
        errors.append("file_missing")
        return QcResult(
            status="failed",
            duration_seconds=None,
            size_bytes=None,
            video_codec=None,
            audio_codec=None,
            width=None,
            height=None,
            fps=None,
            errors=errors,
            warnings=warnings,
        )

    size = int(item.path.stat().st_size)
    ext = item.path.suffix.lower()
    if ext not in {".mov", ".mp4"}:
        warnings.append(f"unexpected_ext:{ext or 'none'}")

    payload = _probe_media(item.path)
    streams = payload.get("streams")
    if not isinstance(streams, list):
        streams = []
    fmt = payload.get("format")
    if isinstance(fmt, dict):
        duration = _safe_float(str(fmt.get("duration") or ""))

    video_stream = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"), None)
    if not isinstance(video_stream, dict):
        errors.append("missing_video_stream")
    else:
        vname = str(video_stream.get("codec_name") or "").strip().lower()
        video_codec = vname or None
        width = int(video_stream.get("width") or 0) or None
        height = int(video_stream.get("height") or 0) or None
        fps = _safe_fraction(str(video_stream.get("r_frame_rate") or ""))
        if vname not in {"h264", "hevc"}:
            warnings.append(f"video_codec:{vname or 'unknown'}")
        if width and height:
            if width > 1920 or height > 1920:
                errors.append(f"resolution_too_large:{width}x{height}")
        else:
            warnings.append("missing_resolution")
        if fps is not None and not (23.0 <= fps <= 60.0):
            warnings.append(f"fps_out_of_range:{fps:.3f}")

    if isinstance(audio_stream, dict):
        aname = str(audio_stream.get("codec_name") or "").strip().lower()
        audio_codec = aname or None
        if aname != "aac":
            warnings.append(f"audio_codec:{aname or 'unknown'}")
    else:
        warnings.append("missing_audio_stream")

    if duration is None:
        warnings.append("missing_duration")
    else:
        if duration < 3.0:
            errors.append(f"duration_short:{duration:.2f}")
        if duration > 900.0:
            errors.append(f"duration_long:{duration:.2f}")

    if size > 300 * 1024 * 1024:
        warnings.append("size_gt_300mb")

    if 66 <= item.index <= 69:
        warnings.append("manual_lipsync_check_required")

    status = "ok"
    if errors:
        status = "failed"
    elif warnings:
        status = "warn"
    return QcResult(
        status=status,
        duration_seconds=duration,
        size_bytes=size,
        video_codec=video_codec,
        audio_codec=audio_codec,
        width=width,
        height=height,
        fps=fps,
        errors=errors,
        warnings=warnings,
    )


def _load_caption_overrides(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Caption overrides file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        out: dict[int, str] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    idx = int(str(k).strip())
                except ValueError:
                    continue
                out[idx] = str(v or "").strip()
        return out

    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(str(row.get("index") or "").strip())
            except ValueError:
                continue
            caption = str(row.get("caption") or "").strip()
            out[idx] = caption
    return out


def resolve_caption(
    *,
    item: UploadItem,
    template: str,
    caption_overrides: dict[int, str],
) -> str:
    if item.index in caption_overrides:
        return caption_overrides[item.index]
    tpl = template.strip()
    if not tpl:
        return ""
    fields: dict[str, Any] = {
        "index": item.index,
        "filename": item.filename,
        "camera": item.camera or "",
        "start": item.start if item.start is not None else 0.0,
        "end": item.end if item.end is not None else 0.0,
    }
    try:
        return tpl.format(**fields).strip()
    except Exception as e:  # noqa: BLE001
        raise InstagramUploadError(f"Invalid --caption-template: {e}") from e


PROGRESS_HEADERS = [
    "index",
    "filename",
    "status",
    "instagram_post_url",
    "published_at",
    "notes",
]


def _load_progress(progress_csv: Path) -> dict[int, dict[str, str]]:
    if not progress_csv.exists():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with progress_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(str(row.get("index") or "").strip())
            except ValueError:
                continue
            rows[idx] = {
                "index": str(idx),
                "filename": str(row.get("filename") or "").strip(),
                "status": str(row.get("status") or "").strip(),
                "instagram_post_url": str(row.get("instagram_post_url") or "").strip(),
                "published_at": str(row.get("published_at") or "").strip(),
                "notes": str(row.get("notes") or "").strip(),
            }
    return rows


def _write_progress(progress_csv: Path, rows: dict[int, dict[str, str]]) -> None:
    progress_csv.parent.mkdir(parents=True, exist_ok=True)
    with progress_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_HEADERS)
        writer.writeheader()
        for idx in sorted(rows):
            row = rows[idx]
            writer.writerow(
                {
                    "index": str(idx),
                    "filename": row.get("filename", ""),
                    "status": row.get("status", ""),
                    "instagram_post_url": row.get("instagram_post_url", ""),
                    "published_at": row.get("published_at", ""),
                    "notes": row.get("notes", ""),
                }
            )


def _set_progress(
    rows: dict[int, dict[str, str]],
    *,
    item: UploadItem,
    status: str,
    instagram_post_url: str = "",
    published_at: str = "",
    notes: str = "",
) -> None:
    rows[item.index] = {
        "index": str(item.index),
        "filename": item.filename,
        "status": status,
        "instagram_post_url": instagram_post_url,
        "published_at": published_at,
        "notes": _truncate(notes, 1400),
    }


def run_instagram_upload(
    *,
    upload_list_csv: Path,
    upload_pack_dir: Path,
    progress_csv: Path,
    summary_json: Path | None,
    caption_template: str,
    caption_overrides_path: Path | None,
    index_from: int | None,
    index_to: int | None,
    indices: str | None,
    batch_size: int,
    sleep_between_seconds: float,
    dry_run: bool,
    qc_only: bool,
    strict_qc: bool,
    force_retry_failed: bool,
    share_to_feed: bool,
    poll_seconds: float,
    publish_timeout_seconds: float,
    api_version: str | None,
) -> dict[str, Any]:
    resolved_api_version = (api_version or os.getenv("INSTAGRAM_GRAPH_API_VERSION", "v24.0")).strip() or "v24.0"
    items = _read_upload_list(upload_list_csv, upload_pack_dir)
    selected = _filter_items(items, index_from=index_from, index_to=index_to, indices_expr=indices)
    caption_overrides = _load_caption_overrides(caption_overrides_path)
    progress_rows = _load_progress(progress_csv)

    summary: dict[str, Any] = {
        "started_at": _now_iso(),
        "upload_list_csv": str(upload_list_csv.resolve()),
        "upload_pack_dir": str(upload_pack_dir.resolve()),
        "progress_csv": str(progress_csv.resolve()),
        "selected_total": len(selected),
        "batch_size": batch_size,
        "dry_run": dry_run,
        "qc_only": qc_only,
        "strict_qc": strict_qc,
        "api_version": resolved_api_version,
        "published": 0,
        "failed": 0,
        "skipped_published": 0,
        "skipped_failed": 0,
        "skipped_qc": 0,
        "processed_now": 0,
        "processed_indices": [],
    }

    client: InstagramGraphClient | None = None
    if not dry_run and not qc_only:
        timeout_raw = os.getenv("INSTAGRAM_HTTP_TIMEOUT", "").strip()
        timeout = 120.0
        if timeout_raw:
            try:
                timeout = max(10.0, float(timeout_raw))
            except ValueError:
                timeout = 120.0
        client = InstagramGraphClient(
            access_token=os.getenv("INSTAGRAM_ACCESS_TOKEN", ""),
            ig_user_id=os.getenv("INSTAGRAM_IG_USER_ID", ""),
            api_version=resolved_api_version,
            timeout_seconds=timeout,
        )

    max_uploads: int | None = None if batch_size <= 0 else batch_size
    uploads_done = 0

    for item in selected:
        prev = progress_rows.get(item.index)
        prev_status = str((prev or {}).get("status") or "").strip().lower()
        if prev_status == "published":
            summary["skipped_published"] += 1
            continue
        if prev_status == "failed" and not force_retry_failed:
            summary["skipped_failed"] += 1
            continue

        qc = inspect_upload_item(item)
        qc_note = qc.notes()
        if qc.status == "failed":
            summary["skipped_qc"] += 1
            _set_progress(
                progress_rows,
                item=item,
                status="failed",
                notes=f"qc_failed: {qc_note}",
            )
            continue
        if strict_qc and qc.status == "warn":
            summary["skipped_qc"] += 1
            _set_progress(
                progress_rows,
                item=item,
                status="skipped",
                notes=f"qc_warn_strict: {qc_note}",
            )
            continue

        caption = resolve_caption(item=item, template=caption_template, caption_overrides=caption_overrides)

        if qc_only:
            summary["processed_indices"].append(item.index)
            _set_progress(
                progress_rows,
                item=item,
                status="skipped",
                notes=f"qc_{qc.status}: {qc_note}",
            )
            summary["processed_now"] += 1
            continue

        if max_uploads is not None and uploads_done >= max_uploads:
            break

        summary["processed_indices"].append(item.index)

        if dry_run:
            _set_progress(
                progress_rows,
                item=item,
                status="skipped",
                notes=f"dry_run | qc_{qc.status}: {qc_note}",
            )
            summary["processed_now"] += 1
            uploads_done += 1
            continue

        assert client is not None
        try:
            creation_id, upload_uri = client.create_reel_container(
                caption=caption,
                share_to_feed=share_to_feed,
            )
            client.upload_binary(upload_uri=upload_uri, video_path=item.path)
            status_payload = client.wait_until_ready(
                creation_id=creation_id,
                poll_seconds=poll_seconds,
                timeout_seconds=publish_timeout_seconds,
            )
            media_id = client.publish_container(creation_id=creation_id)
            permalink = client.get_permalink(media_id=media_id)
            notes = f"qc_{qc.status}: {qc_note}; creation_id={creation_id}; media_id={media_id}; status={status_payload.get('status_code', '')}"
            _set_progress(
                progress_rows,
                item=item,
                status="published",
                instagram_post_url=permalink,
                published_at=_now_iso(),
                notes=notes,
            )
            summary["published"] += 1
        except Exception as e:  # noqa: BLE001
            _set_progress(
                progress_rows,
                item=item,
                status="failed",
                notes=f"qc_{qc.status}: {qc_note}; upload_error={_truncate(str(e), 900)}",
            )
            summary["failed"] += 1

        summary["processed_now"] += 1
        uploads_done += 1
        if sleep_between_seconds > 0:
            time.sleep(sleep_between_seconds)

    summary["finished_at"] = _now_iso()
    _write_progress(progress_csv, progress_rows)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
