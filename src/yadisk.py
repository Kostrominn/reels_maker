from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


API_BASE = "https://cloud-api.yandex.net/v1/disk"


class YandexDiskError(RuntimeError):
    pass


def _require_token() -> str:
    token = os.getenv("YANDEX_DISK_OAUTH_TOKEN", "").strip()
    if not token:
        raise YandexDiskError(
            "Не найден YANDEX_DISK_OAUTH_TOKEN. "
            "Добавьте OAuth-токен в .env (не client secret)."
        )
    return token


def _request(
    method: str,
    path: str,
    *,
    token: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{API_BASE}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"OAuth {token}")
    req.add_header("Accept", "application/json")

    timeout_s = float(os.getenv("YANDEX_DISK_TIMEOUT", "30") or "30")
    debug = os.getenv("REELS_MAKER_YADISK_DEBUG", "").strip() == "1"
    if debug:
        print(f"[yadisk] {method} {url}")

    def _use_no_proxy() -> bool:
        return os.getenv("REELS_MAKER_YADISK_NO_PROXY", "").strip() == "1"

    def _is_timeout(err: urllib.error.URLError) -> bool:  # type: ignore[type-arg]
        reason = getattr(err, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return True
        msg = str(err).lower()
        return "timed out" in msg or "timeout" in msg

    try:
        if _use_no_proxy():
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=timeout_s) as resp:
                data = resp.read().decode("utf-8")
        else:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        body = e.read().decode("utf-8", errors="replace")
        raise YandexDiskError(f"Yandex Disk API error {e.code}: {body}") from e
    except urllib.error.URLError as e:  # type: ignore[attr-defined]
        if not _use_no_proxy() and _is_timeout(e):
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(req, timeout=timeout_s) as resp:
                    data = resp.read().decode("utf-8")
            except urllib.error.URLError as e2:  # type: ignore[attr-defined]
                raise YandexDiskError(f"Network error: {e2}") from e2
        else:
            raise YandexDiskError(f"Network error: {e}") from e

    if not data:
        return {}
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise YandexDiskError(f"Invalid JSON response: {e}") from e


def get_disk_info() -> dict[str, Any]:
    token = _require_token()
    return _request("GET", "/", token=token)


def list_resources(path: str = "/", limit: int = 20) -> dict[str, Any]:
    token = _require_token()
    return _request("GET", "/resources", token=token, params={"path": path, "limit": limit})


def list_all_files(limit: int = 20, media_type: str | None = None) -> dict[str, Any]:
    token = _require_token()
    params: dict[str, Any] = {"limit": limit}
    if media_type:
        params["media_type"] = media_type
    return _request("GET", "/resources/files", token=token, params=params)


def get_download_url(path: str) -> str:
    token = _require_token()
    data = _request("GET", "/resources/download", token=token, params={"path": path})
    href = data.get("href")
    if not href:
        raise YandexDiskError("Не удалось получить ссылку для скачивания (href отсутствует).")
    return str(href)
