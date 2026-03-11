from __future__ import annotations

import json
import os
import re
import socket
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .models import ClipAnalysis, ClipCandidate, Transcript
from .utils import AppConfig, seconds_to_timestamp

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "ollama": "http://127.0.0.1:11434/v1",
}


def _import_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Не найден пакет openai. Установите зависимости: pip install -r requirements.txt"
        ) from e
    return OpenAI


def _load_prompt(prompt_path: str | Path) -> str:
    p = Path(prompt_path)
    return p.read_text(encoding="utf-8")


def _segments_to_text(segments: list[Any]) -> str:
    lines: list[str] = []
    for s in segments:
        ts1 = seconds_to_timestamp(s.start)
        ts2 = seconds_to_timestamp(s.end)
        lines.append(f"[{ts1} - {ts2}] {s.text}")
    return "\n".join(lines)


def _extract_json(text: str) -> Any:
    """
    Best-effort JSON extraction from model output.
    Accepts raw JSON or JSON embedded in extra text.
    """
    text = _replace_timestamp_literals((text or "").strip())
    if not text:
        return []

    # If it's already JSON
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass

    # Try fenced code block first
    fenced = re.search(r"```(?:json)?\\s*(.*?)\\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            return json.loads(_replace_timestamp_literals(candidate))
        except Exception:  # noqa: BLE001
            pass

    # Try to find first JSON array/object
    start_candidates = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not start_candidates:
        raise ValueError("LLM output does not contain JSON")

    start = min(start_candidates)
    snippet = text[start:]

    # Trim trailing code fences if any
    snippet = snippet.strip().strip("```").strip()
    try:
        return json.loads(_replace_timestamp_literals(snippet))
    except Exception:  # noqa: BLE001
        pass

    # Last resort: trim to last closing brace/bracket
    end = max(snippet.rfind("}"), snippet.rfind("]"))
    if end != -1 and end > 0:
        snippet = snippet[: end + 1].strip()
        return json.loads(_replace_timestamp_literals(snippet))

    raise ValueError("LLM output does not contain valid JSON")


def _timestamp_to_seconds(value: str) -> float | None:
    s = (value or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass

    parts = s.split(":")
    if len(parts) not in {2, 3}:
        return None
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return None


def _replace_timestamp_literals(text: str) -> str:
    """
    Convert `start/end` timestamp literals like 00:01:40.000 into seconds.
    Handles both quoted and unquoted values to make JSON parseable.
    """
    if not text:
        return text

    pattern = re.compile(
        r'(?P<prefix>"(?:start|end)"\s*:\s*)(?P<quote>["\']?)'
        r"(?P<ts>\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)"
        r"(?P=quote)"
    )

    def repl(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        ts = match.group("ts")
        sec = _timestamp_to_seconds(ts)
        if sec is None:
            return match.group(0)
        return f"{prefix}{sec:.3f}"

    return pattern.sub(repl, text)


def _mmss_number_to_seconds(value: float) -> float | None:
    minutes = int(value)
    seconds = round((value - minutes) * 100, 3)
    if 0 <= seconds < 60:
        return minutes * 60 + seconds
    return None


def _normalize_candidate_item(
    item: dict[str, Any],
    *,
    chunk_start: float | None = None,
    chunk_end: float | None = None,
) -> dict[str, Any]:
    out = dict(item)
    for key in ("start", "end"):
        val = out.get(key)
        if isinstance(val, str):
            sec = _timestamp_to_seconds(val)
            if sec is not None:
                out[key] = sec
                val = sec
        if isinstance(val, (int, float)) and chunk_start is not None and chunk_end is not None:
            cur = float(val)
            # Some models return minute.second numeric shorthand (e.g. 15.01 for 15:01).
            if chunk_start >= 300 and cur < max(120.0, chunk_start - 120):
                mmss = _mmss_number_to_seconds(cur)
                if mmss is not None and (chunk_start - 180) <= mmss <= (chunk_end + 180):
                    out[key] = mmss
    score = out.get("score")
    if isinstance(score, str):
        try:
            out["score"] = float(score)
        except ValueError:
            pass
    return out


def _redact_proxy_url(proxy_url: str) -> str:
    """
    Remove credentials from proxy URL for safe error messages.
    """
    parts = urlsplit(proxy_url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    scheme = parts.scheme or "http"
    return f"{scheme}://{host}{port}"


def _check_proxy_reachable(proxy_url: str, *, timeout_seconds: float = 2.0) -> None:
    """
    Fail fast if the configured proxy is not reachable (instead of hanging for minutes).
    """
    parts = urlsplit(proxy_url)
    host = parts.hostname
    port = parts.port
    if not host or not port:
        return
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return
    except OSError as e:
        safe = _redact_proxy_url(proxy_url)
        raise RuntimeError(
            f"Не удаётся подключиться к прокси {safe}. "
            "Проверь адрес/порт (частая ошибка: docker-адрес 172.17.0.1 недоступен с хоста; "
            "часто нужен localhost/127.0.0.1 или host.docker.internal)."
        ) from e


def _replace_proxy_host(proxy_url: str, new_host: str) -> str:
    parts = urlsplit(proxy_url)
    userinfo = ""
    if parts.username:
        userinfo = parts.username
        if parts.password:
            userinfo += f":{parts.password}"
        userinfo += "@"
    hostport = f"{new_host}:{parts.port}" if parts.port else new_host
    netloc = userinfo + hostport
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _select_reachable_proxy(proxy_url: str, *, strict: bool = False) -> str | None:
    """
    Try to find a reachable proxy URL. Useful when user copied a docker-bridge IP
    that is not reachable from the host OS.
    """
    # First try as-is.
    try:
        _check_proxy_reachable(proxy_url, timeout_seconds=2.0)
        return proxy_url
    except RuntimeError:
        pass

    # Then try common host-side alternatives.
    candidates = [
        _replace_proxy_host(proxy_url, "127.0.0.1"),
        _replace_proxy_host(proxy_url, "localhost"),
        _replace_proxy_host(proxy_url, "host.docker.internal"),
    ]
    for cand in candidates:
        try:
            _check_proxy_reachable(cand, timeout_seconds=2.0)
            return cand
        except RuntimeError:
            continue

    # No candidates worked; raise original error for clarity.
    if strict:
        _check_proxy_reachable(proxy_url, timeout_seconds=2.0)
        return proxy_url

    warnings.warn(
        "Proxy is configured but not reachable. Falling back to direct connection. "
        "Set REELS_MAKER_PROXY_STRICT=1 to fail instead.",
        RuntimeWarning,
    )
    return None


def _split_segments(
    segments: list[Any],
    *,
    max_chars: int,
    overlap_segments: int = 1,
) -> list[list[Any]]:
    if max_chars <= 0 or not segments:
        return [segments]

    chunks: list[list[Any]] = []
    cur: list[Any] = []
    cur_len = 0

    def seg_len(seg: Any) -> int:
        return len(f"[{seconds_to_timestamp(seg.start)} - {seconds_to_timestamp(seg.end)}] {seg.text}\n")

    for seg in segments:
        seg_l = seg_len(seg)
        if cur and cur_len + seg_l > max_chars:
            chunks.append(cur)
            if overlap_segments > 0:
                cur = cur[-overlap_segments:]
                cur_len = sum(seg_len(s) for s in cur)
            else:
                cur = []
                cur_len = 0
        cur.append(seg)
        cur_len += seg_l

    if cur:
        chunks.append(cur)
    return chunks


def _overlap_ratio(a: ClipCandidate, b: ClipCandidate) -> float:
    a_len = max(0.0, a.end - a.start)
    b_len = max(0.0, b.end - b.start)
    if a_len <= 0 or b_len <= 0:
        return 0.0
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    if inter <= 0:
        return 0.0
    return inter / min(a_len, b_len)


def _select_best_candidates(
    candidates: list[ClipCandidate],
    *,
    target_duration: float,
    max_candidates: int = 6,
    overlap_threshold: float = 0.8,
) -> list[ClipCandidate]:
    if max_candidates <= 0 or not candidates:
        return []

    tgt = max(1.0, float(target_duration))

    def rank_key(c: ClipCandidate) -> tuple[float, float]:
        score = float(c.score) if c.score is not None else 0.0
        dur = max(0.01, c.end - c.start)
        duration_penalty = abs(dur - tgt) / tgt
        effective = score - 0.08 * duration_penalty
        return (effective, score)

    ranked = sorted(candidates, key=rank_key, reverse=True)
    selected: list[ClipCandidate] = []
    for cand in ranked:
        if any(_overlap_ratio(cand, keep) >= overlap_threshold for keep in selected):
            continue
        selected.append(cand)
        if len(selected) >= max_candidates:
            break
    return sorted(selected, key=lambda c: c.start)


def _closest_value(values: list[float], target: float) -> float:
    if not values:
        return target
    return min(values, key=lambda v: abs(v - target))


def _snap_candidate_to_segments(
    cand: ClipCandidate,
    *,
    seg_starts: list[float],
    seg_ends: list[float],
) -> ClipCandidate:
    c = cand.model_copy(deep=True)
    c.start = _closest_value(seg_starts, float(c.start))
    c.end = _closest_value(seg_ends, float(c.end))
    if c.end <= c.start:
        later = [e for e in seg_ends if e > c.start]
        if later:
            c.end = min(later)
    return c


def _enforce_duration_bounds(
    cand: ClipCandidate,
    *,
    seg_starts: list[float],
    seg_ends: list[float],
    min_duration: float,
    max_duration: float,
    soft_max_factor: float = 1.25,
) -> tuple[ClipCandidate | None, bool]:
    """
    Ensure candidate duration is within bounds.
    Returns (candidate_or_none, changed_flag).
    """
    c = cand.model_copy(deep=True)
    changed = False
    min_d = max(1.0, float(min_duration))
    max_d = max(min_d, float(max_duration))
    soft_max = max_d * float(soft_max_factor)

    dur = float(c.end) - float(c.start)
    if dur <= 0:
        return None, changed

    if dur < min_d:
        target_end = float(c.start) + min_d
        feasible = [e for e in seg_ends if e >= target_end and e > float(c.start) and (e - float(c.start)) <= soft_max]
        if feasible:
            c.end = min(feasible)
            changed = True
        else:
            return None, changed

    dur = float(c.end) - float(c.start)
    if dur > soft_max:
        target_end = float(c.start) + max_d
        feasible = [e for e in seg_ends if e > float(c.start)]
        if not feasible:
            return None, changed
        c.end = _closest_value(feasible, target_end)
        changed = True
        if (float(c.end) - float(c.start)) > soft_max:
            return None, changed

    if float(c.end) <= float(c.start):
        return None, changed
    return c, changed


def _provider_from_base_url(base_url: str | None) -> str:
    if not base_url:
        return ""
    host = (urlsplit(base_url).hostname or "").lower()
    if "openrouter.ai" in host:
        return "openrouter"
    if "generativelanguage.googleapis.com" in host:
        return "gemini"
    if "api.openai.com" in host:
        return "openai"
    if "api.groq.com" in host:
        return "groq"
    if "api.together.xyz" in host:
        return "together"
    if "api.fireworks.ai" in host:
        return "fireworks"
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}:
        return "ollama"
    return ""


def _is_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = (urlsplit(base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}


def _resolve_provider(cfg: AppConfig, *, base_url: str | None) -> str:
    env_provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    cfg_provider = (getattr(cfg.llm, "provider", "") or "").strip().lower()
    inferred = _provider_from_base_url(base_url)
    provider = env_provider or cfg_provider or inferred or "openai"
    return provider


def _resolve_base_url(cfg: AppConfig, provider: str) -> str | None:
    explicit = (
        os.getenv("LLM_BASE_URL")
        or getattr(cfg.llm, "base_url", None)
        or os.getenv("OPENAI_BASE_URL")
        or None
    )
    if explicit:
        return explicit
    return _PROVIDER_BASE_URLS.get(provider)


def _resolve_api_key(provider: str, *, base_url: str | None) -> str:
    provider_specific: dict[str, list[str]] = {
        "openrouter": ["OPENROUTER_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "together": ["TOGETHER_API_KEY"],
        "fireworks": ["FIREWORKS_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
    }
    env_candidates = provider_specific.get(provider, [])
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        for key_name in env_candidates:
            api_key = os.getenv(key_name)
            if api_key:
                break
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    if provider == "ollama" or _is_local_base_url(base_url):
        return "local-llm"

    env_hint = "LLM_API_KEY"
    if env_candidates:
        env_hint = f"{env_hint} или {' / '.join(env_candidates)}"
    raise RuntimeError(
        f"Не задан API-ключ LLM ({env_hint}). "
        "Для локального Ollama укажите LLM_PROVIDER=ollama или LLM_BASE_URL=http://127.0.0.1:11434/v1."
    )


def _resolve_default_headers(provider: str) -> dict[str, str] | None:
    headers: dict[str, str] = {}
    if provider == "openrouter":
        site_url = (os.getenv("OPENROUTER_SITE_URL") or "").strip()
        app_name = (os.getenv("OPENROUTER_APP_NAME") or "").strip()
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

    extra_headers_raw = (os.getenv("LLM_EXTRA_HEADERS_JSON") or "").strip()
    if extra_headers_raw:
        try:
            extra_headers = json.loads(extra_headers_raw)
            if not isinstance(extra_headers, dict):
                raise ValueError("must be JSON object")
            for k, v in extra_headers.items():
                if isinstance(k, str) and isinstance(v, str):
                    headers[k] = v
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Некорректный LLM_EXTRA_HEADERS_JSON: ожидается JSON-объект, "
                'например {"Header-Name":"value"}.'
            ) from e

    return headers or None


class LlmAnalyzer:
    def __init__(self, cfg: AppConfig, prompt_path: str | Path = "./prompts/find_reels.txt"):
        self.cfg = cfg
        self.prompt_path = Path(prompt_path)

        OpenAI = _import_openai()
        cfg_base_url = (
            os.getenv("LLM_BASE_URL")
            or getattr(self.cfg.llm, "base_url", None)
            or os.getenv("OPENAI_BASE_URL")
            or None
        )
        self.provider = _resolve_provider(self.cfg, base_url=cfg_base_url)
        self.base_url = _resolve_base_url(self.cfg, self.provider)
        api_key = _resolve_api_key(self.provider, base_url=self.base_url)
        default_headers = _resolve_default_headers(self.provider)

        # Respect system-style proxies (HTTP_PROXY/HTTPS_PROXY). We explicitly pass an httpx client
        # to avoid surprises across environments.
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None
        trust_env = True
        if self.provider == "ollama" or _is_local_base_url(self.base_url):
            # Local Ollama should not depend on external proxies.
            proxy = None
            trust_env = False
        elif proxy:
            strict = os.getenv("REELS_MAKER_PROXY_STRICT") in {"1", "true", "True"}
            proxy = _select_reachable_proxy(proxy, strict=strict)
            if proxy is None:
                trust_env = False
        http_client = httpx.Client(
            proxy=proxy,
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=5.0),
            trust_env=trust_env,
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            default_headers=default_headers,
            http_client=http_client,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def analyze(self, transcript: Transcript, *, max_segments: int | None = None) -> ClipAnalysis:
        prompt_template = _load_prompt(self.prompt_path)
        prompt = prompt_template.format(
            min_duration=self.cfg.llm.min_duration,
            max_duration=self.cfg.llm.max_duration,
            target_duration=self.cfg.llm.target_duration,
        )

        if max_segments:
            chunks = [transcript.segments[:max_segments]]
        else:
            max_chars = getattr(self.cfg.llm, "max_chars", 12000)
            overlap = getattr(self.cfg.llm, "chunk_overlap_segments", 1)
            chunks = _split_segments(transcript.segments, max_chars=max_chars, overlap_segments=overlap)

        max_end = max((s.end for s in transcript.segments), default=0.0)
        seg_starts = [float(s.start) for s in transcript.segments]
        seg_ends = [float(s.end) for s in transcript.segments]
        candidates: list[ClipCandidate] = []
        dropped: int = 0
        dropped_duration: int = 0
        duration_adjusted: int = 0
        clamped: int = 0
        candidates_before_postprocess: int = 0
        for segs in chunks:
            transcript_text = _segments_to_text(segs)
            chunk_start = min((s.start for s in segs), default=0.0)
            chunk_end = max((s.end for s in segs), default=0.0)
            model_name = self.cfg.llm.model or ""
            token_param_env = (os.getenv("LLM_TOKENS_PARAM") or "").strip()
            token_param = "max_tokens"
            if token_param_env in {"max_tokens", "max_completion_tokens"}:
                token_param = token_param_env
            elif self.provider == "openai" and (model_name.startswith("gpt-5") or model_name.startswith("o")):
                token_param = "max_completion_tokens"

            resp = self.client.chat.completions.create(
                model=self.cfg.llm.model,
                temperature=self.cfg.llm.temperature,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            "Вот транскрипт. Выбери лучшие фрагменты.\n"
                            f"Максимальный таймкод: {max_end:.2f} секунд. "
                            "Не выходи за пределы.\n\n"
                            + transcript_text
                        ),
                    },
                ],
                **{token_param: self.cfg.llm.max_tokens},
            )

            content = resp.choices[0].message.content if resp.choices else ""
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
                content = "\n".join(parts)
            try:
                data = _extract_json(content)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"LLM returned non-JSON output for a chunk: {type(e).__name__}: {e}. "
                    "Chunk will be skipped.",
                    RuntimeWarning,
                )
                data = []
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    item = _normalize_candidate_item(
                        item,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                    )
                    if "start" not in item or "end" not in item:
                        continue
                    cand = ClipCandidate.model_validate(item)
                    cand = _snap_candidate_to_segments(cand, seg_starts=seg_starts, seg_ends=seg_ends)
                    cand2, changed = _enforce_duration_bounds(
                        cand,
                        seg_starts=seg_starts,
                        seg_ends=seg_ends,
                        min_duration=float(self.cfg.llm.min_duration),
                        max_duration=float(self.cfg.llm.max_duration),
                        soft_max_factor=1.25,
                    )
                    if cand2 is None:
                        dropped += 1
                        dropped_duration += 1
                        continue
                    cand = cand2
                    if changed:
                        duration_adjusted += 1
                    # Validate bounds against transcript length.
                    if cand.start < 0 or cand.end <= cand.start:
                        dropped += 1
                        continue
                    if cand.end > max_end:
                        # Allow tiny overrun due to rounding; otherwise drop.
                        if cand.end - max_end <= 0.5:
                            cand.end = max_end
                            clamped += 1
                        else:
                            dropped += 1
                            continue
                    candidates.append(cand)

        candidates_before_postprocess = len(candidates)
        candidates = _select_best_candidates(
            candidates,
            target_duration=float(self.cfg.llm.target_duration),
            max_candidates=max(1, int(getattr(self.cfg.llm, "max_candidates", 6))),
        )

        return ClipAnalysis(
            transcript_path="",
            video_path=transcript.video_path,
            llm_model=self.cfg.llm.model,
            candidates=candidates,
            meta={
                "provider": self.provider,
                "base_url": self.base_url,
                "chunks": len(chunks),
                "candidates_total": len(candidates),
                "candidates_before_postprocess": candidates_before_postprocess,
                "candidates_dropped": dropped,
                "candidates_dropped_duration": dropped_duration,
                "candidates_duration_adjusted": duration_adjusted,
                "candidates_clamped": clamped,
                "transcript_max_end": max_end,
            },
        )
