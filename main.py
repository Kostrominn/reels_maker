from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from rich.console import Console

from src.audio_extractor import extract_wav_16k_mono
from src.instagram_uploader import InstagramUploadError, run_instagram_upload
from src.llm_analyzer import LlmAnalyzer
from src.models import ClipAnalysis, Transcript
from src.multicam import cmd_multicam
from src.sync import apply_sync_trim, cache_full_audio, make_sync_mix, sync_offset
from src.transcriber import Transcriber
from src.utils import AppConfig, ensure_dirs, get_storage_paths, load_config, read_json, write_json
from src.video_cutter import cut_clip, format_clip_name
from src.yadisk import YandexDiskError, get_download_url, list_all_files, list_resources

console = Console()


def _save_transcript(cfg: AppConfig, transcript: Transcript, out_path: Path) -> None:
    write_json(out_path, transcript.model_dump(mode="json"))
    console.print(f"[green]Saved transcript[/green] -> {out_path}")


def _save_clip_analysis(analysis: ClipAnalysis, out_path: Path) -> None:
    write_json(out_path, analysis.model_dump(mode="json"))
    console.print(f"[green]Saved clips metadata[/green] -> {out_path}")


def _video_stem(video_path: str | Path) -> str:
    if isinstance(video_path, str) and video_path.startswith("http"):
        parsed = urlparse(video_path)
        q = parse_qs(parsed.query)
        if "filename" in q and q["filename"]:
            name = unquote(q["filename"][0])
        else:
            name = Path(unquote(parsed.path)).name
        return Path(name).stem or "remote"
    return Path(video_path).stem


def cmd_transcribe(
    cfg: AppConfig,
    video_path: str | Path,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> Path:
    storage = get_storage_paths(cfg)
    ensure_dirs(storage.audio, storage.transcripts)

    stem = _video_stem(video_path)
    suffix = ""
    if start_seconds is not None or duration_seconds is not None:
        suffix = f"_s{start_seconds or 0:g}_d{duration_seconds or 0:g}"
    audio_path = storage.audio / f"{stem}{suffix}.wav"
    transcript_path = storage.transcripts / f"{stem}{suffix}.json"

    console.print(f"[cyan]Extract audio[/cyan] {video_path} -> {audio_path}")
    extract_wav_16k_mono(
        video_path,
        audio_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )

    console.print("[cyan]Transcribe[/cyan] (this can take a while)…")
    transcriber = Transcriber(cfg)
    transcript = transcriber.transcribe_audio(audio_path=audio_path, video_path=video_path)

    _save_transcript(cfg, transcript, transcript_path)
    return transcript_path


def cmd_analyze(cfg: AppConfig, transcript_path: Path) -> Path:
    storage = get_storage_paths(cfg)
    ensure_dirs(storage.clips)

    data = read_json(transcript_path)
    transcript = Transcript.model_validate(data)

    console.print("[cyan]Analyze transcript with LLM[/cyan]…")
    analyzer = LlmAnalyzer(cfg)
    analysis = analyzer.analyze(transcript)
    analysis.transcript_path = str(transcript_path)

    out_path = storage.clips / f"{transcript_path.stem}_clips.json"
    _save_clip_analysis(analysis, out_path)
    return out_path


def cmd_generate(cfg: AppConfig, video_path: str | Path, clips_path: Path) -> Path:
    storage = get_storage_paths(cfg)
    ensure_dirs(storage.output)

    data = read_json(clips_path)
    analysis = ClipAnalysis.model_validate(data)

    stem = _video_stem(video_path)
    created: list[str] = []

    for idx, c in enumerate(analysis.candidates, start=1):
        name = format_clip_name(stem, c.start, c.end, idx, ext=cfg.video.output_format)
        out_path = storage.output / name
        console.print(f"[cyan]Cut[/cyan] {c.start:.2f}–{c.end:.2f} -> {out_path}")
        cut_clip(cfg=cfg, video_path=video_path, start=c.start, end=c.end, output_path=out_path)
        created.append(str(out_path))

    manifest_path = storage.output / f"{stem}_manifest.json"
    write_json(
        manifest_path,
        {"video": str(video_path), "clips": created, "clips_metadata": str(clips_path)},
    )
    console.print(f"[green]Saved manifest[/green] -> {manifest_path}")
    return manifest_path


def cmd_full_pipeline(cfg: AppConfig, video_path: str | Path) -> None:
    transcript_path = cmd_transcribe(cfg, video_path)
    clips_path = cmd_analyze(cfg, transcript_path)
    cmd_generate(cfg, video_path, clips_path)


def cmd_yadisk_check(path: str, limit: int) -> None:
    try:
        data = list_resources(path=path, limit=limit)
    except YandexDiskError as e:
        console.print(f"[red]Yandex Disk error[/red]: {e}")
        raise SystemExit(1) from e

    embedded = data.get("_embedded") or {}
    items = embedded.get("items") or []
    console.print(f"[green]OK[/green] Доступ к Яндекс.Диску есть.")
    console.print(f"Path: {path} | Items: {len(items)}")
    for it in items:
        name = it.get("name") or "<no-name>"
        rtype = it.get("type") or "unknown"
        size = it.get("size")
        size_str = f"{size} bytes" if isinstance(size, int) else "-"
        console.print(f"  - {rtype:>5} | {size_str:>10} | {name}")


def cmd_yadisk_files(path_prefix: str, limit: int, media_type: str | None) -> None:
    try:
        data = list_all_files(limit=limit, media_type=media_type)
    except YandexDiskError as e:
        console.print(f"[red]Yandex Disk error[/red]: {e}")
        raise SystemExit(1) from e

    items = data.get("items") or []
    if path_prefix:
        items = [it for it in items if str(it.get("path") or "").startswith(path_prefix)]

    console.print(f"[green]OK[/green] Files: {len(items)} (filtered)")
    for it in items:
        name = it.get("name") or "<no-name>"
        path = it.get("path") or "-"
        size = it.get("size")
        size_str = f"{size} bytes" if isinstance(size, int) else "-"
        mtype = it.get("media_type") or "-"
        console.print(f"  - {mtype:>5} | {size_str:>10} | {path} | {name}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reels Maker CLI")
    p.add_argument("--config", default=None, help="Path to config.yaml (optional)")
    sub = p.add_subparsers(dest="command", required=True)

    p_t = sub.add_parser("transcribe", help="video -> transcript JSON")
    p_t.add_argument("video", type=str)
    p_t.add_argument("--start-seconds", type=float, default=None, help="Start offset (seconds)")
    p_t.add_argument("--duration-seconds", type=float, default=None, help="Limit duration (seconds)")

    p_a = sub.add_parser("analyze", help="transcript JSON -> clips metadata JSON (LLM)")
    p_a.add_argument("transcript", type=str)

    p_g = sub.add_parser("generate", help="video + clips metadata -> output clips")
    p_g.add_argument("video", type=str)
    p_g.add_argument("clips", type=str)

    p_f = sub.add_parser("full-pipeline", help="video -> transcript -> clips -> output clips")
    p_f.add_argument("video", type=str)

    p_y = sub.add_parser("yadisk-check", help="check Yandex Disk API access")
    p_y.add_argument("--path", type=str, default="/", help="Yandex Disk path (default: /)")
    p_y.add_argument("--limit", type=int, default=20, help="Max items to list (default: 20)")

    p_f = sub.add_parser("yadisk-files", help="list files via /resources/files")
    p_f.add_argument("--path-prefix", type=str, default="", help="Filter by path prefix (optional)")
    p_f.add_argument("--limit", type=int, default=20, help="Max items to list (default: 20)")
    p_f.add_argument("--media-type", type=str, default=None, help="Filter by media_type (e.g. video)")

    p_dl = sub.add_parser("yadisk-url", help="get direct download URL for Yandex Disk path")
    p_dl.add_argument("path", type=str, help="Yandex Disk path (e.g. disk:/Интервью/Андрей/IMG_0014.mov)")

    p_sync = sub.add_parser("sync-offset", help="estimate audio sync offset between two videos")
    p_sync.add_argument("ref", type=str, help="Reference video path (local or yadisk:...)")
    p_sync.add_argument("other", type=str, help="Other video path (local or yadisk:...)")
    p_sync.add_argument("--window-seconds", type=float, default=60.0, help="Window length (seconds)")
    p_sync.add_argument("--start-seconds", type=float, default=0.0, help="Window start (seconds)")
    p_sync.add_argument("--max-lag-seconds", type=float, default=10.0, help="Max lag to search (seconds)")
    p_sync.add_argument(
        "--use-end-window",
        action="store_true",
        help="Also estimate offset near the end to detect drift",
    )
    p_sync.add_argument(
        "--auto-search",
        action="store_true",
        help="Auto-search best window in a time range",
    )
    p_sync.add_argument("--search-start", type=float, default=0.0, help="Auto-search start (seconds)")
    p_sync.add_argument("--search-duration", type=float, default=600.0, help="Auto-search duration (seconds)")
    p_sync.add_argument("--step-seconds", type=float, default=30.0, help="Auto-search step (seconds)")
    p_sync.add_argument("--min-score", type=float, default=3.0, help="Min score for a reliable match")
    p_sync.add_argument(
        "--fast-seek-only",
        action="store_true",
        help="Use only fast seek (avoid slow accurate seek on streams)",
    )
    p_sync.add_argument("--cache-dir", type=str, default=None, help="Directory with cached full WAVs")
    p_sync.add_argument(
        "--fit-drift",
        action="store_true",
        help="Estimate linear drift (clock mismatch) over a range of windows",
    )
    p_sync.add_argument("--fit-start", type=float, default=None, help="Drift fit range start (seconds)")
    p_sync.add_argument("--fit-duration", type=float, default=None, help="Drift fit range duration (seconds)")
    p_sync.add_argument("--fit-step-seconds", type=float, default=None, help="Drift fit step (seconds)")
    p_sync.add_argument("--fit-min-points", type=int, default=4, help="Min points for drift fit")
    p_sync.add_argument("--fit-min-score", type=float, default=None, help="Min score per point for drift fit")
    p_sync.add_argument(
        "--fit-outlier-seconds",
        type=float,
        default=3.0,
        help="Drop points far from median offset (seconds, 0 to disable)",
    )
    p_sync.add_argument(
        "--refine-offset",
        action="store_true",
        help="Refine offset on the saved window by minimizing residual RMS",
    )
    p_sync.add_argument(
        "--refine-span-seconds",
        type=float,
        default=0.6,
        help="Refine search span around initial offset (seconds)",
    )
    p_sync.add_argument(
        "--refine-step-seconds",
        type=float,
        default=0.02,
        help="Refine step size (seconds)",
    )

    p_mix = sub.add_parser("sync-mix", help="make quick mix WAV to verify sync by ear")
    p_mix.add_argument("--ref-wav", type=str, default="storage/sync/sync_ref_best.wav")
    p_mix.add_argument("--other-wav", type=str, default="storage/sync/sync_other_best.wav")
    p_mix.add_argument("--offset-seconds", type=float, required=True)

    p_apply = sub.add_parser("sync-apply", help="apply offset by trimming (non-destructive)")
    p_apply.add_argument("ref", type=str, help="Reference video path (local or yadisk:...)")
    p_apply.add_argument("other", type=str, help="Other video path (local or yadisk:...)")
    p_apply.add_argument(
        "--offset-seconds",
        type=float,
        default=None,
        help="Offset seconds (positive = other delayed). If not set, read from --offset-json",
    )
    p_apply.add_argument(
        "--offset-json",
        type=str,
        default="storage/sync/offsets.json",
        help="Path to offsets.json (used when --offset-seconds is not set)",
    )
    p_apply.add_argument("--output-dir", type=str, default=None, help="Output directory for aligned files")
    p_apply.add_argument(
        "--reencode",
        action="store_true",
        help="Re-encode for accurate trims (slower). Default is stream copy.",
    )
    p_apply.add_argument(
        "--copy-unchanged",
        action="store_true",
        help="Also copy the stream even if trim is 0 (default: keep original path)",
    )
    p_apply.add_argument(
        "--audio-only",
        action="store_true",
        help="Do not trim video; only extract aligned WAVs for verification",
    )
    p_apply.add_argument(
        "--audio-window-seconds",
        type=float,
        default=None,
        help="Duration for aligned WAVs when --audio-only is set",
    )
    p_apply.add_argument("--cache-dir", type=str, default=None, help="Directory with cached full WAVs")
    p_apply.add_argument(
        "--auto-correct",
        action="store_true",
        help="Auto-correct offset using verification residual",
    )
    p_apply.add_argument(
        "--auto-correct-iters",
        type=int,
        default=2,
        help="Max iterations for auto-correct",
    )

    p_cache = sub.add_parser("sync-cache", help="cache full audio WAVs for faster sync")
    p_cache.add_argument("ref", type=str, help="Reference video path (local or yadisk:...)")
    p_cache.add_argument("other", type=str, help="Other video path (local or yadisk:...)")
    p_cache.add_argument("--cache-dir", type=str, default=None, help="Cache directory (default: storage/sync/cache)")
    p_cache.add_argument(
        "--fast-seek-only",
        action="store_true",
        help="Use only fast seek (avoid slow accurate seek on streams)",
    )
    p_cache.add_argument("--force", action="store_true", help="Recreate cache even if exists")
    p_cache.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="Cache in chunks instead of full audio (seconds)",
    )
    p_cache.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Fallback duration when it can't be probed (for chunk caching)",
    )
    p_apply.add_argument(
        "--fast-seek-only",
        action="store_true",
        help="Use only fast seek (avoid slow accurate seek on streams)",
    )
    p_apply.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify",
        help="Skip verification and check WAVs",
    )
    p_apply.set_defaults(verify=True)
    p_apply.add_argument(
        "--verify-start-seconds",
        type=float,
        default=0.0,
        help="Verify window start within aligned timeline (seconds)",
    )
    p_apply.add_argument("--verify-window-seconds", type=float, default=20.0, help="Verify window length")
    p_apply.add_argument("--verify-max-lag-seconds", type=float, default=2.0, help="Verify max lag")
    p_apply.add_argument(
        "--verify-tolerance-seconds",
        type=float,
        default=0.15,
        help="Max residual offset to consider aligned",
    )

    p_mc = sub.add_parser("multicam", help="auto-switch cameras by audio energy")
    p_mc.add_argument("--ref", type=str, default=None, help="Reference video (optional if offsets.json has ref)")
    p_mc.add_argument("--other", action="append", default=[], help="Other camera video (repeatable)")
    p_mc.add_argument("--offsets", action="append", default=[], help="Offsets JSON (repeatable)")
    p_mc.add_argument(
        "--ref-channel",
        type=str,
        default=None,
        help="Audio channel for ref (e.g. left/right/0/1)",
    )
    p_mc.add_argument(
        "--other-channel",
        action="append",
        default=[],
        help="Audio channel for other camera (repeatable)",
    )
    p_mc.add_argument(
        "--ref-bias-db",
        type=float,
        default=0.0,
        help="Bias in dB to favor ref camera selection",
    )
    p_mc.add_argument(
        "--other-bias-db",
        action="append",
        default=[],
        help="Bias in dB to favor other camera selection (repeatable)",
    )
    p_mc.add_argument(
        "--ref-vf",
        type=str,
        default=None,
        help="Video filter for ref camera (ffmpeg -vf string)",
    )
    p_mc.add_argument(
        "--other-vf",
        action="append",
        default=[],
        help="Video filter for other cameras (repeatable, ffmpeg -vf string)",
    )
    p_mc.add_argument("--start-seconds", type=float, default=0.0, help="Start time in ref timeline")
    p_mc.add_argument("--end-seconds", type=float, default=None, help="End time in ref timeline")
    p_mc.add_argument("--duration-seconds", type=float, default=None, help="Duration (overrides end)")
    p_mc.add_argument("--hop-seconds", type=float, default=0.1, help="Energy step (seconds)")
    p_mc.add_argument("--frame-ms", type=float, default=200.0, help="Energy window (ms)")
    p_mc.add_argument("--speech-margin-db", type=float, default=8.0, help="Speech margin over noise (dB)")
    p_mc.add_argument("--switch-margin-db", type=float, default=4.0, help="Required lead to switch (dB)")
    p_mc.add_argument("--duck-db", type=float, default=18.0, help="Duck inactive mics (dB)")
    p_mc.add_argument(
        "--audio-crossfade-seconds",
        type=float,
        default=0.08,
        help="Crossfade duration between mic switches (seconds)",
    )
    p_mc.add_argument(
        "--audio-sr",
        type=int,
        default=48000,
        help="Audio sample rate for mixing (e.g. 16000 or 48000)",
    )
    p_mc.add_argument(
        "--hard-gate",
        action="store_true",
        help="Mute inactive mics instead of ducking",
    )
    p_mc.add_argument(
        "--audio-from-sources",
        action="store_true",
        help="Build audio by extracting PCM directly from original sources per segment",
    )
    p_mc.add_argument(
        "--audio-encoder",
        type=str,
        default=("aac_at" if sys.platform == "darwin" else "aac"),
        choices=["aac", "aac_at", "alac", "alac_at", "pcm"],
        help="Audio encoder for final mux (aac/aac_at for lossy, alac/alac_at/pcm for lossless)",
    )
    p_mc.add_argument(
        "--audio-force-camera",
        type=str,
        default=None,
        help="Force audio to come from a specific camera (e.g. IMG_0041)",
    )
    p_mc.add_argument(
        "--speaker-main-camera",
        type=str,
        default=None,
        help="Keep this camera as primary video; use others only for reaction cutaways",
    )
    p_mc.add_argument(
        "--min-shot-seconds",
        type=float,
        default=None,
        help="Override minimum shot duration (seconds)",
    )
    p_mc.add_argument(
        "--reaction-every-seconds",
        type=float,
        default=0.0,
        help="Insert reaction cutaways every N seconds (0 disables)",
    )
    p_mc.add_argument(
        "--reaction-duration-seconds",
        type=float,
        default=2.4,
        help="Duration of reaction cutaway (seconds)",
    )
    p_mc.add_argument(
        "--reaction-min-score",
        type=float,
        default=0.0,
        help="Minimum other-camera score to insert reaction cutaway",
    )
    p_mc.add_argument(
        "--reaction-force-cutaway",
        action="store_true",
        help="Force reaction cutaway even when other-camera score is low",
    )
    p_mc.add_argument(
        "--reaction-search-window-seconds",
        type=float,
        default=1.2,
        help="Search window around reaction insertion target (seconds)",
    )
    p_mc.add_argument(
        "--reaction-search-step-seconds",
        type=float,
        default=0.4,
        help="Search step when selecting reaction insertion point (seconds)",
    )
    p_mc.add_argument(
        "--balance-max-dominant-share",
        type=float,
        default=1.0,
        help="Cap dominant camera share in clip (0..1, 1 disables balancing)",
    )
    p_mc.add_argument(
        "--balance-cutaway-seconds",
        type=float,
        default=2.4,
        help="Cutaway duration used by camera-balance pass (seconds)",
    )
    p_mc.add_argument(
        "--balance-min-gap-seconds",
        type=float,
        default=7.0,
        help="Minimal gap between inserted balance cutaways (seconds)",
    )
    p_mc.add_argument(
        "--fast-seek-only",
        action="store_true",
        help="Use fast seek when extracting audio from remote videos",
    )
    p_mc.add_argument(
        "--level-normalize",
        action="store_true",
        help="Normalize mic levels using median speech loudness",
    )
    p_mc.add_argument(
        "--level-max-db",
        type=float,
        default=6.0,
        help="Max per-mic level shift when normalizing (dB)",
    )
    p_mc.add_argument(
        "--level-target",
        type=str,
        default="median",
        choices=["median", "max", "min"],
        help="Target speech level when normalizing mic levels",
    )
    p_mc.add_argument(
        "--dominance-mode",
        type=str,
        default="energy",
        choices=["energy", "residual", "speaker"],
        help="Camera selection mode: energy, residual (leakage-suppressed), or speaker",
    )
    p_mc.add_argument(
        "--speaker-refs",
        type=str,
        default=None,
        help="Path to speaker samples JSON/YAML (optional if --speaker-ref-dir has WAVs)",
    )
    p_mc.add_argument(
        "--speaker-ref-dir",
        type=str,
        default=None,
        help="Directory with reference WAVs (default: storage/audio/speaker_refs)",
    )
    p_mc.add_argument(
        "--speaker-map",
        action="append",
        default=[],
        help="Map speaker label to camera (label=camera). Repeatable.",
    )
    p_mc.add_argument(
        "--speaker-window-seconds",
        type=float,
        default=1.5,
        help="Speaker-ID window length (seconds)",
    )
    p_mc.add_argument(
        "--speaker-lead-seconds",
        type=float,
        default=0.0,
        help="Shift speaker-ID window forward to reduce switch lag (seconds)",
    )
    p_mc.add_argument(
        "--speaker-min-sim",
        type=float,
        default=0.45,
        help="Minimum cosine similarity to accept speaker-ID",
    )
    p_mc.add_argument(
        "--speaker-margin",
        type=float,
        default=0.1,
        help="Min similarity margin between best and second speaker",
    )
    p_mc.add_argument(
        "--speaker-device",
        type=str,
        default="cpu",
        help="Device for speaker-ID encoder (cpu/cuda)",
    )
    p_mc.add_argument(
        "--subtitle-transcript",
        type=str,
        default=None,
        help="Transcript JSON for subtitles (absolute timeline)",
    )
    p_mc.add_argument(
        "--subtitle-max-chars",
        type=int,
        default=22,
        help="Max chars per subtitle line (wrap)",
    )
    p_mc.add_argument(
        "--subtitle-max-lines",
        type=int,
        default=2,
        help="Max subtitle lines (wrap/truncate)",
    )
    p_mc.add_argument(
        "--subtitle-preset",
        type=str,
        choices=["current", "readable", "safe"],
        default="readable",
        help="Subtitle style preset: current/readable/safe",
    )
    p_mc.add_argument(
        "--subtitle-align-mode",
        type=str,
        choices=["transcript", "rendered-audio"],
        default="transcript",
        help="Subtitle timing source: transcript timeline or rendered clip audio",
    )
    p_mc.add_argument(
        "--pause-accel-profile",
        type=str,
        choices=["none", "soft", "medium", "aggressive"],
        default="none",
        help="Pause-only acceleration profile (keeps speech at 1.0x)",
    )
    p_mc.add_argument(
        "--title-text",
        type=str,
        default=None,
        help="Title text to show at the start of the clip",
    )
    p_mc.add_argument(
        "--title-auto",
        action="store_true",
        help="Auto-generate title text from transcript for the current clip range",
    )
    p_mc.add_argument(
        "--title-seconds",
        type=float,
        default=1.6,
        help="Duration of title card (seconds)",
    )
    p_mc.add_argument(
        "--title-font",
        type=str,
        default=None,
        help="Font file path for title text (optional)",
    )
    p_mc.add_argument(
        "--title-fontsize",
        type=int,
        default=64,
        help="Font size for title text",
    )
    p_mc.add_argument(
        "--title-style",
        type=str,
        choices=["card", "overlay_center_dark"],
        default="card",
        help="Title style: card or overlay_center_dark",
    )
    p_mc.add_argument(
        "--cache-segments",
        action="store_true",
        help="Cache remote segments locally before cutting (faster for many cuts)",
    )
    p_mc.add_argument(
        "--cache-padding-seconds",
        type=float,
        default=2.0,
        help="Padding added around cached segments (seconds)",
    )
    p_mc.add_argument(
        "--cache-vf",
        type=str,
        default=None,
        help="Video filter to apply when creating cached segments",
    )
    p_mc.add_argument("--plan-only", action="store_true", help="Only write shotlist and mix")
    p_mc.add_argument("--output", type=str, default=None, help="Output video path (optional)")

    p_ig = sub.add_parser("instagram-upload", help="upload final reels via Instagram Graph API")
    p_ig.add_argument(
        "--upload-list-csv",
        type=str,
        default="storage/output/andrey_finals_review/UPLOAD_LIST.csv",
        help="CSV with upload order/index",
    )
    p_ig.add_argument(
        "--upload-pack-dir",
        type=str,
        default="storage/output/andrey_finals_review/upload_pack_latest",
        help="Directory with final upload-ready files",
    )
    p_ig.add_argument(
        "--progress-csv",
        type=str,
        default="storage/output/andrey_finals_review/UPLOAD_PROGRESS.csv",
        help="Upload progress log CSV",
    )
    p_ig.add_argument(
        "--summary-json",
        type=str,
        default="storage/output/andrey_finals_review/instagram_upload_summary.json",
        help="Summary JSON output path",
    )
    p_ig.add_argument(
        "--caption-template",
        type=str,
        default="",
        help="Optional caption template (supports {index},{camera},{start},{end},{filename})",
    )
    p_ig.add_argument(
        "--caption-overrides",
        type=str,
        default=None,
        help="Optional CSV/JSON with per-index captions",
    )
    p_ig.add_argument("--index-from", type=int, default=None, help="Start index (inclusive)")
    p_ig.add_argument("--index-to", type=int, default=None, help="End index (inclusive)")
    p_ig.add_argument("--indices", type=str, default=None, help="Explicit index list/ranges (e.g. 1,2,5-10)")
    p_ig.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Max uploads per run (0 = all selected)",
    )
    p_ig.add_argument(
        "--sleep-between-seconds",
        type=float,
        default=5.0,
        help="Pause between uploads",
    )
    p_ig.add_argument(
        "--poll-seconds",
        type=float,
        default=15.0,
        help="Container processing poll interval",
    )
    p_ig.add_argument(
        "--publish-timeout-seconds",
        type=float,
        default=900.0,
        help="Max wait for container readiness before publish",
    )
    p_ig.add_argument(
        "--api-version",
        type=str,
        default=None,
        help="Graph API version (default: INSTAGRAM_GRAPH_API_VERSION or v24.0)",
    )
    p_ig.add_argument(
        "--force-retry-failed",
        action="store_true",
        help="Retry rows already marked as failed in progress CSV",
    )
    p_ig.add_argument(
        "--dry-run",
        action="store_true",
        help="No API calls: run selection + QC and write skipped rows",
    )
    p_ig.add_argument("--qc-only", action="store_true", help="Only run QC inspection (no upload)")
    p_ig.add_argument(
        "--strict-qc",
        action="store_true",
        help="Skip upload when QC has warnings",
    )
    p_ig.add_argument(
        "--share-to-feed",
        dest="share_to_feed",
        action="store_true",
        help="Share reels to feed",
    )
    p_ig.add_argument(
        "--no-share-to-feed",
        dest="share_to_feed",
        action="store_false",
        help="Do not share reels to feed",
    )
    p_ig.set_defaults(share_to_feed=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    if args.command == "transcribe":
        video_arg = args.video
        if video_arg.startswith("yadisk:"):
            yd_path = video_arg[len("yadisk:") :].strip()
            if yd_path and not yd_path.startswith("disk:"):
                yd_path = f"disk:{yd_path}" if yd_path.startswith("/") else f"disk:/{yd_path}"
            try:
                video_arg = get_download_url(yd_path)
                console.print(f"[cyan]Yandex Disk URL[/cyan] {yd_path} -> {video_arg}")
            except YandexDiskError as e:
                console.print(f"[red]Yandex Disk error[/red]: {e}")
                raise SystemExit(1) from e
        cmd_transcribe(
            cfg,
            video_arg if video_arg.startswith("http") else Path(video_arg),
            start_seconds=args.start_seconds,
            duration_seconds=args.duration_seconds,
        )
    elif args.command == "analyze":
        cmd_analyze(cfg, Path(args.transcript))
    elif args.command == "generate":
        video_arg = args.video
        if video_arg.startswith("yadisk:"):
            yd_path = video_arg[len("yadisk:") :].strip()
            if yd_path and not yd_path.startswith("disk:"):
                yd_path = f"disk:{yd_path}" if yd_path.startswith("/") else f"disk:/{yd_path}"
            try:
                video_arg = get_download_url(yd_path)
                console.print(f"[cyan]Yandex Disk URL[/cyan] {yd_path} -> {video_arg}")
            except YandexDiskError as e:
                console.print(f"[red]Yandex Disk error[/red]: {e}")
                raise SystemExit(1) from e
        cmd_generate(
            cfg,
            video_arg if video_arg.startswith("http") else Path(video_arg),
            Path(args.clips),
        )
    elif args.command == "full-pipeline":
        video_arg = args.video
        if video_arg.startswith("yadisk:"):
            yd_path = video_arg[len("yadisk:") :].strip()
            if yd_path and not yd_path.startswith("disk:"):
                yd_path = f"disk:{yd_path}" if yd_path.startswith("/") else f"disk:/{yd_path}"
            try:
                video_arg = get_download_url(yd_path)
                console.print(f"[cyan]Yandex Disk URL[/cyan] {yd_path} -> {video_arg}")
            except YandexDiskError as e:
                console.print(f"[red]Yandex Disk error[/red]: {e}")
                raise SystemExit(1) from e
        cmd_full_pipeline(cfg, video_arg if video_arg.startswith("http") else Path(video_arg))
    elif args.command == "yadisk-check":
        cmd_yadisk_check(args.path, args.limit)
    elif args.command == "yadisk-files":
        cmd_yadisk_files(args.path_prefix, args.limit, args.media_type)
    elif args.command == "yadisk-url":
        try:
            url = get_download_url(args.path)
            console.print(url)
        except YandexDiskError as e:
            console.print(f"[red]Yandex Disk error[/red]: {e}")
            raise SystemExit(1) from e
    elif args.command == "sync-offset":
        try:
            out_path = sync_offset(
                cfg=cfg,
                ref_path=args.ref,
                other_path=args.other,
                window_seconds=args.window_seconds,
                start_seconds=args.start_seconds,
                max_lag_seconds=args.max_lag_seconds,
                use_end_window=args.use_end_window,
                auto_search=args.auto_search,
                search_start=args.search_start,
                search_duration=args.search_duration,
                step_seconds=args.step_seconds,
                min_score=args.min_score,
                fast_seek_only=args.fast_seek_only,
                fit_drift=args.fit_drift,
                fit_start=args.fit_start,
                fit_duration=args.fit_duration,
                fit_step_seconds=args.fit_step_seconds,
                fit_min_points=args.fit_min_points,
                fit_min_score=args.fit_min_score,
                fit_outlier_seconds=args.fit_outlier_seconds,
                refine_offset=args.refine_offset,
                refine_span_seconds=args.refine_span_seconds,
                refine_step_seconds=args.refine_step_seconds,
                cache_dir=args.cache_dir,
            )
            console.print(f"[green]Saved offsets[/green] -> {out_path}")
        except (YandexDiskError, RuntimeError) as e:
            console.print(f"[red]Sync error[/red]: {e}")
            raise SystemExit(1) from e
    elif args.command == "sync-mix":
        try:
            out_path = make_sync_mix(
                cfg=cfg,
                ref_wav=args.ref_wav,
                other_wav=args.other_wav,
                offset_seconds=args.offset_seconds,
            )
            console.print(f"[green]Saved mix[/green] -> {out_path}")
        except RuntimeError as e:
            console.print(f"[red]Sync error[/red]: {e}")
            raise SystemExit(1) from e
    elif args.command == "sync-apply":
        try:
            if args.offset_seconds is None:
                data = read_json(Path(args.offset_json))
                args.offset_seconds = float(data.get("offset_start"))
            report_path = apply_sync_trim(
                cfg=cfg,
                ref_path=args.ref,
                other_path=args.other,
                offset_seconds=float(args.offset_seconds),
                output_dir=args.output_dir,
                reencode=args.reencode,
                fast_seek_only=args.fast_seek_only,
                verify=args.verify,
                verify_window_seconds=args.verify_window_seconds,
                verify_max_lag_seconds=args.verify_max_lag_seconds,
                verify_tolerance_seconds=args.verify_tolerance_seconds,
                verify_start_seconds=args.verify_start_seconds,
                make_checks=args.verify,
                copy_unchanged=args.copy_unchanged,
                audio_only=args.audio_only,
                audio_window_seconds=args.audio_window_seconds,
                auto_correct=args.auto_correct,
                auto_correct_iters=args.auto_correct_iters,
                cache_dir=args.cache_dir,
            )
            console.print(f"[green]Saved aligned report[/green] -> {report_path}")
        except (YandexDiskError, RuntimeError, FileNotFoundError, ValueError) as e:
            console.print(f"[red]Sync error[/red]: {e}")
            raise SystemExit(1) from e
    elif args.command == "sync-cache":
        try:
            report = cache_full_audio(
                cfg=cfg,
                ref_path=args.ref,
                other_path=args.other,
                cache_dir=args.cache_dir,
                fast_seek_only=args.fast_seek_only,
                force=args.force,
                chunk_seconds=args.chunk_seconds,
                duration_seconds=args.duration_seconds,
            )
            cache_dir = Path(report["cache_dir"])
            out_path = cache_dir / "cache_report.json"
            write_json(out_path, report)
            console.print(f"[green]Saved cache report[/green] -> {out_path}")
        except (YandexDiskError, RuntimeError, FileNotFoundError) as e:
            console.print(f"[red]Sync error[/red]: {e}")
            raise SystemExit(1) from e
    elif args.command == "multicam":
        try:
            outputs = cmd_multicam(
                cfg,
                ref=args.ref,
                others=args.other,
                offsets_paths=args.offsets,
                ref_channel=args.ref_channel,
                other_channels=args.other_channel,
                ref_bias_db=args.ref_bias_db,
                other_bias_db=args.other_bias_db,
                ref_vf=args.ref_vf,
                other_vf=args.other_vf,
                start_seconds=args.start_seconds,
                end_seconds=args.end_seconds,
                duration_seconds=args.duration_seconds,
                hop_seconds=args.hop_seconds,
                frame_ms=args.frame_ms,
                speech_margin_db=args.speech_margin_db,
                switch_margin_db=args.switch_margin_db,
                duck_db=args.duck_db,
                audio_crossfade_seconds=args.audio_crossfade_seconds,
                hard_gate=args.hard_gate,
                audio_sample_rate=args.audio_sr,
                audio_from_sources=args.audio_from_sources,
                audio_encoder=args.audio_encoder,
                audio_force_camera=args.audio_force_camera,
                speaker_main_camera=args.speaker_main_camera,
                plan_only=args.plan_only,
                output=args.output,
                fast_seek_only=args.fast_seek_only,
                level_normalize=args.level_normalize,
                level_max_db=args.level_max_db,
                level_target=args.level_target,
                min_shot_seconds=args.min_shot_seconds,
                dominance_mode=args.dominance_mode,
                speaker_refs_path=args.speaker_refs,
                speaker_ref_dir=args.speaker_ref_dir,
                speaker_map_entries=args.speaker_map,
                speaker_window_seconds=args.speaker_window_seconds,
                speaker_lead_seconds=args.speaker_lead_seconds,
                speaker_min_similarity=args.speaker_min_sim,
                speaker_margin=args.speaker_margin,
                speaker_device=args.speaker_device,
                reaction_every_seconds=args.reaction_every_seconds,
                reaction_duration_seconds=args.reaction_duration_seconds,
                reaction_min_score=args.reaction_min_score,
                reaction_force_cutaway=args.reaction_force_cutaway,
                reaction_search_window_seconds=args.reaction_search_window_seconds,
                reaction_search_step_seconds=args.reaction_search_step_seconds,
                balance_max_dominant_share=args.balance_max_dominant_share,
                balance_cutaway_seconds=args.balance_cutaway_seconds,
                balance_min_gap_seconds=args.balance_min_gap_seconds,
                subtitle_transcript=args.subtitle_transcript,
                subtitle_max_chars=args.subtitle_max_chars,
                subtitle_max_lines=args.subtitle_max_lines,
                subtitle_preset=args.subtitle_preset,
                subtitle_align_mode=args.subtitle_align_mode,
                pause_accel_profile=args.pause_accel_profile,
                title_text=args.title_text,
                title_auto=args.title_auto,
                title_seconds=args.title_seconds,
                title_font=args.title_font,
                title_fontsize=args.title_fontsize,
                title_style=args.title_style,
                cache_segments=args.cache_segments,
                cache_padding_seconds=args.cache_padding_seconds,
                cache_vf=args.cache_vf,
            )
            console.print(f"[green]Saved shotlist[/green] -> {outputs['shotlist']}")
            console.print(f"[green]Saved mix[/green] -> {outputs['mix']}")
            if "speaker_timeline" in outputs:
                console.print(f"[green]Saved speaker timeline[/green] -> {outputs['speaker_timeline']}")
            if "video" in outputs:
                console.print(f"[green]Saved video[/green] -> {outputs['video']}")
        except (RuntimeError, YandexDiskError, FileNotFoundError) as e:
            console.print(f"[red]Multicam error[/red]: {e}")
            raise SystemExit(1) from e
    elif args.command == "instagram-upload":
        try:
            summary = run_instagram_upload(
                upload_list_csv=Path(args.upload_list_csv),
                upload_pack_dir=Path(args.upload_pack_dir),
                progress_csv=Path(args.progress_csv),
                summary_json=Path(args.summary_json) if args.summary_json else None,
                caption_template=args.caption_template,
                caption_overrides_path=Path(args.caption_overrides) if args.caption_overrides else None,
                index_from=args.index_from,
                index_to=args.index_to,
                indices=args.indices,
                batch_size=args.batch_size,
                sleep_between_seconds=args.sleep_between_seconds,
                dry_run=args.dry_run,
                qc_only=args.qc_only,
                strict_qc=args.strict_qc,
                force_retry_failed=args.force_retry_failed,
                share_to_feed=args.share_to_feed,
                poll_seconds=args.poll_seconds,
                publish_timeout_seconds=args.publish_timeout_seconds,
                api_version=args.api_version,
            )
            console.print(
                f"[green]Instagram upload finished[/green] "
                f"published={summary.get('published', 0)} "
                f"failed={summary.get('failed', 0)} "
                f"processed={summary.get('processed_now', 0)}"
            )
            console.print(f"[green]Progress[/green] -> {args.progress_csv}")
            if args.summary_json:
                console.print(f"[green]Summary[/green] -> {args.summary_json}")
        except (InstagramUploadError, FileNotFoundError, ValueError) as e:
            console.print(f"[red]Instagram upload error[/red]: {e}")
            raise SystemExit(1) from e
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
