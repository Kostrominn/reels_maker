from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .audio_extractor import extract_wav_16k_mono
from .sync import resolve_video_source
from .utils import AppConfig, ensure_dirs, get_storage_paths


@dataclass(frozen=True)
class SpeakerSample:
    label: str
    video_path: str
    start_seconds: float
    end_seconds: float
    channel: str | int | None = None

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end_seconds) - float(self.start_seconds))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / denom)


def load_encoder(device: str = "cpu"):
    try:
        import torchaudio  # type: ignore
        # Some torchaudio builds (notably on macOS) may omit list_audio_backends.
        # SpeechBrain expects it at import time, so we provide a minimal shim.
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]

        import huggingface_hub
        from huggingface_hub.errors import RemoteEntryNotFoundError
        from requests.exceptions import HTTPError

        # SpeechBrain <-> huggingface_hub compatibility: map use_auth_token -> token.
        if "use_auth_token" not in getattr(huggingface_hub.hf_hub_download, "__code__", ()).co_varnames:
            orig = huggingface_hub.hf_hub_download

            def _hf_hub_download(*args, use_auth_token=None, **kwargs):  # type: ignore[override]
                if use_auth_token is not None and "token" not in kwargs:
                    kwargs["token"] = use_auth_token
                if kwargs.get("filename") == "custom.py":
                    # SpeechBrain treats missing custom.py as optional.
                    raise HTTPError("404 Client Error")
                try:
                    return orig(*args, **kwargs)
                except RemoteEntryNotFoundError as e:
                    raise HTTPError("404 Client Error") from e

            huggingface_hub.hf_hub_download = _hf_hub_download  # type: ignore[assignment]

        from speechbrain.pretrained import EncoderClassifier  # type: ignore
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Не найден speechbrain/torch. Установка: pip install speechbrain torch torchaudio"
        ) from e

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )


def _embed_wav(encoder, wav_path: str | Path) -> np.ndarray:
    import torch

    wav: torch.Tensor
    sr: int

    try:
        import torchaudio  # type: ignore

        wav, sr = torchaudio.load(str(wav_path))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
    except Exception:
        from scipy.io import wavfile
        from scipy.signal import resample_poly

        sr, data = wavfile.read(str(wav_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype.kind in {"i", "u"}:
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / float(max_val)
        else:
            data = data.astype(np.float32)
        if sr != 16000:
            data = resample_poly(data, 16000, sr).astype(np.float32)
            sr = 16000
        wav = torch.from_numpy(data).unsqueeze(0)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    embeddings = encoder.encode_batch(wav)
    emb = embeddings.squeeze().detach().cpu().numpy()
    if emb.ndim > 1:
        emb = emb.reshape(-1)
    return emb.astype(np.float32)


def embed_audio(encoder, audio: np.ndarray, sr: int) -> np.ndarray:
    import torch

    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    audio = audio.astype(np.float32, copy=False)

    wav = torch.from_numpy(audio).unsqueeze(0)
    if sr != 16000:
        try:
            import torchaudio  # type: ignore

            wav = torchaudio.functional.resample(wav, sr, 16000)
        except Exception:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, 16000, sr).astype(np.float32)
            wav = torch.from_numpy(audio).unsqueeze(0)

    with torch.no_grad():
        embeddings = encoder.encode_batch(wav)
    emb = embeddings.squeeze().detach().cpu().numpy()
    if emb.ndim > 1:
        emb = emb.reshape(-1)
    return emb.astype(np.float32)


def extract_sample_wav(cfg: AppConfig, sample: SpeakerSample, out_dir: Path) -> Path:
    ensure_dirs(out_dir)
    name = f"{sample.label}_{sample.start_seconds:.2f}_{sample.end_seconds:.2f}.wav"
    out_path = out_dir / name
    if out_path.exists():
        try:
            if out_path.stat().st_size > 1024:
                return out_path
        except OSError:
            pass
        out_path.unlink(missing_ok=True)
    src = resolve_video_source(sample.video_path)
    is_remote = isinstance(src, str) and src.startswith("http")
    extract_wav_16k_mono(
        src,
        out_path,
        start_seconds=sample.start_seconds,
        duration_seconds=sample.duration,
        accurate_seek=not is_remote,
        channel=sample.channel,
    )
    return out_path


def build_reference_embeddings(
    cfg: AppConfig,
    samples: Iterable[SpeakerSample],
    *,
    device: str = "cpu",
    encoder=None,
) -> dict[str, np.ndarray]:
    if encoder is None:
        encoder = load_encoder(device=device)
    storage = get_storage_paths(cfg)
    out_dir = storage.audio / "speaker_refs"
    per_label: dict[str, list[np.ndarray]] = {}

    for s in samples:
        wav = extract_sample_wav(cfg, s, out_dir)
        emb = _embed_wav(encoder, wav)
        per_label.setdefault(s.label, []).append(emb)

    refs: dict[str, np.ndarray] = {}
    for label, embs in per_label.items():
        if not embs:
            continue
        avg = np.mean(np.stack(embs, axis=0), axis=0)
        # Normalize for cosine
        avg = avg / (np.linalg.norm(avg) + 1e-9)
        refs[label] = avg.astype(np.float32)
    return refs


def build_reference_embeddings_from_wavs(
    ref_dir: str | Path,
    *,
    device: str = "cpu",
    encoder=None,
) -> dict[str, np.ndarray]:
    if encoder is None:
        encoder = load_encoder(device=device)
    ref_path = Path(ref_dir)
    if not ref_path.exists():
        raise RuntimeError(f"Каталог speaker refs не найден: {ref_path}")

    per_label: dict[str, list[np.ndarray]] = {}
    wavs = sorted(ref_path.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"В каталоге нет WAV файлов: {ref_path}")

    for wav in wavs:
        stem = wav.stem
        if stem.startswith(("tmp_", "temp_")):
            continue
        if "_" in stem:
            label = stem.split("_", 1)[0]
        else:
            label = stem
        if not label:
            continue
        emb = _embed_wav(encoder, wav)
        per_label.setdefault(label, []).append(emb)

    refs: dict[str, np.ndarray] = {}
    for label, embs in per_label.items():
        if not embs:
            continue
        avg = np.mean(np.stack(embs, axis=0), axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-9)
        refs[label] = avg.astype(np.float32)
    if not refs:
        raise RuntimeError(f"Не удалось собрать референсы из WAV: {ref_path}")
    return refs


def load_speaker_samples(path: str | Path) -> list[SpeakerSample]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("samples") or data.get("refs") or data.get("segments") or []
    else:
        items = []

    samples: list[SpeakerSample] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        video_path = item.get("video_path")
        start = item.get("start_seconds")
        end = item.get("end_seconds")
        if label is None or video_path is None or start is None or end is None:
            raise RuntimeError(f"Некорректный speaker sample: {item}")
        channel = item.get("channel")
        samples.append(
            SpeakerSample(
                label=str(label),
                video_path=str(video_path),
                start_seconds=float(start),
                end_seconds=float(end),
                channel=channel,
            )
        )

    if not samples:
        raise RuntimeError(f"В файле нет speaker samples: {path}")
    return samples


def score_samples(
    cfg: AppConfig,
    samples: Iterable[SpeakerSample],
    refs: dict[str, np.ndarray],
    *,
    device: str = "cpu",
) -> list[dict[str, float | str]]:
    encoder = load_encoder(device=device)
    storage = get_storage_paths(cfg)
    out_dir = storage.audio / "speaker_refs"
    rows: list[dict[str, float | str]] = []

    for s in samples:
        wav = extract_sample_wav(cfg, s, out_dir)
        emb = _embed_wav(encoder, wav)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        row: dict[str, float | str] = {"label": s.label, "segment": f"{s.start_seconds:.2f}-{s.end_seconds:.2f}"}
        for lbl, ref in refs.items():
            row[f"sim_{lbl}"] = _cosine(emb, ref)
        rows.append(row)
    return rows
