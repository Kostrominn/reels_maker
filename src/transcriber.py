from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import Transcript, TranscriptSegment, Word
from .audio_extractor import slice_wav
from .utils import AppConfig, probe_duration_seconds, seconds_to_timestamp


def _import_faster_whisper():
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Не найден faster-whisper. Установите зависимости: pip install -r requirements.txt"
        ) from e
    return WhisperModel


def _import_gigaam():
    try:
        import gigaam  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Не найден пакет gigaam. Установка: "
            "pip install 'gigaam @ git+https://github.com/salute-developers/GigaAM.git'"
        ) from e
    return gigaam


def _stringify_video_path(video_path: str | Path) -> str:
    if isinstance(video_path, str) and video_path.startswith("http"):
        return video_path
    return str(Path(video_path))


class Transcriber:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._fw_model = None
        self._gigaam_model = None

    def _get_fw_model(self):
        if self._fw_model is None:
            WhisperModel = _import_faster_whisper()
            tcfg = self.cfg.transcription
            kwargs: dict[str, Any] = {
                "device": tcfg.device,
                "compute_type": tcfg.compute_type,
            }
            if tcfg.cpu_threads and int(tcfg.cpu_threads) > 0:
                kwargs["cpu_threads"] = int(tcfg.cpu_threads)
            self._fw_model = WhisperModel(tcfg.model, **kwargs)
        return self._fw_model

    def _get_gigaam_model(self):
        if self._gigaam_model is None:
            gigaam = _import_gigaam()
            tcfg = self.cfg.transcription
            # gigaam supports short names: ctc/rnnt/e2e_ctc/e2e_rnnt -> maps to v3_*
            self._gigaam_model = gigaam.load_model(
                tcfg.model,
                device=tcfg.device,
            )
        return self._gigaam_model

    def transcribe_audio(
        self,
        *,
        audio_path: str | Path,
        video_path: str | Path,
        word_timestamps: bool = False,
    ) -> Transcript:
        tcfg = self.cfg.transcription
        backend = (tcfg.backend or "faster-whisper").strip().lower()

        if backend in {"gigaam", "gigaam-v3"}:
            return self._transcribe_gigaam(audio_path=audio_path, video_path=video_path)

        model = self._get_fw_model()

        segments_iter, info = model.transcribe(
            str(audio_path),
            language=tcfg.language,
            word_timestamps=word_timestamps,
            vad_filter=True,
        )

        segments: list[TranscriptSegment] = []
        for s in segments_iter:
            words = None
            if word_timestamps and getattr(s, "words", None):
                words = [
                    Word(
                        start=w.start,
                        end=w.end,
                        word=w.word,
                        probability=getattr(w, "probability", None),
                    )
                    for w in s.words
                ]
            segments.append(
                TranscriptSegment(
                    start=float(s.start),
                    end=float(s.end),
                    text=(s.text or "").strip(),
                    words=words,
                )
            )

        meta: dict[str, Any] = {
            "duration": getattr(info, "duration", None),
            "language_probability": getattr(info, "language_probability", None),
            "segments_count": len(segments),
            "hint": {
                "first_segment_start": seconds_to_timestamp(segments[0].start)
                if segments
                else None,
            },
        }

        return Transcript(
            video_path=_stringify_video_path(video_path),
            audio_path=str(Path(audio_path)),
            language=tcfg.language,
            model=tcfg.model,
            device=tcfg.device,
            compute_type=tcfg.compute_type,
            segments=segments,
            meta=meta,
        )

    def _transcribe_gigaam(self, *, audio_path: str | Path, video_path: str | Path) -> Transcript:
        model = self._get_gigaam_model()
        tcfg = self.cfg.transcription

        duration = probe_duration_seconds(audio_path)
        segments: list[TranscriptSegment] = []
        meta: dict[str, Any] = {
            "backend": "gigaam",
            "duration": duration,
        }

        # Prefer official longform if available and audio is long.
        if (
            tcfg.gigaam_use_longform
            and duration is not None
            and duration > 25.0
            and hasattr(model, "transcribe_longform")
        ):
            try:
                utterances = model.transcribe_longform(str(audio_path))
                for utt in utterances:
                    b = utt.get("boundaries")
                    if not b or not isinstance(b, (list, tuple)) or len(b) != 2:
                        continue
                    start, end = float(b[0]), float(b[1])
                    text = (utt.get("transcription") or "").strip()
                    if not text:
                        continue
                    segments.append(TranscriptSegment(start=start, end=end, text=text, words=None))
                meta["mode"] = "longform"
                meta["segments_count"] = len(segments)
                return Transcript(
                    video_path=_stringify_video_path(video_path),
                    audio_path=str(Path(audio_path)),
                    language=tcfg.language,
                    model=tcfg.model,
                    device=tcfg.device,
                    compute_type=tcfg.compute_type,
                    segments=segments,
                    meta=meta,
                )
            except Exception as e:  # noqa: BLE001
                # Fall back to chunking if longform deps aren't installed / HF token missing.
                meta["longform_error"] = str(e)

        # Fallback: fixed-size chunks (<=25s) with simple timestamping.
        chunk = float(tcfg.gigaam_chunk_seconds or 20.0)
        if chunk <= 0 or chunk > 25:
            chunk = 20.0

        if duration is None:
            # If we can't probe duration, try a single short transcribe (may fail if long).
            text = str(model.transcribe(str(audio_path))).strip()
            segments = [TranscriptSegment(start=0.0, end=0.0, text=text, words=None)] if text else []
            meta["mode"] = "single"
        else:
            tmp_dir = Path(audio_path).parent / "tmp_chunks"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            t = 0.0
            idx = 0
            while t < duration:
                idx += 1
                dur = min(chunk, max(0.0, duration - t))
                if dur <= 0:
                    break
                tmp = tmp_dir / f"{Path(audio_path).stem}_chunk_{idx:05d}.wav"
                slice_wav(audio_path, tmp, start_seconds=t, duration_seconds=dur)
                text = str(model.transcribe(str(tmp))).strip()
                if text:
                    segments.append(TranscriptSegment(start=t, end=t + dur, text=text, words=None))
                t += dur
            meta["mode"] = "chunked"
            meta["chunk_seconds"] = chunk
            meta["segments_count"] = len(segments)

        meta["hint"] = {
            "first_segment_start": seconds_to_timestamp(segments[0].start) if segments else None,
        }

        return Transcript(
            video_path=_stringify_video_path(video_path),
            audio_path=str(Path(audio_path)),
            language=tcfg.language,
            model=tcfg.model,
            device=tcfg.device,
            compute_type=tcfg.compute_type,
            segments=segments,
            meta=meta,
        )
