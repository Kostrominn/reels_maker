from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Video(BaseModel):
    path: str
    duration: float | None = None
    camera: str | None = None


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] | None = None


class Transcript(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())

    video_path: str
    audio_path: str
    language: str | None = None
    model: str | None = None
    device: str | None = None
    compute_type: str | None = None

    segments: list[TranscriptSegment]

    meta: dict[str, Any] = Field(default_factory=dict)


class ClipCandidate(BaseModel):
    start: float
    end: float
    title: str | None = None
    reason: str | None = None
    score: float | None = None


class ClipAnalysis(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())

    transcript_path: str
    video_path: str
    llm_model: str | None = None

    candidates: list[ClipCandidate]
    meta: dict[str, Any] = Field(default_factory=dict)

