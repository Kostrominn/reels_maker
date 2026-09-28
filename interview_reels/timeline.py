"""From plan time to reel time.

A plan is written in *transcript time*: the timeline of the reference audio
track that every subtitle was transcribed from. Each source (camera) has an
``advance`` = transcript time − time in that source's own file for the same
moment. The audio of every clip is read from the source's ``audio`` file, which
is in transcript time, so the picture can switch cameras anywhere, even in the
middle of someone's line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FPS = 30


@dataclass
class Timeline:
    clips: list = field(default_factory=list)    # [source, a, b] in the source's own time, frame-rounded
    cues: list = field(default_factory=list)     # [start, end, text] in reel time
    insets: list = field(default_factory=list)   # [start, end, path, x, y] in reel time
    duration: float = 0.0


def build(plan: dict, fps: int = FPS) -> Timeline:
    """Lay the clips end to end and move cues and insets into reel time.

    A cue belongs to a source: it is shown only while a clip of that source is
    on screen. Insets (pictures, stickers) are not tied to a camera."""
    tl = Timeline()
    for source, a, b in plan['clips']:
        adv = plan['sources'][source]['advance']
        tl.clips.append([source, round((a - adv) * fps) / fps, round((b - adv) * fps) / fps])
    offset = 0.0
    for source, a, b in tl.clips:
        adv = plan['sources'][source]['advance']
        for k, s, e, txt in plan['cues']:
            if k != source:
                continue
            x = max(a, s - adv)
            y = min(b, e - adv)
            if y - x > .07:
                tl.cues.append([offset + x - a, offset + y - a, txt])
        offset += b - a
    tl.cues.sort(key=lambda cue: cue[0])
    offset = 0.0
    for source, a, b in tl.clips:
        adv = plan['sources'][source]['advance']
        for s, e, path, x, y in plan.get('insets') or []:
            lo = max(a, s - adv)
            hi = min(b, e - adv)
            if hi - lo > .07:
                tl.insets.append([offset + lo - a, offset + hi - a, path, x, y])
        offset += b - a
    tl.duration = offset
    return tl


def srt_time(t: float) -> str:
    m = round(t * 1000)
    return f'{m // 3600000:02}:{m // 60000 % 60:02}:{m // 1000 % 60:02},{m % 1000:03}'


def srt(cues) -> str:
    return '\n\n'.join(f'{n + 1}\n{srt_time(a)} --> {srt_time(b)}\n{txt}'
                       for n, (a, b, txt) in enumerate(cues)) + '\n'


def parse_srt_time(s: str) -> float:
    h, m, rest = s.strip().split(':')
    sec, ms = rest.split(',')
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000
