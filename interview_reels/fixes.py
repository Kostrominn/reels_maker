"""Mechanical fixes of a plan: cuts nudged into real pauses, captions rewrapped,
per-clip loudness trims. Always look at the plan after them: autofix does not
see overlapping clips."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import media
from .project import Project, write_json
from .timeline import build

# -- cuts and captions ----------------------------------------------------------------

_smooth: dict = {}


def _levels(path):
    """10 ms levels smoothed over 100 ms, and the recording's speech level."""
    if path not in _smooth:
        db = media.levels_db(media.pcm(path, 16000), 160)
        _smooth[path] = (np.convolve(db, np.ones(10) / 10, mode='same'), float(np.percentile(db, 88)))
    return _smooth[path]


def quietest(path, t, lo, hi):
    sm, speech = _levels(path)
    i0, i1 = max(0, int((t + lo) * 100)), int((t + hi) * 100)
    k = i0 + int(np.argmin(sm[i0:i1]))
    return k / 100, sm[k], speech, sm[int(t * 100)]


def nudge_cuts(plan: dict, resolve=lambda p: p) -> list[str]:
    """A start may move up to 0.5 s earlier or 0.15 s later, an end 0.15 s earlier
    or 0.5 s later — outward, so a word is never shortened. Seamless joins
    (same audio, end == next start) are left alone."""
    log = []
    clips = plan['clips']
    for n, c in enumerate(clips):
        k, a, b = c
        audio = str(resolve(plan['sources'][k]['audio']))
        prev_seam = n > 0 and plan['sources'][clips[n - 1][0]]['audio'] == plan['sources'][k]['audio'] \
            and abs(clips[n - 1][2] - a) < 0.02
        next_seam = n + 1 < len(clips) and plan['sources'][clips[n + 1][0]]['audio'] == plan['sources'][k]['audio'] \
            and abs(clips[n + 1][1] - b) < 0.02
        if not prev_seam:
            t, lv, sp, was = quietest(audio, a, -0.5, 0.15)
            if was > sp - 9 and lv < was:
                log.append(f'{k} начало {a:.2f} -> {t:.2f} ({was:.0f} -> {lv:.0f} dB, речь {sp:.0f})')
                c[1] = round(t, 2)
        if not next_seam:
            t, lv, sp, was = quietest(audio, b, -0.15, 0.5)
            if was > sp - 9 and lv < was:
                log.append(f'{k} конец  {b:.2f} -> {t:.2f} ({was:.0f} -> {lv:.0f} dB, речь {sp:.0f})')
                c[2] = round(t, 2)
    return log


LIMIT = 850      # widest caption line, px


def wrap(text: str, font) -> str:
    """Fewest lines first, then the most even split; no dangling «и», «всё»."""
    draw = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    width = lambda line: draw.textlength(line, font=font)
    words = text.replace('\n', ' ').split(' ')
    best = None
    for k in (1, 2, 3, 4):
        for cuts in itertools.combinations(range(1, len(words)), k - 1):
            parts = [' '.join(words[a:b]) for a, b in zip((0,) + cuts, cuts + (len(words),))]
            if not all(width(x) <= LIMIT for x in parts):
                continue
            if any(len(x.split()) == 1 and len(x) <= 4 for x in parts[1:]):
                continue
            score = max(width(x) for x in parts) - 0.3 * min(width(x) for x in parts)
            if best is None or score < best[0]:
                best = (score, parts)
        if best:
            return '\n'.join(best[1])
    return text


def rewrap(plan: dict, font) -> list[str]:
    log = []
    draw = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    for c in plan['cues']:
        if not all(draw.textlength(line, font=font) <= LIMIT for line in c[3].split('\n')):
            new = wrap(c[3], font)
            log.append(f'перенос: {c[3]!r} -> {new!r}')
            c[3] = new
    return log


def autofix(project: Project, ids=None, draft=None):
    """Fix plans in plans.json (by id) or one draft plan file."""
    font = project.style.caption_font()
    if draft:
        path = Path(draft)
        plan = json.loads(path.read_text())
        for line in nudge_cuts(plan, project.path) + rewrap(plan, font):
            print(line)
        write_json(path, plan)
        return
    plans = project.raw_plans()
    for plan in plans:
        if ids and plan['id'] not in ids:
            continue
        for line in nudge_cuts(plan, project.path) + rewrap(plan, font):
            print(plan['id'], line)
    project.save_plans(plans)


# -- loudness -----------------------------------------------------------------------------

CLAMP = 5.0
STRENGTH = 0.75      # partial correction keeps the dialogue from sounding flattened


def balance_levels(project: Project, ids=None):
    """Even out speech level between clips of a reel: the same room is recorded
    by several microphones, so a line spoken next to one of them is several
    decibels louder there. Writes ``clip_gains`` into plans.json.

    ±5 dB at most: short lines of a person sitting at their own microphone can
    be 6–9 dB louder — set those gains by hand (``clip-levels``) and do not run
    this again for that reel, it would overwrite them."""
    plans = project.raw_plans()
    for p in plans:
        if ids and p['id'] not in ids:
            continue
        per_clip = []
        for key, a, b in p['clips']:
            src = p['sources'][key]
            # the audio file is in transcript time, the plan's clips too
            db = media.file_levels(project.path(src['audio']), 16000, 800)
            seg = db[int(a / 0.05):int(b / 0.05)]
            speech = seg[seg > seg.max() - 16] if len(seg) else seg
            per_clip.append(float(np.median(speech)) if len(speech) else None)
        known = [v for v in per_clip if v is not None]
        if not known:
            continue
        target = float(np.median(known))
        gains = []
        for v in per_clip:
            g = 0.0 if v is None else max(-CLAMP, min(CLAMP, (target - v) * STRENGTH))
            gains.append(round(g, 2) if abs(g) >= 0.3 else 0.0)
        for src in p['sources'].values():
            src.pop('gain_db', None)
        p['clip_gains'] = gains
        print(f'{p["id"]}: цель {target:.1f} dB, разброс исходно {max(known) - min(known):.1f} dB · '
              + ' '.join(f'{g:+.1f}' for g in gains))
    project.save_plans(plans)
    print('поправки записаны в plans.json')


def clip_levels(project: Project, ident: str):
    """Speech level of every clip in a finished reel."""
    p = project.plan(ident)
    tl = build(p)
    video = project.reel_path(p)
    db = media.levels_db(media.pcm(video), 800)
    t = 0.0
    gains = p.get('clip_gains') or [0] * len(tl.clips)
    for n, ((k, a, b), g) in enumerate(zip(p['clips'], gains)):
        d = tl.clips[n][2] - tl.clips[n][1]
        seg = db[int(t / .05):int((t + d) / .05)]
        sp = seg[seg > seg.max() - 18] if len(seg) else seg
        level = float(np.median(sp)) if len(sp) else float('nan')
        print(f'{n:2} {k:3} {a:7.2f}-{b:7.2f} @{t:5.1f}s gain {g:+5.1f} level {level:6.1f}')
        t += d
    print('длительность', round(t, 2), video.name)
