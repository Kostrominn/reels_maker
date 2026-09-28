"""Render one reel from its plan.

Picture: each clip is trimmed from its camera by frame, cameras cut from the
plan, two consecutive clips of one camera get slightly different framing so an
internal cut does not read as a glitch, then one colour grade for the whole reel.
Sound: every clip is read from the source's audio file (transcript time), with a
4 ms fade at each edge, optional per-clip gain, then a two-pass loudness
normalisation to −16 LUFS. Graphics (title, subtitles, insets) are drawn with
Pillow into a transparent overlay and composited on top.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import media
from .project import Project
from .timeline import FPS, Timeline, build, srt

W, H = 1080, 1920
TITLE_HOLD, TITLE_FADE = 3.55, 3.85      # the title stays, then fades out in 0.3 s
CAPTION_BASELINE = 1434                  # baseline of the last caption line
CAPTION_LINE = 72


class TextOverflow(ValueError):
    """A subtitle or a title does not fit the frame."""


# -- graphics -----------------------------------------------------------------

def title_layer(plan: dict, style) -> Image.Image:
    arr = np.zeros((H, W, 4), dtype=np.uint8)
    arr[:410, :, :3] = style.shade
    arr[:410, :, 3] = (207 * np.clip(1 - np.arange(410) / 410, 0, 1) ** 1.65).astype(np.uint8)[:, None]
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    titlefont = style.title_font()
    for row, line in enumerate(plan['kicker'].split('\n')):
        d.text((80, 48 + row * 35), line, font=style.kicker_font(), fill=style.accent, anchor='lt')
    for n, line in enumerate(plan['title']):
        if d.textlength(line, font=titlefont) >= 900:
            raise TextOverflow(f'заголовок не влезает: {line!r}')
        d.text((80, 92 + n * 60), line, font=titlefont, fill=style.white if n == 0 else style.accent, anchor='lt')
    return im


def accent_pattern(plan: dict):
    # an accent starts at a word boundary and colours the whole word, so a stem
    # like «плюсник» covers «плюсника» and «пи» never lights up inside «олимпиаду»
    terms = sorted(plan.get('accent') or [], key=len, reverse=True)
    return r'(?<!\w)(?:' + '|'.join(re.escape(t) for t in terms) + r')\w*' if terms else None


def caption_layer(plan: dict, txt: str, style):
    cap = style.caption_font()
    im = Image.new('RGBA', (W, H))
    d = ImageDraw.Draw(im)
    lines = txt.split('\n')
    boxes = []
    for n, line in enumerate(lines):
        yy = CAPTION_BASELINE - (len(lines) - 1 - n) * CAPTION_LINE
        x = 515 - d.textlength(line, font=cap) / 2
        boxes.append(d.textbbox((x, yy), line, font=cap, anchor='ls'))
    left = min(b[0] for b in boxes) - 24
    right = max(b[2] for b in boxes) + 24
    top = min(b[1] for b in boxes) - 18
    bottom = max(b[3] for b in boxes) + 18
    if left < 60 or right > 965:
        raise TextOverflow(f'субтитр не влезает: {txt!r} ({left:.0f}…{right:.0f})')
    d.rounded_rectangle((left, top, right, bottom), radius=14, fill=(*style.plate, 187))
    pattern = accent_pattern(plan)
    for n, line in enumerate(lines):
        yy = CAPTION_BASELINE - (len(lines) - 1 - n) * CAPTION_LINE
        x = 515 - d.textlength(line, font=cap) / 2
        last = 0
        if pattern:
            for match in re.finditer(pattern, line, flags=re.I):
                plain = line[last:match.start()]
                d.text((x, yy), plain, font=cap, anchor='ls', fill=style.white)
                x += d.textlength(plain, font=cap)
                term = match.group(0)
                d.text((x, yy), term, font=cap, anchor='ls', fill=style.accent)
                x += d.textlength(term, font=cap)
                last = match.end()
        d.text((x, yy), line[last:], font=cap, anchor='ls', fill=style.white)
    return im, [left, top, right, bottom]


def make_overlay(project: Project, plan: dict, tl: Timeline) -> Path:
    """Transparent overlay (QuickTime Animation) with the title, subtitles and
    insets: one PNG per interval between graphic changes, joined with ffconcat."""
    style = project.style
    tmp = project.tmp(plan['id'])
    nframes = round(tl.duration * FPS)
    edges = {0, nframes, round(TITLE_HOLD * FPS), round(TITLE_FADE * FPS)}
    for a, b, _ in tl.cues:
        edges.update([max(0, round(a * FPS)), min(nframes, round(b * FPS))])
    for a, b, *_ in tl.insets:
        edges.update([max(0, round(a * FPS)), min(nframes, round(b * FPS))])
    edges.update(range(round(TITLE_HOLD * FPS), round(TITLE_FADE * FPS) + 1))
    edges = sorted(x for x in edges if 0 <= x <= nframes)
    title = title_layer(plan, style)
    caps = {txt: caption_layer(plan, txt, style) for _, _, txt in tl.cues}
    pics = {path: Image.open(path).convert('RGBA') for _, _, path, _, _ in tl.insets}
    graphics = tmp / 'graphics'
    graphics.mkdir(exist_ok=True)
    entries, bounds, file = [], [], None
    for n, (a, b) in enumerate(zip(edges, edges[1:])):
        if b <= a:
            continue
        t = (a + b) / 2 / FPS
        im = Image.new('RGBA', (W, H))
        if t < TITLE_FADE:
            layer = title.copy()
            if t > TITLE_HOLD:
                layer.putalpha(layer.getchannel('A').point(
                    lambda v: round(v * (TITLE_FADE - t) / (TITLE_FADE - TITLE_HOLD))))
            im = Image.alpha_composite(im, layer)
        for s, e, path, x, y in tl.insets:
            if s - .5 / FPS <= t < e + .5 / FPS:
                im.alpha_composite(pics[path], (x, y))
        for s, e, txt in tl.cues:
            if s - .5 / FPS <= t < e + .5 / FPS:
                im = Image.alpha_composite(im, caps[txt][0])
                bounds.append(dict(start=s, end=e, text=txt, box=caps[txt][1]))
        file = graphics / f'{n:03}.png'
        im.save(file)
        entries += ['file ' + str(file), f'duration {(b - a) / FPS:.9f}']
    entries += ['file ' + str(file)]
    concat = graphics / 'frames.ffconcat'
    concat.write_text('ffconcat version 1.0\n' + '\n'.join(entries) + '\n')
    overlay = tmp / 'overlay.mov'
    media.run([media.FFMPEG, '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', concat,
               '-vf', 'fps=30,tpad=stop_mode=clone:stop_duration=2', '-frames:v', str(nframes),
               '-c:v', 'qtrle', '-pix_fmt', 'argb', '-threads', '2', overlay])
    (tmp / 'caption_bounds.json').write_text(json.dumps(bounds, ensure_ascii=False, indent=2))
    return overlay


# -- sound ----------------------------------------------------------------------

def edit_audio(project: Project, plan: dict, tl: Timeline) -> Path:
    keys = list(plan['sources'])
    index = {k: n for n, k in enumerate(keys)}
    inputs = []
    for k in keys:
        inputs += ['-i', plan['sources'][k]['audio']]
    gains = plan.get('clip_gains') or [0.0] * len(tl.clips)
    g = []
    for n, (k, a, b) in enumerate(tl.clips):
        src = plan['sources'][k]
        adv = src['advance']
        d = b - a
        # one room mic records both people, so whoever sits closer to it comes out
        # louder; a static per-source trim evens the voices out across cuts
        db = src.get('gain_db', 0.0) + gains[n]
        gain = f',volume={db:+.2f}dB' if abs(db) >= 0.05 else ''
        g.append(f'[{index[k]}:a]atrim=start={a + adv:.6f}:end={b + adv:.6f},asetpts=PTS-STARTPTS{gain},'
                 f'afade=t=in:st=0:d=0.004,afade=t=out:st={d - .004:.6f}:d=0.004[a{n}]')
    finish = ',acompressor=threshold=0.16:ratio=2:attack=15:release=150:makeup=1' if plan.get('compress_audio') else ''
    # single-mic outdoor takes peak near 0 dBTP at -25 LUFS; without a ceiling the
    # linear loudnorm below cannot lift them to -16 and the reel ends up 2 dB quiet
    if plan.get('limit_audio'):
        finish += f',alimiter=limit={plan["limit_audio"]}:attack=2:release=60:level=0'
    g.append(''.join(f'[a{n}]' for n in range(len(tl.clips))) +
             f'concat=n={len(tl.clips)}:v=0:a=1,highpass=f=65,lowpass=f=14000{finish}[a]')
    audio = project.tmp(plan['id']) / 'edit.wav'
    media.run([media.FFMPEG, '-v', 'error', '-y', *inputs, '-filter_complex', ';'.join(g),
               '-map', '[a]', '-c:a', 'pcm_s24le', '-ar', '48000', audio])
    return audio


def loudnorm_filter(project: Project, plan: dict, audio: Path) -> str:
    r = media.run([media.FFMPEG, '-hide_banner', '-i', audio,
                   '-af', 'loudnorm=I=-16:TP=-2:LRA=9:print_format=json', '-f', 'null', '-'])
    m = media.loudnorm_stats(r.stderr)
    (project.tmp(plan['id']) / 'normalization.json').write_text(json.dumps(m, indent=2))
    return (f'loudnorm=I=-16:TP=-2:LRA=9:measured_I={m["input_i"]}:measured_TP={m["input_tp"]}:'
            f'measured_LRA={m["input_lra"]}:measured_thresh={m["input_thresh"]}:'
            f'offset={m["target_offset"]}:linear=true')


# -- the reel -------------------------------------------------------------------

def render(project: Project, plan: dict) -> Path:
    i = plan['id']
    tl = build(plan)
    project.out.mkdir(parents=True, exist_ok=True)
    (project.out / f'{plan["name"]}.srt').write_text(srt(tl.cues))
    overlay = make_overlay(project, plan, tl)
    audio = edit_audio(project, plan, tl)
    norm = loudnorm_filter(project, plan, audio)
    keys = list(plan['sources'])
    index = {k: n for n, k in enumerate(keys)}
    inputs = []
    for k in keys:
        inputs += ['-i', plan['sources'][k]['video']]
    inputs += ['-i', audio, '-i', overlay]
    g = []
    seen = {}
    for n, (k, a, b) in enumerate(tl.clips):
        src = plan['sources'][k]
        advance = src.get('video_advance', 0)
        turn = seen.get(k, 0)
        seen[k] = turn + 1
        zoom = ''
        if src.get('crop'):
            # these takes are cropped out of a wide master, so the upscale gets
            # a light unsharp to bring detail back to the level of the close takes
            box = src['crop'] if turn % 2 == 0 else src.get('crop_alt', src['crop'])
            zoom = (',crop=' + box + ',scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int'
                    ',unsharp=5:5:0.55:5:5:0.0')
        elif turn % 2 == 1:
            x = plan.get('zoom_x', 36)
            zoom = f',crop=1008:1792:{x}:0,scale=1080:1920:flags=lanczos'
        g.append(f'[{index[k]}:v]trim=start_frame={round((a + advance) * FPS)}:end_frame={round((b + advance) * FPS)},'
                 f'setpts=N/(30*TB){zoom},setsar=1[v{n}]')
    g.append(''.join(f'[v{n}]' for n in range(len(tl.clips))) +
             f'concat=n={len(tl.clips)}:v=1:a=0,setpts=N/(30*TB),{project.style.grade},'
             'setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709[base]')
    g += [f'[base][{len(keys) + 1}:v]overlay=eof_action=repeat:format=auto[v]', f'[{len(keys)}:a]{norm}[a]']
    out = project.reel_path(plan)
    print('Рендер', i, round(tl.duration, 2), 'с', flush=True)
    media.run([media.FFMPEG, '-hide_banner', '-y', '-threads', '3', *inputs, '-filter_complex_threads', '2',
               '-filter_complex', ';'.join(g), '-map', '[v]', '-map', '[a]',
               '-t', f'{tl.duration:.6f}', '-frames:v', str(round(tl.duration * FPS)),
               '-c:v', 'libx264', '-crf', '17', '-preset', 'fast', '-threads', '3',
               '-profile:v', 'high', '-level:v', '4.2', '-pix_fmt', 'yuv420p', '-r', '30',
               '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-movflags', '+faststart',
               '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
               '-map_metadata', '-1', out], project.tmp(i) / 'render.log')
    print('Готово', i, out.name, flush=True)
    overlay.unlink(missing_ok=True)
    audio.unlink(missing_ok=True)
    for png in (project.tmp(i) / 'graphics').glob('*.png'):
        png.unlink()
    return out
