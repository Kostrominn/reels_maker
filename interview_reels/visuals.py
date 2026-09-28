"""Picture tools: split-screen sources, stickers, mouth strips and frame strips."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import media


# -- split screen ---------------------------------------------------------------------

def split_screen(out, top: tuple, bottom: tuple, start=2.0):
    """Two 1080×1920 cameras stacked into one vertical frame, no audio.

    ``top``/``bottom`` = (file, shift, face_x, face_y): ``shift`` = time in that
    file − reference time; the face position is in the source frame. Top: zoom
    4/3, the face at 45 % of the half. Bottom: zoom 3/2, the face at 70 % of the
    half, so the caption plate (y 1250–1450) sits on the wall above the head.

    Time 0 of the result = reference time ``start``, so neither side needs a
    negative seek. In a plan: ``video`` = this file, ``audio`` = the reference
    recording, ``advance`` = ``start``.

    Use it where lines overlap and giving them to one camera would be a guess:
    fast dialogue, reactions, lines two people finish together."""
    tf, tsh, tx0, ty0 = top
    bf, bsh, bx0, by0 = bottom
    cl = lambda v, lo, hi: max(lo, min(hi, v))
    tx, ty = cl(tx0 - 405, 0, 270), cl(ty0 - 324, 0, 1200)
    bx, by = cl(bx0 - 360, 0, 360), cl(by0 - 448, 0, 1280)
    g = (f'[0:v]trim=start={start + tsh:.3f},setpts=PTS-STARTPTS,crop=810:720:{tx}:{ty},scale=1080:960[t];'
         f'[1:v]trim=start={start + bsh:.3f},setpts=PTS-STARTPTS,crop=720:640:{bx}:{by},scale=1080:960[b];'
         '[t][b]vstack,drawbox=x=0:y=955:w=iw:h=10:color=0x11181d@1:t=fill,fps=30[v]')
    media.run([media.FFMPEG, '-v', 'error', '-y', '-i', tf, '-i', bf, '-filter_complex', g,
               '-map', '[v]', '-an', '-c:v', 'libx264', '-crf', '17', '-preset', 'fast', '-pix_fmt', 'yuv420p',
               '-shortest', out])
    print('сохранено', Path(out).name)


# -- stickers ----------------------------------------------------------------------------

def _emoji(style, ch, size):
    if not style.emoji_path:
        raise SystemExit('Нет цветного emoji-шрифта: укажите REELS_EMOJI_FONT')
    font = ImageFont.truetype(style.emoji_path, 160)       # colour emoji fonts come in one size
    im = Image.new('RGBA', (200, 200))
    ImageDraw.Draw(im).text((20, 20), ch, font=font, embedded_color=True)
    im = im.crop(im.getbbox())
    return im.resize((round(im.width * size / im.height), size), Image.LANCZOS)


def sticker(out, parts, style, height=150, pad=34, gap=18):
    """A pill-shaped sticker in the pack style. ``parts``: ``emoji:❤️``,
    ``text:120`` (white) or ``accent:180`` (accent colour).

    Rule from real use: a sticker shows only what is said in the reel itself
    (numbers, names), is timed to the spoken word and stays 1–4 s."""
    f = style.font(92, 760)
    pieces = []
    for part in parts:
        kind, _, value = part.partition(':')
        if kind == 'emoji':
            pieces.append(_emoji(style, value, 96))
            continue
        colour = style.accent if kind == 'accent' else style.white
        w = int(ImageDraw.Draw(Image.new('RGBA', (1, 1))).textlength(value, font=f)) + 4
        im = Image.new('RGBA', (w, height))
        ImageDraw.Draw(im).text((0, height / 2), value, font=f, fill=colour, anchor='lm')
        pieces.append(im)
    width = sum(p.width for p in pieces) + gap * (len(pieces) - 1) + 2 * pad
    im = Image.new('RGBA', (width, height))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, width - 1, height - 1), radius=height // 2, fill=(*style.plate, 232))
    x = pad
    for p in pieces:
        im.alpha_composite(p, (x, (height - p.height) // 2))
        x += p.width + gap
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(Path(out).name, im.size)
    return im


# -- who speaks ----------------------------------------------------------------------------

def mouths(out, step: float, a: float, b: float, cams):
    """Mouth strips: one row per camera, one frame every ``step`` seconds.
    ``cams``: (file, shift, crop) with shift = time in the file − plan time.

    On real interviews loudness differences between the cameras did not tell
    the speakers apart; these strips did. If a line still cannot be attributed,
    keep it audible without a subtitle."""
    ts = list(np.arange(a, b + 1e-6, step))
    cw, ch = 150, 112
    img = Image.new('RGB', (cw * len(ts), (ch + 2) * len(cams) + 14), (15, 15, 15))
    d = ImageDraw.Draw(img)
    for j, (f, sh, crop) in enumerate(cams):
        raw = subprocess.run([media.FFMPEG, '-v', 'error', '-ss', f'{a + float(sh):.3f}', '-t', f'{b - a + step:.3f}',
                              '-i', str(f), '-vf', f'fps={1 / step},crop={crop},scale={cw}:{ch}',
                              '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(-1, ch, cw, 3)
        for i in range(min(len(ts), len(fr))):
            img.paste(Image.fromarray(fr[i]), (i * cw, j * (ch + 2)))
    for i, t in enumerate(ts):
        if i % 5 == 0:
            d.text((i * cw + 2, (ch + 2) * len(cams)), f'{t:.1f}', fill=(255, 255, 0))
    img.save(out)
    print('сохранено', Path(out).name, img.size)


def frames(video, out, times):
    """A strip of frames from a finished reel."""
    im = Image.new('RGB', (240 * len(times), 446))
    d = ImageDraw.Draw(im)
    for i, t in enumerate(times):
        im.paste(Image.fromarray(media.frame(video, t, 'scale=240:427', (240, 427))), (240 * i, 0))
        d.text((240 * i + 4, 430), f'{t:.1f}s', fill='yellow')
    im.save(out)
    print(Path(video).name, '→', out)
