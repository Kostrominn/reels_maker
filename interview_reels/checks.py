"""Checks before and after the render.

Before: ``check_text`` (everything fits) and ``check_cuts`` (every cut sits in a
pause). After, on the finished files: container and loudness
(``check_outputs``), lip sync (``verify_sync``), every subtitle visible on a real
frame (``verify_captions``) and loudness steps between clips (``check_levels``).
The results land next to the reels as JSON; ``qc.report`` turns them into text.
"""
from __future__ import annotations

import io
import json
import subprocess
import warnings

import numpy as np
from PIL import Image, ImageDraw

from . import media
from .project import Project, write_json
from .render import TextOverflow, caption_layer, title_layer
from .timeline import FPS, build

# -- before the render ------------------------------------------------------------


def check_text(project: Project, ids=None) -> int:
    """Subtitles and titles fit the frame. Run it before every render."""
    bad = 0
    for p in project.plans(ids):
        for _, _, _, txt in p['cues']:
            try:
                caption_layer(p, txt, project.style)
            except TextOverflow as e:
                bad += 1
                print(p['id'], e)
        try:
            title_layer(p, project.style)
        except TextOverflow as e:
            bad += 1
            print(p['id'], e)
    print('всё влезает' if not bad else f'НЕ ВЛЕЗАЕТ: {bad}')
    return bad


SR, HOP = 16000, 160     # 10 ms levels
HALF = 0.05              # window around a cut, seconds
MARGIN_DB = 9            # how far below the speech level counts as a pause


def cut_level(path, t):
    """Quietest level in ±50 ms around t, and the recording's speech level.

    Word timings are unreliable at the edges (whisper stretches a word over the
    pause that follows), so the ground truth is the audio itself."""
    db = media.file_levels(path, SR, HOP)
    speech = float(np.percentile(db, 88))
    i0 = max(0, int((t - HALF) * SR / HOP))
    i1 = min(len(db), int((t + HALF) * SR / HOP) + 1)
    if i1 <= i0:
        return None, speech
    return float(db[i0:i1].min()), speech


def seamless_joins(plan: dict):
    """Clip boundaries where the next clip continues the same recording at the
    same moment: the sound is continuous there, only the picture cuts."""
    starts, ends = set(), set()
    clips = plan['clips']
    for n in range(1, len(clips)):
        pk, _, pb = clips[n - 1]
        key, a, _ = clips[n]
        if plan['sources'][pk]['audio'] == plan['sources'][key]['audio'] and abs(pb - a) < 0.02:
            starts.add(n)
            ends.add(n - 1)
    return starts, ends


def check_cuts(project: Project, ids=None) -> int:
    """Every cut must land in a pause, not in the middle of a word."""
    problems = 0
    for p in project.plans(ids):
        starts, ends = seamless_joins(p)
        for n, (key, a, b) in enumerate(p['clips']):
            audio = p['sources'][key]['audio']
            edges = ([(a, 'начало')] if n not in starts else []) + ([(b, 'конец')] if n not in ends else [])
            for edge, label in edges:
                lo, speech = cut_level(audio, edge)
                if lo is not None and lo > speech - MARGIN_DB:
                    problems += 1
                    print(f'{p["id"]} {key} {label} {edge:.2f}: рез по речи ({lo:.0f} dB при речи {speech:.0f} dB)')
    print('все склейки в паузах' if not problems else f'ПРОБЛЕМНЫХ СКЛЕЕК: {problems}')
    return problems


# -- the finished reels -------------------------------------------------------------

def _finished(project: Project, ids):
    for p in project.plans(ids):
        video = project.reel_path(p)
        if not video.exists():
            print(p['id'], 'ролик ещё не собран —', video.name)
            continue
        yield p, video


def speech_wav(project: Project, plan: dict, video) -> str:
    wav = project.tmp(plan['id']) / 'final_speech.wav'
    if not wav.exists() or wav.stat().st_mtime < video.stat().st_mtime:
        subprocess.run([media.FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video),
                        '-vn', '-ac', '1', '-ar', '16000', str(wav)], check=True)
    return str(wav)


def check_outputs(project: Project, ids=None) -> list[dict]:
    """Container, geometry, codecs, loudness, black frames and freezes, plus a
    contact sheet of seven frames per reel."""
    results = []
    for p, video in _finished(project, ids):
        i = p['id']
        q = media.probe(video)
        d = float(q['format']['duration'])
        v = next(x for x in q['streams'] if x['codec_type'] == 'video')
        a = next(x for x in q['streams'] if x['codec_type'] == 'audio')
        print('Проверяю', video.name, flush=True)
        r = subprocess.run([media.FFMPEG, '-hide_banner', '-i', str(video),
                            '-af', 'loudnorm=I=-16:TP=-2:LRA=9:print_format=json',
                            '-vf', 'blackdetect=d=0.15:pix_th=0.05,freezedetect=n=-50dB:d=1.2',
                            '-f', 'null', '-'], capture_output=True, text=True)
        tmp = project.tmp(i)
        (tmp / 'qc.log').write_text(r.stderr)
        loud = media.loudnorm_stats(r.stderr)
        times = [0.2, 1.0, 2.4, 3.9, d / 2, d - 1.3, min(d, float(v['duration'])) - 0.08]
        sheet = Image.new('RGB', (360 * 4, 680 * 2), (23, 27, 34))
        draw = ImageDraw.Draw(sheet)
        for n, t in enumerate(times):
            im = media.frame(video, t, 'scale=360:640', (360, 640))
            sheet.paste(Image.fromarray(im), ((n % 4) * 360, (n // 4) * 680))
            draw.text(((n % 4) * 360 + 14, (n // 4) * 680 + 646), f'{i} / {t:.2f} s', fill='white')
        sheet.save(tmp / 'contact.jpg', quality=93)
        speech_wav(project, p, video)
        valid = dict(
            decode_ok=r.returncode == 0,
            geometry=v['width'] == 1080 and v['height'] == 1920,
            video_codec=v['codec_name'] == 'h264',
            audio_codec=a['codec_name'] == 'aac',
            audio_rate=a['sample_rate'] == '48000',
            loudness=abs(float(loud['input_i']) + 16) <= 1,
            true_peak=float(loud['input_tp']) <= -1.0,
            av_duration=abs(float(v['duration']) - float(a['duration'])) <= 1 / 30,
            frame_rate=v['r_frame_rate'] == '30/1',
            sdr_color=all(v.get(k) == 'bt709' for k in ['color_space', 'color_transfer', 'color_primaries']),
            no_black='black_start' not in r.stderr,
            no_freeze='freeze_start' not in r.stderr)
        results.append(dict(id=i, file=video.name, duration=d, megabytes=round(video.stat().st_size / 1e6, 2),
                            loudness=loud, checks=valid))
        bad = [k for k, ok in valid.items() if not ok]
        print(i, 'всё в норме' if not bad else 'НЕ ПРОШЛО: ' + ', '.join(bad),
              f'· {float(loud["input_i"]):.1f} LUFS, пик {float(loud["input_tp"]):.1f} dBTP', flush=True)
    _merge(project, 'quality_checks.json', results)
    return results


def _band(path) -> np.ndarray:
    from scipy import signal
    r = subprocess.run([media.FFMPEG, '-v', 'error', '-i', str(path), '-vn', '-ac', '1', '-ar', '8000',
                        '-f', 'f32le', '-'], capture_output=True)
    # a split-screen source has no audio track at all: treat it like a silent camera
    a = np.frombuffer(r.stdout, np.float32) if r.returncode == 0 and r.stdout else np.zeros(8000, np.float32)
    return signal.sosfiltfilt(signal.butter(4, [180, 2800], btype='bandpass', fs=8000, output='sos'), a)


def verify_sync(project: Project, ids=None) -> list[dict]:
    """Lip sync on the finished file: the reel's sound is matched against the
    camera's own track at the clip position. Normal is under one frame (33 ms).

    A camera without sound (or a split-screen source) cannot be checked this
    way; for it the check confirms that the sound came from the right place of
    the reference file, and the source is listed in ``visual_sync_sources``."""
    from scipy import signal
    res = []
    for p, video in _finished(project, ids):
        tl = build(p)
        actual = _band(video)
        sources, visual = {}, set()
        for k, v in p['sources'].items():
            a = _band(v['video'])
            if np.sqrt(np.mean(a ** 2)) < 1e-4:
                a = _band(v['audio'])
                visual.add(k)
            sources[k] = a
        outpos, windows = 0.0, []
        for key, a, b in tl.clips:
            source = sources[key]
            d = b - a
            for s in ([0.15] if d < 3 else [0.25, max(0.25, d - 2.3)]):
                dur = min(2.0, d - s - 0.1)
                x = actual[round((outpos + s) * 8000):round((outpos + s + dur) * 8000)]
                shift = p['sources'][key]['advance'] if key in visual else p['sources'][key].get('video_advance', 0)
                refstart = a + shift if key in visual else round((a + shift) * FPS) / FPS
                y = source[round((refstart + s) * 8000):round((refstart + s + dur) * 8000)]
                n = min(len(x), len(y))
                x, y = x[:n], y[:n]
                if n < 800:
                    continue
                co = signal.correlate(x, y, mode='full', method='fft')
                lags = signal.correlation_lags(n, n)
                mask = abs(lags) <= 800
                k = np.argmax(co[mask])
                lag = lags[mask][k] / 8000
                coeff = float(co[mask][k] / (np.sqrt(np.sum(x * x) * np.sum(y * y)) or 1.0))
                windows.append(dict(output_start=round(outpos + s, 3), source_start=round(a + s, 3),
                                    lag_seconds=float(lag), correlation=round(coeff, 3)))
            outpos += d
        if not windows:
            continue
        item = dict(id=p['id'], visual_sync_sources=sorted(visual), windows=windows,
                    median_lag_seconds=float(np.median([w['lag_seconds'] for w in windows])),
                    max_abs_lag_seconds=float(max(abs(w['lag_seconds']) for w in windows)))
        res.append(item)
        print(f'{p["id"]}: медиана {item["median_lag_seconds"] * 1000:+.0f} мс, '
              f'максимум {item["max_abs_lag_seconds"] * 1000:.0f} мс по {len(windows)} окнам'
              + (f' · без своего звука: {", ".join(sorted(visual))}' if visual else ''), flush=True)
    _merge(project, 'sync_verification.json', res)
    return res


def verify_captions(project: Project, ids=None) -> list[dict]:
    """Every subtitle is looked up on a real frame of the finished reel: the
    share of the expected white glyph pixels that are light there."""
    result = []
    for p, video in _finished(project, ids):
        tl = build(p)
        i = p['id']
        cues = tl.cues
        sheet = Image.new('RGB', (1440, 190 * max(1, (len(cues) + 1) // 2)), (23, 27, 34))
        dr = ImageDraw.Draw(sheet)
        for n, (a, b, txt) in enumerate(cues):
            mid = (a + b) / 2
            dat = subprocess.check_output([media.FFMPEG, '-v', 'error', '-ss', str(mid), '-i', str(video),
                                           '-frames:v', '1', '-vf', 'crop=1080:230:0:1270',
                                           '-f', 'image2pipe', '-vcodec', 'png', '-'])
            im = Image.open(io.BytesIO(dat)).convert('RGB')
            actual = np.asarray(im)
            expected = np.asarray(caption_layer(p, txt, project.style)[0])[1270:1500]
            white = (expected[:, :, :3].min(axis=2) > 220) & (expected[:, :, 3] > 245)
            light = (actual.min(axis=2) > 205) & ((actual.max(axis=2) - actual.min(axis=2)) < 45)
            recall = float(light[white].mean()) if white.any() else 1.0
            result.append(dict(id=i, time=round(mid, 3), text=txt, visible_glyph_ratio=round(recall, 4),
                               passed=recall > .88))
            sheet.paste(im.resize((720, 153)), ((n % 2) * 720, (n // 2) * 190))
            dr.text(((n % 2) * 720 + 14, (n // 2) * 190 + 157), f'{i} / {mid:.2f} s / {recall:.1%}', fill='white')
        sheet.save(project.tmp(i) / 'all_captions.jpg', quality=92)
    _merge(project, 'caption_verification.json', result, key=('id', 'time'))
    failed = [x for x in result if not x['passed']]
    print('Субтитров проверено', len(result), '· прошли', len(result) - len(failed))
    for x in failed:
        print('  проверить:', x['id'], f'{x["time"]:.1f} с', repr(x['text']))
    return result


LIMIT_DB = 3.5


def check_levels(project: Project, ids=None) -> float:
    """Loudness steps between clips of one reel. Loudness is normalised per reel,
    so a take recorded a little louder than its neighbour stays louder."""
    worst = 0.0
    warnings.filterwarnings('ignore')
    for p, video in _finished(project, ids):
        tl = build(p)
        if len(tl.clips) < 2:
            print(f'{p["id"]}: один клип — сравнивать нечего')
            continue
        db = media.levels_db(media.pcm(video), 800)
        levels, t = [], 0.0
        for key, a, b in tl.clips:
            seg = db[int(t / 0.05):int((t + (b - a)) / 0.05)]
            speech = seg[seg > seg.max() - 18] if len(seg) else seg
            levels.append((key, round(t, 2), float(np.median(speech)) if len(speech) else float('nan')))
            t += b - a
        vals = [v for _, _, v in levels if v == v]
        if len(vals) < 2:
            print(f'{p["id"]}: файл пуст — пропускаю')
            continue
        med = float(np.median(vals))
        flags = [(k, s, v) for k, s, v in levels if abs(v - med) > LIMIT_DB]
        spread = max(vals) - min(vals)
        worst = max(worst, spread)
        print(f'{p["id"]}: разброс между клипами {spread:.1f} dB, медиана {med:.1f} dB'
              + ('' if not flags else '  ← ' + ', '.join(f'{k}@{s:.1f}с {v - med:+.1f}' for k, s, v in flags)))
    print(f'худший разброс {worst:.1f} dB —', 'слышимых скачков нет' if worst <= 5 else 'ПРОВЕРИТЬ')
    return worst


def _merge(project: Project, name: str, items: list[dict], key=('id',)):
    """Update one JSON result file next to the reels without losing other reels."""
    path = project.out / name
    old = json.loads(path.read_text()) if path.exists() else []
    fresh_ids = {x['id'] for x in items}
    merged = [x for x in old if x['id'] not in fresh_ids] + items
    merged.sort(key=lambda x: tuple(x[k] for k in key))
    project.out.mkdir(parents=True, exist_ok=True)
    write_json(path, merged, indent=2)
