"""Transcription tools (openai-whisper, imported only when used).

How they were used on real interviews:
- ``words`` (model small): word timings of a fragment, to pick cut points;
- ``hear`` (model medium): re-hear short disputed spots in narrow windows,
  sometimes slowed down to 0.85 — a long window misses word tails cut by a clip end;
- ``verify_text`` (model medium): read the finished reels back and print every
  planned subtitle next to what is actually heard.

Rule: if the models hear a word differently, it does not go into a subtitle.
Dropping a filler word is fine, replacing a word is not.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import media
from .project import Project, write_json
from .timeline import build, parse_srt_time


def _whisper():
    try:
        import torch
        import whisper
    except ImportError as e:
        raise SystemExit('Нужен openai-whisper: pip install openai-whisper') from e
    return whisper, torch


def words(audio, out, model='small', language='ru', threads=3) -> dict:
    """Word-level transcript of one file. The temperature ladder recovers the
    cases where whisper ends the transcript early; coverage is reported."""
    whisper, torch = _whisper()
    torch.set_num_threads(threads)
    m = whisper.load_model(model, device='cpu')
    j = m.transcribe(str(audio), language=language, fp16=False,
                     temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), beam_size=5,
                     condition_on_previous_text=False, word_timestamps=True,
                     no_speech_threshold=0.75, logprob_threshold=-1.2)
    write_json(out, j, indent=2)
    total = media.duration(audio)
    end = j['segments'][-1]['end'] if j['segments'] else 0.0
    note = '' if end >= total - 3 else '  ← текст обрывается раньше звука, проверьте хвост'
    print(f'{Path(audio).name}: текст до {end:.1f} с из {total:.1f}{note}')
    return j


def slice_words(full_json, start: float, dur: float, out) -> dict:
    """Cut a fragment's word transcript out of a whole-file one."""
    full = json.loads(Path(full_json).read_text())
    segs = []
    for s in full['segments']:
        if s['end'] < start or s['start'] > start + dur:
            continue
        ws = [dict(w, start=round(w['start'] - start, 2), end=round(w['end'] - start, 2))
              for w in s.get('words') or [] if start <= w['start'] <= start + dur]
        if ws:
            segs.append(dict(s, start=round(s['start'] - start, 2), end=round(s['end'] - start, 2), words=ws))
    data = dict(segments=segs, sliced_from=str(full_json), offset=start)
    write_json(out, data)
    print(Path(out).name, len(segs), 'сегментов')
    return data


def show_words(words_json, lo=0.0, hi=1e9):
    """Word timings of a range, compact enough to pick cuts from."""
    d = json.loads(Path(words_json).read_text())
    for s in d['segments']:
        if s['end'] < lo or s['start'] > hi:
            continue
        print(f"\n[{s['start']:7.2f}-{s['end']:7.2f}] {s['text'].strip()}")
        print('   ' + '  '.join(f"{w['word'].strip()}·{w['start']:.2f}" for w in s.get('words') or []))


def hear(spots_file, model='medium', language='ru'):
    """Re-hear short spots. Lines of the file: ``tag file a b [atempo]``."""
    whisper, _ = _whisper()
    m = whisper.load_model(model)
    for line in Path(spots_file).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        sp = line.split()
        tag, f, a, b = sp[:4]
        tempo = sp[4] if len(sp) > 4 else None
        x = media.pcm(f, 16000, start=a, end=b, filters=f'atempo={tempo}' if tempo else None)
        r = m.transcribe(x, language=language, temperature=0, condition_on_previous_text=False)
        print(f'{tag:6} {Path(f).name[:22]:22} {a:>7}-{b:<7} {tempo or "":4} → {r["text"].strip()}', flush=True)


def spots_from_srt(project: Project, specs) -> list[str]:
    """``ID:substring`` → hear() lines for matching subtitles of a finished reel:
    the cue itself (±0.3 s) and the cue with 1.5 s of context on both sides."""
    lines = []
    plans = {p['id']: p for p in project.plans()}
    for spec in specs:
        rid, sub = spec.split(':', 1)
        mp4 = project.reel_path(plans[rid])
        srt = mp4.with_suffix('.srt').read_text()
        for block in srt.strip().split('\n\n'):
            rows = block.split('\n')
            a, b = [parse_srt_time(x) for x in rows[1].split('-->')]
            if sub in ' '.join(rows[2:]):
                lines += [f'{rid} {mp4} {max(0, a - 0.3):.2f} {b + 0.3:.2f}',
                          f'{rid}+ {mp4} {max(0, a - 1.5):.2f} {b + 1.5:.2f}']
    print('\n'.join(lines))
    return lines


def verify_text(project: Project, ids=None, model='medium', language='ru', threads=6) -> dict:
    """Read the finished reels back with a bigger model; print every planned
    subtitle next to what is heard there."""
    from .checks import speech_wav
    whisper, torch = _whisper()
    torch.set_num_threads(threads)
    m = whisper.load_model(model, device='cpu')
    out = {}
    for p in project.plans(ids):
        video = project.reel_path(p)
        if not video.exists():
            print(p['id'], 'ролик ещё не собран')
            continue
        wav = speech_wav(project, p, video)
        j = m.transcribe(wav, language=language, fp16=False, temperature=0, beam_size=5,
                         condition_on_previous_text=False, word_timestamps=True)
        write_json(project.tmp(p['id']) / 'final_heard.json', j, indent=2)
        print('\n########', p['id'], video.name, flush=True)
        heard = [(s['start'], s['end'], s['text'].strip()) for s in j['segments']]
        for a, b, txt in build(p).cues:
            overlap = ' / '.join(t for s, e, t in heard if e > a and s < b)
            print(f'[{a:6.2f}-{b:6.2f}] СУБТИТР: {txt.replace(chr(10), " | ")}')
            print(f'                СЛЫШНО : {overlap}', flush=True)
        out[p['id']] = heard
    path = project.out / 'final_heard.json'
    prev = json.loads(path.read_text()) if path.exists() else {}
    write_json(path, {**prev, **out}, indent=2)
    print('\nПРОВЕРКА ТЕКСТА ГОТОВА')
    return out
