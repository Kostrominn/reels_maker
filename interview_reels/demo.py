"""A self-contained demo project made of synthetic media, so the whole pipeline
can be tried (and tested) without anyone's real interview.

Two «cameras» of a test pattern share one recording: pink noise in 1.4 s
«phrases» with 0.6 s pauses. Camera B started 0.5 s later; the demo measures
that by sound, like on a real shoot, and writes it into the plan."""
from __future__ import annotations

from pathlib import Path

from . import media
from .project import write_json

PHRASE_GATE = 'if(gte(mod(t,2),0.6),1,0.003)'     # pauses at [2k, 2k+0.6)


def build_media(root: Path) -> tuple[Path, Path]:
    m = root / 'media'
    m.mkdir(parents=True, exist_ok=True)
    ref = m / 'room.wav'
    media.run([media.FFMPEG, '-v', 'error', '-y', '-f', 'lavfi',
               '-i', 'anoisesrc=d=13:c=pink:r=48000:a=0.25:seed=7',
               '-af', f"volume='{PHRASE_GATE}':eval=frame", '-c:a', 'pcm_s16le', ref])
    cam_a, cam_b = m / 'cam_a.mov', m / 'cam_b.mov'
    media.run([media.FFMPEG, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=1080x1920:rate=30:duration=13',
               '-i', ref, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast',
               '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le', '-shortest', cam_a])
    media.run([media.FFMPEG, '-v', 'error', '-y', '-f', 'lavfi',
               '-i', 'testsrc2=size=1080x1920:rate=30:duration=12.5,hue=h=150',
               '-i', ref, '-filter_complex', '[1:a]atrim=start=0.5,asetpts=PTS-STARTPTS[a]',
               '-map', '0:v', '-map', '[a]', '-c:v', 'libx264', '-crf', '23', '-preset', 'veryfast',
               '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le', '-shortest', cam_b])
    return cam_a, cam_b


def build_project(root) -> Path:
    from .style import Style
    from .sync import audio_offset
    from .visuals import sticker
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cam_a, cam_b = build_media(root)
    offset, score = audio_offset(cam_a, cam_b)          # time in B = time in A + offset
    print(f'синхрон камеры B по звуку: {offset:+.3f} с (score {score:.2f})')
    style = Style.from_meta({}, root)
    sticker(root / 'insets' / 'st_demo.png', ['text:демо', 'accent:OK'], style)
    plan = dict(
        id='01', name='01_Демо', kicker='ДЕМО',
        title=['Синтетический ролик,', 'чтобы проверить конвейер'],
        accent=['склейки', 'субтитр', 'синхрон'],
        sources=dict(
            a=dict(video='media/cam_a.mov', audio='media/cam_a.mov', advance=0.0),
            b=dict(video='media/cam_b.mov', audio='media/cam_a.mov', advance=round(-offset, 3),
                   crop='900:1600:90:160', crop_alt='860:1529:110:190')),
        # cuts sit in the pauses; the last join continues the same sound, only the picture cuts
        clips=[['a', 0.2, 4.3], ['b', 6.3, 10.3], ['a', 10.3, 12.3]],
        cues=[['a', 0.6, 2.0, 'Это синтетический ролик:\nшум вместо речи'],
              ['a', 2.6, 4.0, 'Склейки стоят в паузах'],
              ['b', 6.6, 8.0, 'Вторая камера,\nсинхрон замерен по звуку'],
              ['b', 8.6, 10.0, 'Субтитр виден\nна реальном кадре'],
              ['a', 10.6, 12.0, 'Конец демо']],
        insets=[[6.6, 8.0, 'insets/st_demo.png', 60, 260]])
    write_json(root / 'plans.json', [plan])
    write_json(root / 'rows.json', [dict(
        id='01', title='Демо', audience='Разработчикам',
        cover_source='b', cover_time=2.0, cover_kicker='ДЕМО', cover_title=['interview_reels', 'синтетический пример'],
        note='Две «камеры», шум вместо речи, склейки в паузах.',
        caption='Синтетический ролик: проверка того, что конвейер собирается и проходит все проверки.')])
    write_json(root / 'pack.json', dict(title='Демо interview_reels', out='out',
                                        intro='Один синтетический ролик: две «камеры», шум вместо речи.'))
    return root


def run_all(root) -> bool:
    from . import checks, qc
    from .package import package
    from .project import Project
    from .render import render
    project = Project.open(root)
    assert checks.check_text(project) == 0
    assert checks.check_cuts(project) == 0
    for plan in project.plans():
        render(project, plan)
    checks.check_outputs(project)
    checks.verify_sync(project)
    checks.verify_captions(project)
    checks.check_levels(project)
    ok = qc.report(project)
    package(project)
    return ok
