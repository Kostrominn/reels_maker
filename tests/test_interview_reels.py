import re
import shutil
import wave

import numpy as np
import pytest

from interview_reels.render import accent_pattern
from interview_reels.timeline import build, parse_srt_time, srt, srt_time

needs_ffmpeg = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='нужен ffmpeg')


def plan(**extra):
    p = dict(
        id='01', name='01_test', kicker='K', title=['T'], accent=[],
        sources=dict(a=dict(video='a.mov', audio='a.mov', advance=0.0),
                     b=dict(video='b.mov', audio='a.mov', advance=0.5)),
        clips=[['a', 1.0, 3.0], ['b', 5.0, 6.0]],
        cues=[['a', 1.5, 2.5, 'first'], ['b', 4.0, 5.5, 'second'], ['a', 5.2, 5.8, 'hidden: camera a is off']])
    p.update(extra)
    return p


def test_timeline_moves_clips_into_source_time_and_cues_into_reel_time():
    tl = build(plan(insets=[[5.2, 5.4, 'st.png', 10, 20]]))
    assert tl.clips == [['a', 1.0, 3.0], ['b', 4.5, 5.5]]
    assert tl.duration == pytest.approx(3.0)
    # a cue is clipped to its clip and shown only while its own camera is on screen
    assert [[round(a, 3), round(b, 3), t] for a, b, t in tl.cues] == [[0.5, 1.5, 'first'], [2.0, 2.5, 'second']]
    assert [[round(a, 3), round(b, 3)] for a, b, *_ in tl.insets] == [[2.2, 2.4]]


def test_clip_bounds_are_rounded_to_frames():
    tl = build(plan(clips=[['b', 5.012, 6.0]], cues=[]))
    assert tl.clips[0][1] * 30 == pytest.approx(round(tl.clips[0][1] * 30))


def test_srt_times():
    assert srt_time(3723.456) == '01:02:03,456'
    assert parse_srt_time('01:02:03,456') == pytest.approx(3723.456)
    assert srt([[0.0, 1.5, 'a\nb']]).startswith('1\n00:00:00,000 --> 00:00:01,500\na\nb')


def test_accent_colours_whole_words_only():
    pattern = accent_pattern(dict(accent=['пи', 'плюсник']))
    hits = lambda s: [m.group(0) for m in re.finditer(pattern, s, flags=re.I)]
    assert hits('на олимпиаду') == []
    assert hits('Пи или е') == ['Пи']
    assert hits('культ плюсника') == ['плюсника']


def test_wrap_prefers_fewest_even_lines(tmp_path):
    from interview_reels.fixes import LIMIT, wrap
    from interview_reels.style import Style
    font = Style.from_meta({}, tmp_path).caption_font()
    text = 'Если летом ничего не делаешь, то в сентябре очень тяжело'
    out = wrap(text, font)
    assert out.replace('\n', ' ') == text
    assert 1 < len(out.split('\n')) <= 3
    from PIL import Image, ImageDraw
    d = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    assert all(d.textlength(line, font=font) <= LIMIT for line in out.split('\n'))


def _wav(path, x, sr=16000):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())


@needs_ffmpeg
def test_audio_offset_finds_a_late_start(tmp_path):
    from interview_reels.sync import audio_offset
    rng = np.random.default_rng(1)
    sr = 16000
    gate = (np.arange(10 * sr) % sr < 0.7 * sr).astype(np.float32)       # phrases and pauses
    x = rng.normal(0, 0.2, 10 * sr).astype(np.float32) * gate
    _wav(tmp_path / 'a.wav', x)
    _wav(tmp_path / 'b.wav', x[int(0.37 * sr):])                          # camera B started 0.37 s later
    offset, score = audio_offset(tmp_path / 'a.wav', tmp_path / 'b.wav')
    assert offset == pytest.approx(-0.37, abs=0.011)
    assert score > 0.9


@needs_ffmpeg
def test_cut_checker_flags_a_cut_inside_a_phrase(tmp_path):
    from interview_reels.checks import cut_level, MARGIN_DB
    sr = 16000
    gate = (np.arange(6 * sr) % (2 * sr) >= 0.6 * sr).astype(np.float32)  # pauses at [2k, 2k+0.6)
    x = np.random.default_rng(2).normal(0, 0.2, 6 * sr).astype(np.float32) * gate
    _wav(tmp_path / 'room.wav', x)
    quiet, speech = cut_level(tmp_path / 'room.wav', 2.3)
    loud, _ = cut_level(tmp_path / 'room.wav', 3.0)
    assert quiet < speech - MARGIN_DB
    assert loud > speech - MARGIN_DB


@needs_ffmpeg
def test_demo_project_renders_and_passes_every_check(tmp_path):
    pytest.importorskip('scipy')
    from interview_reels.demo import build_project, run_all
    from interview_reels.project import Project
    root = build_project(tmp_path / 'demo')
    project = Project.open(root)
    assert project.plans()[0]['sources']['b']['advance'] == pytest.approx(0.5, abs=0.011)
    assert run_all(root)
    out = project.out
    for name in ('01_Демо.mp4', '01_Демо.srt', '01_Демо_обложка.jpg', 'QC_REPORT.md', 'СМОТРЕТЬ.html', 'ПОДПИСИ.md'):
        assert (out / name).exists(), name
