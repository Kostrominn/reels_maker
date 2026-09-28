"""Command line: python -m interview_reels [-p PROJECT] <command> …

Typical order for one reel:
  check-text → check-cuts (→ autofix / balance) → render →
  check-outputs → verify-sync → verify-captions → check-levels → verify-text →
  qc → package
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__, media


def _project(args):
    from .project import Project
    return Project.open(args.project, args.out_dir, args.cache)


def _ids(args):
    return args.ids or None


def _range(text):
    a, b = text.split('-', 1)
    return float(a), float(b)


# -- small audio tools -------------------------------------------------------------------

def envelope(path, ranges):
    """Audio level at 50 ms steps, 20 values per line."""
    for r in ranges:
        a, b = _range(r)
        db = media.levels_db(media.pcm(path, 16000, start=a, end=b), 800)
        print(f'== {path} {r}  (50 мс)')
        for i in range(0, len(db), 20):
            print(f'  {a + i * 0.05:7.2f} ' + ' '.join(f'{v:4.0f}' for v in db[i:i + 20]))


def find_pause(path, times, win=0.4):
    """The quietest 100 ms within ±win of each time."""
    db = media.levels_db(media.pcm(path, 16000), 160)
    sm = np.convolve(db, np.ones(10) / 10, mode='same')
    for t in times:
        i0, i1 = max(0, int((t - win) * 100)), int((t + win) * 100)
        k = i0 + int(np.argmin(sm[i0:i1]))
        print(f'{t:8.2f} -> {k / 100:8.2f}  ({sm[k]:.0f} dB, было {sm[int(t * 100)]:.0f})')


def _cam(spec):
    f, shift, x, y = spec.rsplit(':', 3)
    return f, float(shift), int(x), int(y)


# -- commands ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m interview_reels', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', action='version', version=__version__)
    ap.add_argument('-p', '--project', default='.', help='папка проекта с plans.json (по умолчанию текущая)')
    ap.add_argument('--out', dest='out_dir',
                    help='куда класть готовые ролики (по умолчанию pack.json "out" или <проект>/out)')
    ap.add_argument('--cache', help='промежуточные файлы (по умолчанию <проект>/.cache)')
    sub = ap.add_subparsers(dest='cmd', required=True, metavar='КОМАНДА')

    def cmd(name, help, ids=False):
        p = sub.add_parser(name, help=help, description=help)
        if ids:
            p.add_argument('ids', nargs='*', metavar='ID', help='какие ролики (по умолчанию все)')
        return p

    # assembly
    cmd('check-text', 'субтитры и заголовки влезают в кадр (до рендера)', ids=True)
    cmd('check-cuts', 'все склейки стоят в паузах, а не посреди слова (до рендера)', ids=True)
    p = cmd('autofix', 'сдвинуть склейки в ближайшие паузы и перенести длинные субтитры', ids=True)
    p.add_argument('--draft', help='исправить один план из отдельного JSON-файла')
    cmd('balance', 'выровнять громкость клипов (пишет clip_gains в plans.json)', ids=True)
    cmd('render', 'собрать ролики', ids=True)
    # checks of the finished files
    cmd('check-outputs', 'контейнер, геометрия, громкость, чёрные кадры, фризы', ids=True)
    cmd('verify-sync', 'звук совпадает с картинкой (норма — меньше кадра)', ids=True)
    cmd('verify-captions', 'каждый субтитр виден на реальном кадре', ids=True)
    cmd('check-levels', 'скачки громкости между клипами', ids=True)
    p = cmd('verify-text', 'сверить субтитры со звуком готового ролика (whisper medium)', ids=True)
    p.add_argument('--model', default='medium')
    cmd('qc', 'сводка QC_REPORT.md по результатам проверок')
    cmd('package', 'обложки, ПОДПИСИ.md, НАЧАТЬ_ЗДЕСЬ.md и страница СМОТРЕТЬ.html')
    p = cmd('all', 'всё подряд: проверки до рендера, рендер, проверки после, qc, package', ids=True)
    p.add_argument('--with-text', action='store_true', help='включить verify-text (медленно)')
    # tools
    p = cmd('fetch', 'вырезать фрагмент из файла, URL или Яндекс.Диска (disk:/…)')
    p.add_argument('src'); p.add_argument('start', type=float); p.add_argument('duration', type=float)
    p.add_argument('out'); p.add_argument('--hlg', action='store_true', help='тонмаппинг HLG → SDR (нужен zscale)')
    p.add_argument('--ffmpeg', help='ffmpeg с фильтром zscale')
    p = cmd('sync-audio', 'сдвиг между двумя записями по огибающей звука')
    p.add_argument('ref', help='опорная запись (её время — время плана)'); p.add_argument('cam')
    p.add_argument('--ref-range', help='окно в опорной записи, a-b'); p.add_argument('--cam-range')
    p.add_argument('--max-lag', type=float)
    p = cmd('sync-motion', 'сдвиг по движению в кадре — для камеры без звука')
    p.add_argument('a'); p.add_argument('crop_a', help='например crop=1080:1920:0:0')
    p.add_argument('b'); p.add_argument('crop_b'); p.add_argument('--max-lag', type=float, default=4.0)
    p = cmd('words', 'пословная расшифровка (whisper small)')
    p.add_argument('audio'); p.add_argument('out'); p.add_argument('--model', default='small')
    p = cmd('slice-words', 'вырезать кусок пословной расшифровки целого файла')
    p.add_argument('full'); p.add_argument('start', type=float); p.add_argument('duration', type=float)
    p.add_argument('out')
    p = cmd('show-words', 'пословные тайминги, чтобы выбрать склейки')
    p.add_argument('words'); p.add_argument('lo', nargs='?', type=float, default=0.0)
    p.add_argument('hi', nargs='?', type=float, default=1e9)
    p = cmd('hear', 'переслушать короткие места (строки: метка файл a b [atempo])')
    p.add_argument('spots'); p.add_argument('--model', default='medium')
    p = cmd('spots-from-srt', 'строки для hear по субтитрам готового ролика: ID:подстрока …')
    p.add_argument('specs', nargs='+')
    p = cmd('envelope', 'уровень звука шагом 50 мс')
    p.add_argument('file'); p.add_argument('ranges', nargs='+', metavar='A-B')
    p = cmd('find-pause', 'самые тихие 100 мс рядом со склейкой')
    p.add_argument('file'); p.add_argument('times', nargs='+', type=float); p.add_argument('--win', type=float, default=0.4)
    p = cmd('mouths', 'раскадровка ртов, чтобы понять, кто говорит')
    p.add_argument('out'); p.add_argument('step', type=float); p.add_argument('range', metavar='A-B')
    p.add_argument('cams', nargs='+', metavar='ФАЙЛ:СДВИГ:CROP', help='сдвиг = время в файле − время плана')
    p = cmd('frames', 'полоса кадров готового ролика')
    p.add_argument('id'); p.add_argument('out'); p.add_argument('times', nargs='+', type=float)
    p = cmd('clip-levels', 'громкость каждого клипа готового ролика')
    p.add_argument('id')
    p = cmd('split', 'разделённый экран из двух камер (без звука)')
    p.add_argument('out'); p.add_argument('--top', required=True, metavar='ФАЙЛ:СДВИГ:X:Y')
    p.add_argument('--bottom', required=True, metavar='ФАЙЛ:СДВИГ:X:Y'); p.add_argument('--start', type=float, default=2.0)
    p = cmd('sticker', 'наклейка в стиле пака: части emoji:❤️ text:120 accent:180')
    p.add_argument('out'); p.add_argument('parts', nargs='+')
    p = cmd('serve', 'локальный сервер с перемоткой для СМОТРЕТЬ.html')
    p.add_argument('--port', type=int, default=8765); p.add_argument('--dir')
    p = cmd('demo', 'создать демо-проект из синтетических данных и прогнать весь конвейер')
    p.add_argument('dir'); p.add_argument('--no-run', action='store_true', help='только создать проект')

    args = ap.parse_args(argv)
    c = args.cmd
    if c in ('check-text', 'check-cuts'):
        from . import checks
        fn = checks.check_text if c == 'check-text' else checks.check_cuts
        return 1 if fn(_project(args), _ids(args)) else 0
    if c == 'autofix':
        from .fixes import autofix
        autofix(_project(args), _ids(args), args.draft)
    elif c == 'balance':
        from .fixes import balance_levels
        balance_levels(_project(args), _ids(args))
    elif c == 'render':
        from .render import render
        project = _project(args)
        for plan in project.plans(_ids(args)):
            render(project, plan)
    elif c in ('check-outputs', 'verify-sync', 'verify-captions', 'check-levels'):
        from . import checks
        getattr(checks, c.replace('-', '_'))(_project(args), _ids(args))
    elif c == 'verify-text':
        from .speech import verify_text
        verify_text(_project(args), _ids(args), args.model)
    elif c == 'qc':
        from .qc import report
        return 0 if report(_project(args)) else 1
    elif c == 'package':
        from .package import package
        package(_project(args))
    elif c == 'all':
        from . import checks, qc
        from .package import package
        from .render import render
        project = _project(args)
        ids = _ids(args)
        if checks.check_text(project, ids) or checks.check_cuts(project, ids):
            print('Сначала исправьте план: см. выше')
            return 1
        for plan in project.plans(ids):
            render(project, plan)
        checks.check_outputs(project, ids)
        checks.verify_sync(project, ids)
        checks.verify_captions(project, ids)
        checks.check_levels(project, ids)
        if args.with_text:
            from .speech import verify_text
            verify_text(project, ids)
        ok = qc.report(project)
        if project.rows():
            package(project)
        return 0 if ok else 1
    elif c == 'fetch':
        from .fetch import fetch
        fetch(args.src, args.start, args.duration, args.out, args.hlg, args.ffmpeg)
    elif c == 'sync-audio':
        from .sync import audio_offset
        ra = _range(args.ref_range) if args.ref_range else (None, None)
        ca = _range(args.cam_range) if args.cam_range else (None, None)
        offset, score = audio_offset(args.ref, args.cam, ra[0], None if ra[1] is None else ra[1] - ra[0],
                                     ca[0], None if ca[1] is None else ca[1] - ca[0], args.max_lag)
        warn = '' if score > 0.5 else '  ← низкая уверенность, проверьте другим окном'
        print(f'время в {Path(args.cam).name} = время в {Path(args.ref).name} {offset:+.3f} с (score {score:.3f}){warn}')
        print(f'advance для этой камеры в плане: {-offset:+.3f}')
    elif c == 'sync-motion':
        from .sync import motion_offset
        shift, corr, nxt = motion_offset(args.a, args.crop_a, args.b, args.crop_b, args.max_lag)
        print(f'b = a {shift:+.3f} с (кадров {round(shift * 30):+d}), корреляция {corr:.3f}, следующий пик {nxt:.3f}')
    elif c == 'words':
        from .speech import words
        words(args.audio, args.out, args.model)
    elif c == 'slice-words':
        from .speech import slice_words
        slice_words(args.full, args.start, args.duration, args.out)
    elif c == 'show-words':
        from .speech import show_words
        show_words(args.words, args.lo, args.hi)
    elif c == 'hear':
        from .speech import hear
        hear(args.spots, args.model)
    elif c == 'spots-from-srt':
        from .speech import spots_from_srt
        spots_from_srt(_project(args), args.specs)
    elif c == 'envelope':
        envelope(args.file, args.ranges)
    elif c == 'find-pause':
        find_pause(args.file, args.times, args.win)
    elif c == 'mouths':
        from .visuals import mouths
        a, b = _range(args.range)
        mouths(args.out, args.step, a, b, [spec.split(':', 2) for spec in args.cams])
    elif c == 'frames':
        from .visuals import frames
        project = _project(args)
        frames(project.reel_path(project.plan(args.id)), args.out, args.times)
    elif c == 'clip-levels':
        from .fixes import clip_levels
        clip_levels(_project(args), args.id)
    elif c == 'split':
        from .visuals import split_screen
        split_screen(args.out, _cam(args.top), _cam(args.bottom), args.start)
    elif c == 'sticker':
        from .project import read_json
        from .style import Style
        root = Path(args.project).resolve()
        from .visuals import sticker
        sticker(args.out, args.parts, Style.from_meta(read_json(root / 'pack.json', {}), root))
    elif c == 'serve':
        from .serve import serve
        serve(args.dir or _project(args).out, args.port)
    elif c == 'demo':
        from .demo import build_project, run_all
        root = build_project(args.dir)
        print('Демо-проект:', root)
        if not args.no_run:
            return 0 if run_all(root) else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
