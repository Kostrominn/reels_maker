"""Human-readable QC summary (QC_REPORT.md) from the JSON results of the checks.

Manual findings go to qc_notes.json in the project, per reel id:
``{"06": {"remeasured": "...", "still_ok": "...", "visual": "..."}}`` —
``remeasured``: sync windows that failed and were re-measured by a more
reliable method; ``still_ok``: a «freeze» that turned out to be a still scene;
``visual``: how a camera without sound was synced."""
from __future__ import annotations

import json

from .project import Project

NAMES = {
    'decode_ok': 'файл декодируется целиком',
    'geometry': 'кадр 1080 × 1920',
    'video_codec': 'видео H.264',
    'audio_codec': 'звук AAC',
    'audio_rate': 'частота звука 48 кГц',
    'loudness': 'громкость в пределах ±1 LUFS от −16',
    'true_peak': 'пик не выше −1 dBTP',
    'av_duration': 'звук и видео одной длины',
    'frame_rate': '30 кадров в секунду',
    'sdr_color': 'цвет SDR / BT.709',
    'no_black': 'нет чёрных провалов',
    'no_freeze': 'нет подвисших кадров',
}
VISUAL = ('Часть кадров взята с камеры без собственного звука (или из разделённого экрана): картинка совмещена '
          'со звуком по замеру, ниже проверено, что звук взят с нужного места.')
FRAME = 1 / 30


def _load(project: Project, name: str):
    path = project.out / name
    return json.loads(path.read_text()) if path.exists() else []


def report(project: Project) -> bool:
    checks = _load(project, 'quality_checks.json')
    sync = {x['id']: x for x in _load(project, 'sync_verification.json')}
    caps = _load(project, 'caption_verification.json')
    notes = project.qc_notes()
    if not checks:
        raise SystemExit('Нет quality_checks.json — сначала check-outputs')
    lines = [f'# {project.text("qc_title", "Проверка качества")}', '',
             'Проверено автоматически на готовых файлах: параметры контейнера, громкость, '
             'совпадение звука с картинкой и читаемость каждого субтитра на реальном кадре.', '']
    all_ok = True
    for c in checks:
        i = c['id']
        note = notes.get(i, {})
        lines += [f'## {c["file"]}', '',
                  f'Длительность {c["duration"]:.1f} с · {c["megabytes"]} МБ · '
                  f'громкость {float(c["loudness"]["input_i"]):.1f} LUFS · '
                  f'пик {float(c["loudness"]["input_tp"]):.1f} dBTP', '']
        bad = [NAMES.get(k, k) for k, v in c['checks'].items()
               if not v and not (k == 'no_freeze' and note.get('still_ok'))]
        if bad:
            all_ok = False
            lines += ['Не прошло: ' + ', '.join(bad), '']
        else:
            lines += ['Технические проверки пройдены: ' + ', '.join(NAMES[k] for k in c['checks']) + '.', '']
        if note.get('still_ok'):
            lines += [note['still_ok'], '']
        s = sync.get(i)
        if s:
            # windows that compare the reel with ANOTHER camera's microphone correlate
            # weakly and are noisy, so the median and the confident windows decide
            solid = [w for w in s['windows'] if w['correlation'] >= 0.25]
            drift = max((abs(w['lag_seconds']) for w in solid), default=0.0)
            med = abs(s['median_lag_seconds'])
            off = max(drift, med) > FRAME + 1e-6
            verdict = 'в пределах одного кадра' if not off else 'ТРЕБУЕТ ПРОВЕРКИ'
            if off and note.get('remeasured'):
                verdict = 'перемерено вручную, см. ниже'
            elif off:
                all_ok = False
            if s.get('visual_sync_sources'):
                lines += [note.get('visual', VISUAL), '']
            lines += [f'Синхронность звука и картинки: медиана {med * 1000:.0f} мс, '
                      f'максимум по уверенным окнам {drift * 1000:.0f} мс '
                      f'({len(solid)} из {len(s["windows"])}) — {verdict}.', '']
            if note.get('remeasured'):
                lines += [note['remeasured'], '']
        mine = [x for x in caps if x['id'] == i]
        failed = [x for x in mine if not x['passed']]
        if failed:
            all_ok = False
            lines += [f'Субтитры: {len(mine) - len(failed)} из {len(mine)} читаются уверенно. '
                      'Проверить вручную: ' + '; '.join(f'{x["time"]:.1f} с' for x in failed), '']
        elif mine:
            lines += [f'Субтитры: все {len(mine)} проверены на кадре, текст виден полностью.', '']
    lines += ['## Чего проверка не покрывает', '',
              'Автоматические проверки не оценивают смысл, интонацию и то, интересно ли смотреть. '
              'Текст субтитров сверяется с расшифровкой готового звука (verify-text), но живого '
              'прослушивания это не заменяет. Публикация не выполнялась.', '']
    (project.out / 'QC_REPORT.md').write_text('\n'.join(lines))
    print('QC_REPORT.md готов', '— всё чисто' if all_ok else '— ЕСТЬ ЗАМЕЧАНИЯ')
    return all_ok
