"""The publication pack: covers, captions, a watch page with a «copy caption»
button and a short start file. Texts come from rows.json (one card per reel)
and pack.json (pack title, intro and the suggested order)."""
from __future__ import annotations

import html
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from . import media
from .project import Project, write_json
from .timeline import build, srt

ROW_FIELDS = ('id', 'title', 'caption', 'cover_source', 'cover_time', 'cover_kicker', 'cover_title')


def plural(n, one, few, many):
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def cover(project: Project, plan: dict, row: dict, video) -> str:
    """A frame of the chosen camera with the pack grade, a dark gradient and
    the kicker and title at the bottom."""
    style = project.style
    src = plan['sources'][row['cover_source']]
    frame = project.tmp(plan['id']) / 'cover_base.png'
    vf = style.grade
    if src.get('crop'):
        vf = 'crop=' + src['crop'] + ',scale=1080:1920:flags=lanczos,' + vf
    subprocess.run([media.FFMPEG, '-v', 'error', '-y', '-ss', str(row['cover_time']), '-i', src['video'],
                    '-frames:v', '1', '-vf', vf, str(frame)], check=True)
    im = Image.open(frame).convert('RGBA')
    arr = np.zeros((1920, 1080, 4), dtype=np.uint8)
    arr[:, :, :3] = style.shade
    arr[:, :, 3] = (230 * np.clip((np.arange(1920) - 800) / 620, 0, 1) ** .8).astype(np.uint8)[:, None]
    im = Image.alpha_composite(im, Image.fromarray(arr))
    d = ImageDraw.Draw(im)
    d.text((80, 1110), row['cover_kicker'], font=style.font(28, 620), fill=style.accent, anchor='lt')
    f = style.font(70, 730)
    for n, line in enumerate(row['cover_title']):
        if d.textlength(line, font=f) >= 910:
            raise ValueError(f'{plan["id"]}: строка обложки не влезает: {line!r}')
        d.text((80, 1180 + n * 86), line, font=f, fill=style.white if n == 0 else style.accent, anchor='lt')
    path = project.out / f'{video.stem}_обложка.jpg'
    im.convert('RGB').save(path, quality=95, subsampling=0)
    return path.name


CSS = ('*{box-sizing:border-box}body{margin:0;background:#eff3f1;color:#19252b;'
       'font:17px/1.5 Manrope,system-ui,sans-serif}@font-face{font-family:Manrope;src:url(%(font)s)}'
       'main{max-width:1230px;margin:auto;padding:42px 28px}h1{font-size:42px;line-height:1.15;margin:12px 0 14px}'
       'header p{max-width:820px;color:#506369}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:28px;margin-top:32px}'
       'article{background:white;border:1px solid #d6dfdc;border-radius:15px;overflow:hidden}'
       'video{width:100%%;aspect-ratio:9/16;display:block;background:#19252b;max-height:78vh}'
       '.body{padding:25px}small{color:#507075;font-size:13px}h2{font-size:25px;line-height:1.25;margin:10px 0 12px}'
       'h3{font-size:16px;font-weight:650;margin-top:25px}.audience{color:#5d7478;font-size:14px}'
       '.note{border-left:3px solid #84aeb4;padding-left:14px}.caption{white-space:pre-line}'
       'nav{display:flex;gap:8px;flex-wrap:wrap}a{color:#28505b}'
       'nav a,button{background:#edf4f2;border:0;border-radius:7px;padding:10px 12px;font:14px inherit;cursor:pointer;'
       'text-decoration:none;color:#28505b}button{background:#24424b;color:white;font:14px Manrope,system-ui,sans-serif}'
       'a:focus-visible,button:focus-visible{outline:3px solid #22718b;outline-offset:3px}'
       'footer{margin-top:32px;color:#536c72}'
       '@media(max-width:760px){.grid{grid-template-columns:1fr}main{padding:25px 18px}h1{font-size:32px}.body{padding:20px}}')
SCRIPT = ("async function copyCaption(button,id){const text=document.getElementById(id).innerText;"
          "try{await navigator.clipboard.writeText(text);button.textContent='Скопировано';}"
          "catch(e){const range=document.createRange();range.selectNodeContents(document.getElementById(id));"
          "const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);"
          "button.textContent='Текст выделен — скопируйте';}}")


def package(project: Project):
    rows = project.rows()
    if not rows:
        raise SystemExit('Нет rows.json: карточки публикаций (id, title, caption, cover_*) пишутся туда')
    for r in rows:
        missing = [k for k in ROW_FIELDS if k not in r]
        if missing:
            raise SystemExit(f'rows.json, {r.get("id")}: нет полей {", ".join(missing)}')
    plans = {p['id']: p for p in project.plans()}
    title = project.text('title', 'Рилсы')
    cards, captions, summary = [], [], []
    for r in rows:
        i = r['id']
        p = plans[i]
        video = project.reel_path(p)
        if not video.exists():
            raise SystemExit(f'{i}: ролик ещё не собран — {video.name}')
        (project.out / f'{p["name"]}.srt').write_text(srt(build(p).cues))
        r['duration'] = media.duration(video)
        r['video'] = video.name
        r['cover'] = cover(project, p, r, video)
        e = html.escape
        note, audience = r.get('note', ''), r.get('audience', '')
        captions.append(f'## {i}. {r["title"]}\n\n{r["caption"]}\n')
        summary.append(f'## {i}. {r["title"]} — {r["duration"]:.1f} с\n\n{note}\n\n'
                       f'[Смотреть ролик]({video.name})' + (f' · {audience}' if audience else '') + '\n')
        cards.append(
            f'<article><video controls playsinline preload="metadata" poster="{e(r["cover"])}" src="{e(video.name)}"></video>'
            f'<div class="body"><small>{i} · {r["duration"]:.1f} сек.</small><h2>{e(r["title"])}</h2>'
            f'<p class="audience">{e(audience)}</p><p class="note">{e(note)}</p>'
            f'<h3>Подпись к публикации</h3><p id="caption-{i}" class="caption">{e(r["caption"])}</p>'
            f'<nav><button type="button" onclick="copyCaption(this,\'caption-{i}\')">Скопировать подпись</button>'
            f'<a download href="{e(video.name)}">Видео</a><a download href="{e(r["cover"])}">Обложка</a></nav></div></article>')
    out = project.out
    n = len(rows)
    (out / 'ПОДПИСИ.md').write_text(f'# {project.text("captions_title", "Подписи к публикациям")}\n\n' + '\n'.join(captions))
    (out / 'НАЧАТЬ_ЗДЕСЬ.md').write_text(
        f'# {title}\n\n'
        + project.text('intro', f'{n} {plural(n, "готовый ролик", "готовых ролика", "готовых роликов")}.') + '\n\n'
        '[Смотреть все ролики](СМОТРЕТЬ.html) · [Подписи](ПОДПИСИ.md) · [Проверка качества](QC_REPORT.md)\n\n'
        + '\n'.join(summary)
        + ('\n' + project.meta['order'] + '\n' if project.meta.get('order') else '')
        + '\nВсе ролики — MP4, 1080 × 1920, 30 кадров/с. Публикация не выполнялась.\n')
    font = os.path.relpath(project.style.font_path, out)
    label = project.text('label', title.upper())
    heading = project.text('heading', f'{n} {plural(n, "ролик", "ролика", "роликов")}, готовых к публикации')
    page_intro = project.text('page_intro', 'Каждая реплика в субтитрах сверена со звуком; спорные слова не додуманы.')
    page = ('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>' + CSS % {'font': font} + '</style></head><body><main><header>'
            f'<small>{html.escape(label)}</small><h1>{html.escape(heading)}</h1><p>{html.escape(page_intro)}</p>'
            '</header><section class="grid">' + ''.join(cards) + '</section><footer>Субтитры уже в видео, SRT приложены отдельно.<br>'
            '1080 × 1920 · 30 кадров/с · Ничего не опубликовано.</footer></main><script>' + SCRIPT + '</script></body></html>')
    (out / 'СМОТРЕТЬ.html').write_text(page)
    write_json(project.cache / 'package_rows.json', rows, indent=2)
    print('Обложки, подписи и страница просмотра готовы:', out)
