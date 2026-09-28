"""Cut a fragment out of a long recording without downloading all of it.

The source can be a local file, an http(s) URL or a Yandex Disk path
(``disk:/…``, token in .env as YANDEX_DISK_OAUTH_TOKEN, see src/yadisk.py).
iPhone cameras that shot HLG need tone mapping to SDR; that uses the ``zscale``
filter, which not every ffmpeg build has — point REELS_FFMPEG_HLG (or --ffmpeg)
at one that does. Fragments of a camera whose offset is only estimated should
be cut with a margin (1.5 s on both sides worked) and synced afterwards."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import media

TONEMAP = ('zscale=t=linear:npl=200,format=gbrpf32le,zscale=p=bt709,'
           'tonemap=tonemap=hable:desat=2,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,'
           'sidedata=mode=delete,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709')


def resolve(src: str) -> str:
    if src.startswith('disk:/'):
        import sys
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root))
        try:
            from dotenv import load_dotenv
            load_dotenv(root / '.env')
        except ImportError:
            pass
        from src.yadisk import get_download_url
        return get_download_url(src)
    return src


def fetch(src: str, start: float, dur: float, out, hlg=False, ffmpeg=None, audio=True):
    out = Path(out)
    if out.exists() and out.stat().st_size > 100000:
        print('уже есть', out.name)
        return out
    binary = ffmpeg or (os.environ.get('REELS_FFMPEG_HLG') if hlg else None) or media.FFMPEG
    url = resolve(src)
    args = [binary, '-v', 'error']
    if '://' in url:
        args += ['-rw_timeout', '30000000']      # a stalled network read fails instead of hanging
    args += ['-ss', f'{start:.3f}', '-i', url, '-t', f'{dur:.3f}', '-map', '0:v:0']
    if audio:
        args += ['-map', '0:a:0?']
    if hlg:
        args += ['-vf', TONEMAP]
    args += ['-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-threads', '2', '-pix_fmt', 'yuv420p']
    if audio:
        args += ['-c:a', 'pcm_s24le', '-ar', '48000']
    args += ['-map_metadata', '-1', '-y', str(out)]
    r = subprocess.run(args, capture_output=True, timeout=3600)
    if r.returncode:
        raise RuntimeError(f'{out.name}: {r.stderr[-400:]!r}')
    print('сохранено', out.name, round(out.stat().st_size / 1e6, 1), 'МБ')
    return out
