"""Visual language of the reels: one font, one accent colour, a dark readable
caption plate and a light colour grade. Change it per project in pack.json."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

PACKAGE = Path(__file__).resolve().parent

GRADE = ('colorbalance=rs=-0.010:rm=-0.018:gs=-0.008:gm=-0.016:bs=0.024:bm=0.045:bh=0.012,'
         'eq=contrast=1.05:brightness=0.006:saturation=0.95:gamma=1.01')

# Manrope (SIL OFL 1.1) is the intended font; it is not shipped with the package,
# see interview_reels/README.md. Any font with Cyrillic works as a fallback.
FALLBACK_FONTS = [
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/Library/Fonts/Arial.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
]
EMOJI_FONTS = [
    '/System/Library/Fonts/Apple Color Emoji.ttc',
    '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf',
    '/usr/share/fonts/noto/NotoColorEmoji.ttf',
]


def _first_existing(paths):
    for p in paths:
        if p and Path(p).expanduser().is_file():
            return str(Path(p).expanduser())
    return None


def find_font(root: Path | None = None, explicit: str | None = None) -> str:
    candidates = [explicit, os.environ.get('REELS_FONT')]
    if root is not None:
        candidates += [root / 'Manrope.ttf', root / 'assets' / 'Manrope.ttf']
    candidates += [PACKAGE / 'assets' / 'Manrope.ttf']
    candidates += sorted(Path('~/Library/Fonts').expanduser().glob('Manrope*.ttf'))
    candidates += FALLBACK_FONTS
    found = _first_existing(str(c) if c else None for c in candidates)
    if not found:
        raise SystemExit('Не найден шрифт: положите Manrope.ttf в папку проекта или укажите REELS_FONT')
    return found


@lru_cache(maxsize=None)
def load_font(path: str, size: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path, size)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight])   # Manrope is a variable font (wght 200–800)
        except (OSError, AttributeError):
            pass                                # a static fallback font keeps its own weight
    return f


@dataclass
class Style:
    font_path: str
    emoji_path: str | None = None
    white: tuple = (245, 246, 242, 255)
    accent: tuple = (174, 219, 224, 255)
    plate: tuple = (17, 24, 29)        # caption plate and stickers, RGB
    shade: tuple = (17, 23, 29)        # title and cover gradients, RGB
    grade: str = GRADE

    @classmethod
    def from_meta(cls, meta: dict, root: Path) -> 'Style':
        font = meta.get('font')
        if font and not Path(font).is_absolute():
            font = str(root / font)
        style = cls(font_path=find_font(root, font),
                    emoji_path=_first_existing([meta.get('emoji_font'), os.environ.get('REELS_EMOJI_FONT'),
                                                *EMOJI_FONTS]))
        if meta.get('accent'):
            style.accent = tuple(meta['accent']) + ((255,) if len(meta['accent']) == 3 else ())
        if meta.get('grade') is not None:
            style.grade = meta['grade']
        return style

    def font(self, size: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
        return load_font(self.font_path, size, weight)

    # the sizes the pack was designed with
    def caption_font(self):
        return self.font(58, 610)

    def kicker_font(self):
        return self.font(27, 620)

    def title_font(self):
        return self.font(54, 740)
