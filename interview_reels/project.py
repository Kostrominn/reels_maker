"""A project is one folder: plans.json plus optional rows.json (publication
cards), pack.json (pack texts and settings) and qc_notes.json (manual QC notes).

Media paths in plans may be absolute or relative to the project folder.
Finished reels go to ``out`` (pack.json "out", default ``<project>/out``),
intermediate files to ``cache`` (default ``<project>/.cache``)."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from .style import Style


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise SystemExit(f'Нет файла {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, data, indent=1):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=indent) + '\n', encoding='utf-8')


@dataclass
class Project:
    root: Path
    out: Path
    cache: Path
    meta: dict
    style: Style

    @classmethod
    def open(cls, root='.', out=None, cache=None) -> 'Project':
        root = Path(root).expanduser().resolve()
        if not (root / 'plans.json').exists():
            raise SystemExit(f'В {root} нет plans.json — это не папка проекта')
        meta = read_json(root / 'pack.json', {})
        out = Path(out).expanduser() if out else root / meta.get('out', 'out')
        cache = Path(cache).expanduser() if cache else root / '.cache'
        return cls(root, out.resolve(), cache.resolve(), meta, Style.from_meta(meta, root))

    # -- paths ---------------------------------------------------------------
    def path(self, p) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else self.root / p

    def reel_path(self, plan: dict) -> Path:
        return self.out / f'{plan["name"]}.mp4'

    def tmp(self, *parts) -> Path:
        d = self.cache.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- plans ---------------------------------------------------------------
    def raw_plans(self) -> list[dict]:
        """plans.json as written, for editing and saving back."""
        return read_json(self.root / 'plans.json')

    def save_plans(self, plans: list[dict]):
        write_json(self.root / 'plans.json', plans)

    def plans(self, ids=None) -> list[dict]:
        """Plans with every media path made absolute; ``ids`` narrows the list."""
        out = []
        for p in self.raw_plans():
            if ids and p['id'] not in ids:
                continue
            p = copy.deepcopy(p)
            for src in p['sources'].values():
                src['video'] = str(self.path(src['video']))
                src['audio'] = str(self.path(src['audio']))
            p['insets'] = [[s, e, str(self.path(f)), x, y] for s, e, f, x, y in p.get('insets') or []]
            out.append(p)
        if ids:
            missing = set(ids) - {p['id'] for p in out}
            if missing:
                raise SystemExit('Нет планов с id: ' + ', '.join(sorted(missing)))
        return out

    def plan(self, ident: str) -> dict:
        return self.plans([ident])[0]

    def rows(self) -> list[dict]:
        return read_json(self.root / 'rows.json', [])

    def qc_notes(self) -> dict:
        return read_json(self.root / 'qc_notes.json', {})

    def text(self, key: str, default: str) -> str:
        return self.meta.get(key) or default
