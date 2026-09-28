"""Thin ffmpeg helpers shared by the render and the checks."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

FFMPEG = os.environ.get('REELS_FFMPEG', 'ffmpeg')
FFPROBE = os.environ.get('REELS_FFPROBE', 'ffprobe')


def run(args, log=None) -> subprocess.CompletedProcess:
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if log:
        Path(log).write_text(r.stderr)
    if r.returncode:
        raise RuntimeError(r.stderr[-3500:])
    return r


def probe(path) -> dict:
    return json.loads(subprocess.check_output(
        [FFPROBE, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]))


def duration(path) -> float:
    return float(subprocess.check_output(
        [FFPROBE, '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)]))


def pcm(path, sr=16000, start=None, dur=None, end=None, filters=None) -> np.ndarray:
    """Mono float samples of a file's audio; an empty array if it has none."""
    args = [FFMPEG, '-v', 'error']
    if start is not None:
        args += ['-ss', f'{start}']
    if end is not None:
        args += ['-to', f'{end}']
    if dur is not None:
        args += ['-t', f'{dur}']
    args += ['-i', str(path), '-vn', '-ac', '1', '-ar', str(sr)]
    if filters:
        args += ['-af', filters]
    args += ['-f', 's16le', '-']
    r = subprocess.run(args, capture_output=True)
    return np.frombuffer(r.stdout, dtype='<i2').astype(np.float32) / 32768.0


def levels_db(x: np.ndarray, hop: int) -> np.ndarray:
    """RMS level per hop in dBFS."""
    n = len(x) // hop * hop
    return 20 * np.log10(np.sqrt((x[:n].reshape(-1, hop) ** 2).mean(axis=1)) + 1e-9)


_levels_cache: dict = {}


def file_levels(path, sr=16000, hop=None) -> np.ndarray:
    hop = hop or sr // 100
    key = (str(path), sr, hop)
    if key not in _levels_cache:
        _levels_cache[key] = levels_db(pcm(path, sr), hop)
    return _levels_cache[key]


def loudnorm_stats(stderr: str) -> dict:
    return json.loads(re.findall(r'\{[^{}]*"input_i"[^{}]*\}', stderr, re.S)[-1])


def frame(path, t, vf=None, size=None):
    """One RGB frame as a numpy array (h, w, 3)."""
    args = [FFMPEG, '-v', 'error', '-ss', f'{max(0, t):.3f}', '-i', str(path), '-frames:v', '1']
    if vf:
        args += ['-vf', vf]
    args += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.run(args, capture_output=True, check=True).stdout
    if size is None:
        w, h = _frame_size(path, vf)
    else:
        w, h = size
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def _frame_size(path, vf):
    if vf:
        m = re.search(r'scale=(\d+):(\d+)', vf) or re.search(r'crop=(\d+):(\d+)', vf)
        if m:
            return int(m.group(1)), int(m.group(2))
    v = next(s for s in probe(path)['streams'] if s['codec_type'] == 'video')
    return int(v['width']), int(v['height'])
