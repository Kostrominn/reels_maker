"""Offsets between cameras.

``audio_offset``: cross-correlation of the sound envelopes. Use it whenever both
files have sound; on real material two independent windows agreed within 10 ms.

``motion_offset``: for a camera that recorded no sound. Correlates the amount of
motion in one frame region with the motion of the same thing seen by the other
camera (e.g. the guest's whole frame against his hands in the host's shot). On
a known pair it gave a peak of 0.69 against a background of 0.13; always look at
the next best peak printed next to it. Methods based on the mouth alone were
wrong on the test pair and are not included.

Note: iPhone creation times are rounded to a second, so file metadata gives only
a starting point. Recordings can also drift or jump: measure each fragment.
"""
from __future__ import annotations

import subprocess

import numpy as np

from . import media

SR, HOP = 16000, 160
FPS = 30


def _envelope(x, hop):
    n = len(x) // hop * hop
    e = np.abs(x[:n]).reshape(-1, hop).mean(1)
    return e - e.mean()


def audio_offset(a_path, b_path, a_start=None, a_dur=None, b_start=None, b_dur=None,
                 max_lag=None, hop=HOP):
    """Return (offset, score): time in B = time in A + offset."""
    a0 = float(a_start or 0.0)
    b0 = float(b_start or 0.0)
    a = _envelope(media.pcm(a_path, SR, start=a_start, dur=a_dur), hop)
    b = _envelope(media.pcm(b_path, SR, start=b_start, dur=b_dur), hop)
    n = max(len(a), len(b))
    size = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    cc = np.concatenate([cc[-n:], cc[:n]])
    lags = np.arange(-n, n) * hop / SR
    if max_lag is not None:
        keep = np.abs(lags) <= max_lag
        cc, lags = cc[keep], lags[keep]
    k = int(np.argmax(cc))
    score = float(cc[k] / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) or 1.0))
    # cc[m] = sum a[n]·b[n+m]: the moment at A-local t sits at B-local t + m;
    # convert to whole-file times
    return round(float(lags[k]) + b0 - a0, 4), score


def motion(path, crop='crop=iw:ih:0:0'):
    raw = subprocess.run([media.FFMPEG, '-v', 'error', '-i', str(path),
                          '-vf', f'fps={FPS},{crop},scale=64:64,format=gray', '-f', 'rawvideo', '-'],
                         capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 64, 64).astype(np.float32)
    x = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    x = x - np.convolve(x, np.ones(31) / 31, mode='same')
    return (x - x.mean()) / (x.std() + 1e-9)


def motion_offset(a_path, crop_a, b_path, crop_b, max_lag=4.0):
    """Return (shift, corr, next_peak): add ``shift`` to B's time to line it up with A."""
    a, b = motion(a_path, crop_a), motion(b_path, crop_b)
    lim = int(max_lag * FPS)
    res = []
    for lag in range(-lim, lim + 1):              # a frame i  <->  b frame i + lag
        x, y = a[max(0, -lag):], b[max(0, lag):]
        k = min(len(x), len(y))
        res.append((float(np.dot(x[:k], y[:k]) / k), lag))
    res.sort(reverse=True)
    s, lag = res[0]
    s2 = next(v for v, l in res if abs(l - lag) > 3)
    return lag / FPS, s, s2
