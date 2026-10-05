"""Turn a clip (or the brain's response to it) into a fixed-length feature vector.

response_stats() is used for BOTH the ear's input (band loudness) and the brain's output (group
spike rates), so "does the brain add anything beyond what its ear received" compares like with
like: same summary, same classifier, same splits.

audio_stats() is the ordinary full-band baseline: what a plain audio detector could use, including
frequencies the fly can't hear (AI-generation artefacts often sit high up).
"""
from __future__ import annotations

import numpy as np

from .audio import FULL_SR

BIN = 5                         # steps per bin (100 ms at 20 ms steps)


def _band_power(x: np.ndarray, rate: float, lo: float, hi: float) -> np.ndarray:
    """Share of each column's variance between lo and hi Hz (its modulation at those rhythms)."""
    x = x - x.mean(0)
    p = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / rate)
    tot = p[f > 0].sum(0) + 1e-12
    return p[(f >= lo) & (f < hi)].sum(0) / tot


def response_stats(series: np.ndarray, dt: float = 0.020) -> np.ndarray:
    """series (steps, channels) -> per channel: mean, std over 100 ms bins, and how much of its
    variation follows slow rhythms (0.5-4 Hz: beats, phrases) and faster ones (4-12 Hz)."""
    steps = len(series) - len(series) % BIN
    binned = series[:steps].reshape(-1, BIN, series.shape[1]).mean(1)
    rate = 1 / (dt * BIN)
    return np.concatenate([binned.mean(0), binned.std(0),
                           _band_power(binned, rate, 0.5, 4.0), _band_power(binned, rate, 4.0, 12.0)])


LOG_BANDS = np.geomspace(50, 11000, 25)


def audio_stats(x: np.ndarray, sr: int = FULL_SR) -> np.ndarray:
    """Full-band spectral statistics of a clip (sr 22.05 kHz)."""
    n, hop = 2048, 512
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    P = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12
    f = np.fft.rfftfreq(n, 1 / sr)
    bands = np.stack([P[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in zip(LOG_BANDS[:-1], LOG_BANDS[1:])], 1)
    lb = np.log(bands)
    total = P.sum(1)
    centroid = (P * f).sum(1) / total
    cum = np.cumsum(P, 1)
    rolloff = f[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]
    flat = np.exp(np.log(P).mean(1)) / P.mean(1)
    hf = P[:, f >= 8000].sum(1) / total
    flux = np.r_[0, np.sqrt((np.diff(lb, axis=0).clip(0) ** 2).sum(1))]
    per = [centroid, rolloff, flat, hf, flux]
    return np.concatenate([lb.mean(0), lb.std(0)] + [np.r_[v.mean(), v.std()] for v in per])
