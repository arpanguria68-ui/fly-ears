"""The fly's ear: sound -> Johnston's organ (JO-A and JO-B, the antenna's sound-sensing neurons).

The antenna's arista vibrates with air particle velocity; JO-A and JO-B neurons respond to that
vibration. Here the clip is split into frequency bands inside the fly's hearing range, each band's
loudness is taken every brain step (20 ms) and injected as voltage into its JO neurons.

Assumption (stated, and tested by the "flat" ablation): the ear is tonotopic in a coarse way,
JO-B more for low frequencies and JO-A more for high ones (broadly following Kamikouchi et al.
2009, Nature 458:165; Matsuo et al. 2014). Bands are assigned to JO subtypes in a fixed order,
never by looking at any song or label.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .audio import EAR_SR, TARGET_RMS

BAND_EDGES = np.geomspace(80, 1200, 9)       # 8 bands, 80 Hz - 1.2 kHz (log spaced)
N_BANDS = len(BAND_EDGES) - 1
LOW_BANDS, HIGH_BANDS = range(0, 4), range(4, 8)
EAR_CAP = 0.8                                 # max extra voltage per step (as flytalk's ear)
REF = TARGET_RMS / np.sqrt(N_BANDS)           # a fixed level scale: an even spread of a level clip -> 1


def _filters():
    return [signal.butter(4, (lo, hi), "bandpass", fs=EAR_SR, output="sos")
            for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]


_SOS = _filters()


def envelopes(x: np.ndarray, dt: float = 0.020) -> np.ndarray:
    """(steps, bands) loudness per brain step, compressed to 0..1 by a fixed scale (no data fitting)."""
    per = int(round(dt * EAR_SR))
    steps = len(x) // per
    out = np.empty((steps, N_BANDS), np.float32)
    x = x.astype(np.float64)
    fade = np.hanning(2 * per)                             # 20 ms fades: cutting a clip out of a song makes
    x[:per] *= fade[:per]                                  # a click, which would sound in every band
    x[-per:] *= fade[per:]
    for b, sos in enumerate(_SOS):
        y = signal.sosfiltfilt(sos, x)[: steps * per]
        rms = np.sqrt((y.reshape(steps, per) ** 2).mean(1))
        out[:, b] = np.sqrt(np.clip(rms / REF, 0, 4)) / 2      # sqrt compression, 4x REF -> 1
    return out


def ear_map(brain, mode: str = "tonotopic") -> tuple[np.ndarray, np.ndarray]:
    """JO-A/JO-B neuron indices and the band each one hears.
    tonotopic: JO-B subtypes share the 4 low bands, JO-A subtypes the 4 high ones (round robin in
    a fixed order of type, then neuron index). flat: every ear neuron hears all bands (mean)."""
    types = sorted({str(t) for t in np.unique(brain.cell_type) if str(t).startswith(("JO-A", "JO-B"))})
    cells, bands = [], []
    for prefix, group in (("JO-B", LOW_BANDS), ("JO-A", HIGH_BANDS)):
        idx = np.concatenate([np.sort(brain.cells([t])) for t in types if t.startswith(prefix)])
        cells.append(idx)
        bands.append(np.array([group[i % len(group)] for i in range(len(idx))]))
    cells, bands = np.concatenate(cells), np.concatenate(bands)
    if mode == "flat":
        bands = np.full(len(cells), -1)
    return cells, bands


def drive(env: np.ndarray, bands: np.ndarray) -> np.ndarray:
    """Voltage for each ear neuron at one step: env (bands, flies) -> (neurons, flies)."""
    if bands[0] == -1:
        return np.repeat(env.mean(0, keepdims=True), len(bands), 0) * EAR_CAP
    return env[bands] * EAR_CAP


def injections(cells: np.ndarray, bands: np.ndarray, env: np.ndarray) -> list:
    """The same drive as (neuron set, one value per fly) pairs, one per band, which every
    FlyBrain version accepts: env (bands, flies)."""
    if bands[0] == -1:
        return [(cells, env.mean(0) * EAR_CAP)]
    return [(cells[bands == b], env[b] * EAR_CAP) for b in np.unique(bands)]
