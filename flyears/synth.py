"""Synthetic test sets that check the method before any real song is used.

null:   both folders come from the SAME generator. Every feature set must score AUC ~0.5 here; if
        anything scores higher, the pipeline leaks (format, order, folder) and real results are void.
signal: the folders differ in one known way, inside the fly's hearing range: "ai" songs are
        machine-tight (no timing jitter, constant loudness per note), "human" songs are played
        (timing jitter, varying note loudness). A working pipeline must find this.

Songs are written as WAV and then go through exactly the same path as real files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR = 22050
SECONDS = 20.0


def song(rng, tight: bool) -> np.ndarray:
    n = int(SECONDS * SR)
    out = np.zeros(n)
    bpm = rng.uniform(90, 130)
    beat = 60 / bpm
    root = rng.uniform(110, 220)
    scale = root * 2 ** (np.array([0, 2, 3, 5, 7, 8, 10, 12, 14, 15]) / 12)
    jitter, vel_sd = (0.0, 0.0) if tight else (0.018, 0.25)
    t = np.arange(int(0.6 * SR)) / SR
    k = np.arange(int(0.25 * SR)) / SR
    for i in range(int(SECONDS / beat * 2)):              # eighth notes
        at = i * beat / 2 + rng.normal(0, jitter)
        s = int(max(0, at) * SR)
        vel = max(0.2, 1 + rng.normal(0, vel_sd))
        f0 = rng.choice(scale)
        note = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3)) * np.exp(-t * 6) * 0.25 * vel
        out[s:s + len(note)] += note[: n - s]
        if i % 2 == 0:                                    # kick on the beat
            kick = np.sin(2 * np.pi * np.cumsum(60 + 90 * np.exp(-k * 30)) / SR) * np.exp(-k * 12) * 0.6 * vel
            out[s:s + len(kick)] += kick[: n - s]
    hat = rng.standard_normal(n) * 0.02                   # broadband hiss, the same for both classes
    out += hat
    return (out / np.max(np.abs(out)) * 0.9).astype(np.float32)


def make(folder: Path, kind: str, per_class: int, seed: int = 0) -> Path:
    """folder/<kind>/{human,ai}/*.wav"""
    rng = np.random.default_rng(seed)
    root = folder / kind
    for cls in ("human", "ai"):
        (root / cls).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            tight = kind == "signal" and cls == "ai"
            wavfile.write(root / cls / f"{cls}_{i:02d}.wav", SR, song(rng, tight))
    return root
