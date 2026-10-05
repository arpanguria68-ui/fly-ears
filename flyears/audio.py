"""Audio in: every file goes through the same path, so the format can't give the class away.

decode()  ffmpeg -> mono float32 at a fixed rate (optionally through one shared MP3 codec first,
          so a FLAC-vs-MP3 or bitrate difference between the two folders can't leak)
clips()   fixed-length windows at fixed positions in each song (not chosen by content)
level()   every clip to the same loudness (AI masters are often louder; loudness is not the question)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

EAR_SR = 8000            # the fly's ear works below ~1 kHz; 8 kHz keeps everything up to 4 kHz
FULL_SR = 22050          # only for the full-band audio baseline (what an ordinary detector could use)
AUDIO_TYPES = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma", ".aiff", ".aif"}
TARGET_RMS = 0.1


def decode(path: Path, sr: int, same_codec: bool = False) -> np.ndarray:
    """The whole file as mono float32 samples at `sr`."""
    src = str(path)
    if same_codec:                                       # everything through one MP3 encoder first
        mp3 = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame",
                              "-b:a", "128k", "-f", "mp3", "pipe:1"], capture_output=True, check=True).stdout
        raw = subprocess.run(["ffmpeg", "-v", "error", "-f", "mp3", "-i", "pipe:0", "-ac", "1", "-ar", str(sr),
                              "-f", "f32le", "pipe:1"], input=mp3, capture_output=True, check=True).stdout
    else:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"],
                             capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def clips(x: np.ndarray, sr: int, seconds: float, n: int) -> list[np.ndarray]:
    """n windows of `seconds`, centred at evenly spaced points of the song (1/(n+1), 2/(n+1), ...).
    Positions depend only on the song's length, never on its content."""
    L = int(seconds * sr)
    if len(x) < L:
        return []
    starts = [int((len(x) - L) * (k + 1) / (n + 1)) for k in range(n)]
    return [x[s:s + L] for s in starts]


def level(x: np.ndarray) -> np.ndarray:
    """Same loudness for every clip (RMS), so the fly can't tell classes apart by volume."""
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return (x * (TARGET_RMS / rms)).astype(np.float32) if rms > 1e-6 else x


def songs_in(folder: Path) -> dict[str, list[Path]]:
    """{class name: audio files} from the subfolders of `folder` (e.g. human/ and ai/)."""
    out = {}
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(f for f in d.rglob("*") if f.suffix.lower() in AUDIO_TYPES)
        if files:
            out[d.name] = files
    return out
