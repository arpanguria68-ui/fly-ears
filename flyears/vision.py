"""The fly's eyes: video frames -> photoreceptors, motion detectors (T4/T5) and looming detectors (LPLC2).

The video is taken as the fly's frontal view: the left half of the frame falls on the left eye,
the right half on the right eye, the frame's centre is straight ahead.

* Photoreceptors (6,006): brightness of the frame column at each one's azimuth (flybrain's eye).
* T4/T5 (13,581 cells, each placed in its eye column with its own preferred direction, from
  flybrain's columns.npz): optical flow at the cell's spot, projected on its preferred direction.
  Brightening edges drive T4 (ON pathway), darkening edges drive T5 (OFF pathway).
* LPLC2 (looming): expansion of the flow field on each side.

Fixed scales, never fitted to any video or label. Modelling choices, not measurements: the frame
covers the frontal field only, and flow is a local Lucas-Kanade estimate (the fly's own elementary
motion detection happens inside T4/T5, which here receive its result).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy import ndimage

FLOW_W, FLOW_H = 160, 90          # the frame size flow is measured at
MOTION_GAIN = 0.4                 # volts per pixel/frame of preferred-direction motion
LOOM_GAIN = 1.5                   # volts per pixel/frame of four-way outward motion (fast approach -> cap)
CAP = 0.8                         # most extra voltage per step on any cell (as the ear)
POLARITY_FLOOR = 0.3              # T4 still answers dark edges weakly, T5 bright ones
LEVELS = 16                       # drive levels used to inject cell-by-cell drive (see level_sets)


def fly_data() -> Path:
    return Path(os.environ.get("FLY_DATA") or Path.home() / "fly-data")


def load_columns() -> dict | None:
    f = fly_data() / "columns.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {k: d[k] for k in d.files}


def flow(prev: np.ndarray, cur: np.ndarray, sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lucas-Kanade optical flow (u right, v down, pixels per frame) and the brightness change It."""
    a, b = prev.astype(np.float32) / 255, cur.astype(np.float32) / 255
    m = (a + b) / 2
    Ix = ndimage.sobel(m, 1) / 8
    Iy = ndimage.sobel(m, 0) / 8
    It = b - a
    g = lambda x: ndimage.gaussian_filter(x, sigma)  # noqa: E731
    Sxx, Syy, Sxy, Sxt, Syt = g(Ix * Ix), g(Iy * Iy), g(Ix * Iy), g(Ix * It), g(Iy * It)
    det = Sxx * Syy - Sxy ** 2
    tr = Sxx + Syy
    lam_min = tr / 2 - np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    ok = lam_min > 1e-5                               # enough texture to tell the direction
    safe = np.where(ok, det, 1.0)
    u = np.where(ok, (-Syy * Sxt + Sxy * Syt) / safe, 0.0)
    v = np.where(ok, (Sxy * Sxt - Sxx * Syt) / safe, 0.0)
    return np.clip(u, -8, 8), np.clip(v, -8, 8), It


class Eyes:
    def __init__(self, brain, cols: dict | None = None):
        self.brain = brain
        cols = cols if cols is not None else load_columns()
        self.has_motion = cols is not None
        if self.has_motion:
            self.cells = cols["cells"].astype(np.int64)
            left = cols["eye"] == "L"
            front, up = cols["front"].astype(np.float32), cols["up"].astype(np.float32)
            x = np.where(left, 0.5 - 0.5 * front, 0.5 + 0.5 * front)       # front of the eye = frame centre
            y = 1 - up
            self.px = np.clip(np.round(x * (FLOW_W - 1)), 0, FLOW_W - 1).astype(int)
            self.py = np.clip(np.round(y * (FLOW_H - 1)), 0, FLOW_H - 1).astype(int)
            self.sign = np.where(left, 1.0, -1.0).astype(np.float32)        # frame +x is "front" for the left eye
            self.pref = cols["pref"].astype(np.float32)
            self.on = np.char.startswith(cols["types"].astype(str), "T4")
            self.left = left
        ct = brain.cell_type.astype(str)
        self.lplc2 = {s: brain.cells(["LPLC2"], side=s) for s in "LR"}
        self.az = np.asarray(brain.azimuth, np.float32)
        self.photo_x = np.clip(np.round((0.5 + 0.5 * self.az) * (FLOW_W - 1)), 0, FLOW_W - 1).astype(int)
        self.prev = None
        self.last = {"motion": None, "loom": (0.0, 0.0)}
        del ct

    def photoreceptors(self, frame: np.ndarray) -> np.ndarray:
        """0..1 per photoreceptor: the mean brightness of the frame column at its azimuth."""
        return (frame.mean(0) / 255)[self.photo_x].astype(np.float32)

    def see(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """One video frame (FLOW_H, FLOW_W uint8): photoreceptor drive and the injections."""
        photo = self.photoreceptors(frame)
        inject = []
        if self.prev is not None:
            u, v, It = flow(self.prev, frame)
            if self.has_motion:
                fu, fv = u[self.py, self.px], v[self.py, self.px]
                front, upw = fu * self.sign, -fv
                along = front * self.pref[:, 0] + upw * self.pref[:, 1]
                it = ndimage.gaussian_filter(It, 2)[self.py, self.px]
                on_share = np.clip(0.5 + 20 * it, 0, 1)                       # brightening -> ON (T4)
                share = np.where(self.on, on_share, 1 - on_share)
                polarity = POLARITY_FLOOR + (1 - POLARITY_FLOOR) * share       # mostly, not only, its polarity
                drive = np.clip(along, 0, None) * MOTION_GAIN * polarity
                drive = np.minimum(drive, CAP).astype(np.float32)
                self.last["motion"] = drive
                inject += level_sets(self.cells, drive)
            loom = [float(np.clip(looming(u, v, side) * LOOM_GAIN, 0, CAP)) for side in "LR"]
            self.last["loom"] = tuple(loom)
            for s, amt in zip("LR", loom):
                if amt > 0 and len(self.lplc2[s]):
                    inject.append((self.lplc2[s], amt))
        self.prev = frame
        return photo, inject


LOOM_R = 22                        # px: half-size of an LPLC2 receptive field at FLOW_W x FLOW_H


def looming(u: np.ndarray, v: np.ndarray, side: str) -> float:
    """LPLC2-like: edges moving OUTWARD in all four directions around a point (up above it, down
    below, left on its left, right on its right; Klapoetke et al. 2017). The weakest of the four
    outward motions counts, so sliding motion (outward on one side, inward on the other) gives ~0
    and only expansion drives it. Best point on that side of the view, pixels per frame."""
    H, W = u.shape
    xs = np.linspace(W * (0.12 if side == "L" else 0.55), W * (0.45 if side == "L" else 0.88), 4).astype(int)
    ys = np.linspace(H * 0.25, H * 0.75, 3).astype(int)
    R = LOOM_R
    best = 0.0
    for cy in ys:
        for cx in xs:
            arms = [-v[max(0, cy - R):cy, max(0, cx - 4):cx + 5],          # above: moving up
                    v[cy + 1:cy + R + 1, max(0, cx - 4):cx + 5],           # below: moving down
                    -u[max(0, cy - 4):cy + 5, max(0, cx - R):cx],          # left: moving left
                    u[max(0, cy - 4):cy + 5, cx + 1:cx + R + 1]]           # right: moving right
            if all(a.size for a in arms):
                best = max(best, min(float(a.mean()) for a in arms))
    return best


def level_sets(cells: np.ndarray, drive: np.ndarray, levels: int = LEVELS) -> list:
    """Cell-by-cell drive as a few (cells, one voltage) injections, which every FlyBrain version
    accepts: drive rounded to `levels` steps between 0 and CAP (error at most CAP / (2 * levels))."""
    q = np.round(drive / CAP * levels).astype(int)
    return [(cells[q == k], k * CAP / levels) for k in np.unique(q) if k > 0]
