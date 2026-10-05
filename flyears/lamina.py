"""Fly-own vision: drive the lamina the way real lamina cells respond, and let the fly's wiring do the rest.

Why: in this spiking simulation the photoreceptors' signal stops at their first synapse. Photoreceptors
(R1-6) inhibit the lamina cells (histamine), and in a real fly those cells are graded: they carry vision as
voltage swings and do not need to spike. Here every neuron is a spiking cell resting below threshold, so
that inhibition can only keep the lamina silent, and nothing reaches the medulla or T4/T5.

What this does instead: L2 and L3 get what real L2/L3 signal: they depolarise when the light at their
column gets darker (light decrements, transient). From L2/L3 on, everything is the connectome: L2 -> Tm1,
Tm2, Tm4; L3 -> Tm9, Mi9; Tm1/Tm2/Tm9 -> T5 (all excitatory here), then lobula plate, LPLC2, the giant fiber.
No optical flow, no motion or looming detector of ours.

What it cannot do: bright edges (the ON pathway). L1 reaches Mi1/Tm3 through inhibitory synapses; in a fly
the ON response comes from releasing a constant inhibition, which this model's silent resting cells do not
have. So fly-own vision here sees dark edges moving (T5), not bright ones (T4).

Cell positions: each L2/L3 cell's eye column comes from the MaleCNS optic-column table and the wiring
(fly.ai's flybrain/columns.py method), in the same front/up coordinates as the T4/T5 map.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse

from .vision import CAP, FLOW_H, FLOW_W, fly_data

LAMINA = ("L2", "L3")
GAIN = 4.0                      # volts per unit of darkening (0..1 of full brightness); 0.2 darker -> the cap


def _columns_module():
    """flybrain.columns (fly.ai): from the installed flybrain, else from a fly.ai checkout that has it."""
    try:
        from flybrain import columns
        return columns
    except ImportError:
        pass
    here = Path(__file__).resolve().parents[2]
    for cand in [os.environ.get("FLYBRAIN_COLUMNS_SRC")] + [str(p) for p in here.glob("firefly/.claude/worktrees/*")]:
        if cand and (Path(cand) / "flybrain" / "columns.py").exists():
            sys.path.insert(0, cand)
            for m in [k for k in sys.modules if k == "flybrain" or k.startswith("flybrain.")]:
                del sys.modules[m]
            from flybrain import columns
            return columns
    raise ImportError("needs fly.ai's flybrain/columns.py (set FLYBRAIN_COLUMNS_SRC to a checkout that has it)")


def build_map() -> dict:
    """front/up (0..1 per eye) and eye of every L2/L3 cell; cached as <fly data>/lamina.npz."""
    data = fly_data()
    cache = data / "lamina.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    columns = _columns_module()
    from flybrain.build import optic_columns
    meta = np.load(data / "brain.npz")
    W = sparse.load_npz(data / "weights.npz").tocsr()
    ct, ids = meta["cell_type"].astype(str), meta["ids"]
    xy, eye = columns._place(W, ct, ids, optic_columns(data / "raw" / "optic-columns.xlsx"))
    t4 = np.flatnonzero(np.char.startswith(ct, "T4") & (eye != ""))
    own = dict(zip(t4, columns._offsets(W, ct, xy, t4)))
    cells = np.flatnonzero(np.isin(ct, LAMINA) & (eye != ""))
    front_up = np.zeros((len(cells), 2), np.float32)
    for side in "LR":                                     # the eye's axes, as columns.build_eye_map
        mean = {d: np.nanmean([own[i] for i in t4 if ct[i] == f"T4{d}" and eye[i] == side], axis=0) for d in "abcd"}
        to_fu = np.linalg.inv(np.column_stack([-(mean["a"] - mean["b"]) / 2, (mean["c"] - mean["d"]) / 2]))
        m = eye[cells] == side
        front_up[m] = (to_fu @ xy[cells[m]].T).T
        lo, hi = front_up[m].min(0), front_up[m].max(0)
        front_up[m] = (front_up[m] - lo) / (hi - lo)
    out = {"cells": cells.astype(np.int32), "eye": eye[cells], "front": front_up[:, 0], "up": front_up[:, 1],
           "types": ct[cells]}
    np.savez(cache, **out)
    return out


class LaminaDrive:
    """Light decrements at each L2/L3 cell's place in the eye -> voltage into it."""

    def __init__(self, lam: dict):
        self.cells = lam["cells"].astype(np.int64)
        left = lam["eye"] == "L"
        x = np.where(left, 0.5 - 0.5 * lam["front"], 0.5 + 0.5 * lam["front"])     # front of the eye = frame centre
        self.px = np.clip(np.round(x * (FLOW_W - 1)), 0, FLOW_W - 1).astype(int)
        self.py = np.clip(np.round((1 - lam["up"]) * (FLOW_H - 1)), 0, FLOW_H - 1).astype(int)
        self.prev = None
        self.last = None

    def see(self, frame: np.ndarray) -> list:
        cur = ndimage.gaussian_filter(frame.astype(np.float32) / 255, 1.5)    # an ommatidium's blur
        inject = []
        if self.prev is not None:
            darker = np.clip(self.prev - cur, 0, None)[self.py, self.px]       # transient: only what changed
            drive = np.minimum(darker * GAIN, CAP)
            self.last = drive
            q = np.round(drive / CAP * 16).astype(int)
            inject = [(self.cells[q == k], k * CAP / 16) for k in np.unique(q) if k > 0]
        self.prev = cur
        return inject
