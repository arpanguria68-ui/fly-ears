"""More of what a real fruit fly has: colour vision, smell, taste, wind, temperature, humidity, touch.

Colour (from the video): the photoreceptors are split by type, as in a real eye. R1-6 (broadband, motion
and brightness) get the frame's brightness, R8 its green, R7 its blue. A video has no ultraviolet, which
R7 mostly sees, so blue stands in for it (stated, not hidden).

Everything else is given by you, on a schedule ("vinegar:5-15,heat:20-30"), as voltage into the fly's own
receptor neurons for that sense (the cell types are fly.ai's / the literature's, listed below). A stimulus
is a fixed voltage per 20 ms step while it is on; nothing is fitted.
"""
from __future__ import annotations

import re

import numpy as np

STRENGTH = 0.6                      # volts per step into a stimulated receptor neuron (fly.ai's probes: 0.5-0.8)

# name: (cell types, side or None, sense, what it is)
STIMULI = {
    "fruit":     (["ORN_DM1", "ORN_DM2"], None, "smell", "ripe fruit / food esters"),
    "vinegar":   (["ORN_VL2a"], None, "smell", "vinegar (acetic acid)"),
    "mould":     (["ORN_DA2"], None, "smell", "geosmin: mould, a danger smell flies avoid"),
    "CO2":       (["ORN_V"], None, "smell", "carbon dioxide: a stress odour"),
    "pheromone": (["ORN_DA1", "ORN_VA1d"], None, "smell", "cVA, the male sex pheromone"),
    "sugar":     (["LB3", "LB2d"], None, "taste", "sweet taste neurons"),
    "bitter":    (["LB1a,LB1d", "LB1b", "LB1c", "LB1e"], None, "taste", "bitter taste neurons"),
    "wind":      (["JO-CM", "JO-EV1", "JO-EV2", "JO-EV3", "JO-EV5"], None, "wind", "wind on the antennae"),
    "heat":      (["TRN_VP2"], None, "temperature", "hot cells of the antenna"),
    "cold":      (["TRN_VP3a", "TRN_VP3b"], None, "temperature", "cold cells of the antenna"),
    "dry air":   (["HRN_VP4"], None, "humidity", "dry-air cells"),
    "moist air": (["HRN_VP5"], None, "humidity", "moist-air cells"),
    "touch L":   (["SNta*"], "L", "touch", "bristles on the left of the body"),
    "touch R":   (["SNta*"], "R", "touch", "bristles on the right of the body"),
    "head touch": (["BM_InOm"], None, "touch", "bristles on the head"),
}


def cells_for(brain, name: str) -> np.ndarray:
    types, side, *_ = STIMULI[name]
    ct = np.asarray(brain.cell_type).astype(str)
    mask = np.zeros(len(ct), bool)
    for t in types:
        mask |= np.char.startswith(ct, t[:-1]) if t.endswith("*") else (ct == t)
    if side:
        mask &= np.asarray(brain.side).astype(str) == side
    return np.flatnonzero(mask)


def parse_schedule(text: str) -> list[tuple[str, float, float]]:
    """'vinegar:5-15, heat:20-30' -> [("vinegar", 5, 15), ("heat", 20, 30)] (seconds into the clip)."""
    out = []
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        m = re.fullmatch(r"(.+?)\s*:\s*([\d.]+)\s*-\s*([\d.]+)", part)
        if not m or m.group(1).strip() not in STIMULI:
            raise ValueError(f"not a stimulus: {part!r} (use name:from-to, names: {', '.join(STIMULI)})")
        a, b = float(m.group(2)), float(m.group(3))
        if b <= a:
            raise ValueError(f"{part!r}: the end must be after the start")
        out.append((m.group(1).strip(), a, b))
    return out


class Stimuli:
    """The scheduled stimuli as injections at each moment."""

    def __init__(self, brain, schedule: list[tuple[str, float, float]]):
        self.schedule = schedule
        self.cells = {name: cells_for(brain, name) for name in {s[0] for s in schedule}}

    def at(self, t: float) -> list:
        return [(self.cells[name], STRENGTH) for name, a, b in self.schedule if a <= t < b and len(self.cells[name])]

    def on(self, t: float) -> list[str]:
        return [name for name, a, b in self.schedule if a <= t < b]


# ---------------------------------------------------------------- colour vision
class ColourEyes:
    """Per-photoreceptor drive from a colour frame: R1-6 brightness, R8 green, R7 blue (for UV)."""

    def __init__(self, brain, width: int):
        visual = np.asarray(brain.visual)
        ct = np.asarray(brain.cell_type).astype(str)[visual]
        self.kind = np.where(ct == "R7", 2, np.where(ct == "R8", 1, 0))   # 0 brightness, 1 green, 2 blue
        az = np.asarray(brain.azimuth, np.float32)
        self.x = np.clip(np.round((0.5 + 0.5 * az) * (width - 1)), 0, width - 1).astype(int)

    def drive(self, rgb: np.ndarray) -> np.ndarray:
        """rgb (H, W, 3) uint8 -> 0..1 per photoreceptor (the mean of its frame column)."""
        col = rgb.mean(0) / 255                                  # (W, 3)
        lum = col.mean(1)
        table = np.stack([lum, col[:, 1], col[:, 2]], 1)         # brightness, green, blue
        return table[self.x, self.kind].astype(np.float32)


# ---------------------------------------------------------------- what the fly does with them (readouts)
def _types(ct: np.ndarray, test) -> np.ndarray:
    keep = {t for t in set(ct.tolist()) if test(t)}
    return np.flatnonzero(np.isin(ct, list(keep)))


def systems(brain) -> dict[str, np.ndarray]:
    """Groups for the senses above and the brain systems that use them (fixed in advance)."""
    ct = np.asarray(brain.cell_type).astype(str)
    g: dict[str, np.ndarray] = {}
    vis = np.asarray(brain.visual)
    vt = ct[vis]
    g["R1-6 (brightness)"], g["R7 (UV/blue)"], g["R8 (blue/green)"] = vis[vt == "R1-6"], vis[vt == "R7"], vis[vt == "R8"]
    for name in STIMULI:
        g[f"sense: {name}"] = cells_for(brain, name)
    g["smell receptors (all ORN)"] = _types(ct, lambda t: t.startswith("ORN_"))
    g["smell projection neurons"] = _types(ct, lambda t: re.search(r"_[a-z0-9]*PN\d*$", t) is not None)
    g["mushroom body (Kenyon cells)"] = _types(ct, lambda t: t.startswith("KC"))
    g["memory output (MBON)"] = _types(ct, lambda t: t.startswith("MBON"))
    g["lateral horn (innate smell)"] = _types(ct, lambda t: re.match(r"LH\d", t) is not None)
    g["compass (EPG)"] = _types(ct, lambda t: t.startswith("EPG"))
    g["compass shift (PEN)"] = _types(ct, lambda t: t.startswith("PEN"))
    g["steering (PFL)"] = _types(ct, lambda t: t.startswith("PFL"))
    g["clock neurons"] = _types(ct, lambda t: t.startswith(("l-LNv", "s-LNv", "LNd", "DN1")))
    g["insulin cells (IPC)"] = _types(ct, lambda t: t == "IPC")
    return {k: v for k, v in g.items() if len(v)}


# Spectral sensitivity at a display's red, green and blue primaries (about 610, 545 and 465 nm), read
# roughly off the Drosophila rhodopsin curves (Salcedo et al. 1999 J Neurosci 19:10716); white = 1.
#   R1-6, Rh1 (peak ~480 nm): nearly red-blind;  R8: Rh6 (~510 nm, the "yellow" R8s, about 70%; the
#   dataset does not say which R8 is pale, Rh5 blue, so all get Rh6);  R7: Rh3/Rh4 are UV (345/375 nm),
#   which a screen does not emit; only the blue primary reaches their long-wavelength tail.
SPECTRAL = np.array([[0.03, 0.34, 0.63],      # R1-6
                     [0.06, 0.55, 0.39],      # R8
                     [0.00, 0.00, 1.00]],     # R7
                    np.float32)


def srgb_to_light(rgb: np.ndarray) -> np.ndarray:
    """8-bit sRGB (gamma-encoded) -> the screen's physical light, 0..1 (IEC 61966-2-1)."""
    c = rgb.astype(np.float32) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


class ColumnEyes:
    """Light from a colour frame into each photoreceptor at its own eye column (2-D), as a fly's eye takes it:

    * optics: each ommatidium sees a blurred patch (Gaussian acceptance angle, full width at half maximum
      = the spacing between neighbouring columns, as in Drosophila, ~5 deg each), not one pixel;
    * light: the frame's sRGB values are turned back into the screen's physical light;
    * colour: each type weighs red, green and blue by its rhodopsin (SPECTRAL).

    Columns from lamina.build_eye_positions(). The few photoreceptors without a column get their eye's mean
    light (no made-up position)."""

    def __init__(self, brain, width: int, height: int):
        from scipy.spatial import cKDTree

        from .lamina import build_eye_positions
        pos = build_eye_positions()
        where = {int(c): k for k, c in enumerate(pos["cells"])}
        visual = np.asarray(brain.visual)
        ct = np.asarray(brain.cell_type).astype(str)[visual]
        self.kind = np.where(ct == "R7", 2, np.where(ct == "R8", 1, 0))
        k = np.array([where.get(int(c), -1) for c in visual])
        ok = k >= 0
        x = np.zeros(len(visual), np.float32)
        y = np.zeros(len(visual), np.float32)
        left = pos["eye"][k[ok]] == "L"
        x[ok] = np.where(left, 0.5 * pos["front"][k[ok]], 1 - 0.5 * pos["front"][k[ok]])   # front = centre
        y[ok] = 1 - pos["up"][k[ok]]
        self.x = np.clip(np.round(x * (width - 1)), 0, width - 1).astype(int)
        self.y = np.clip(np.round(y * (height - 1)), 0, height - 1).astype(int)
        self.ok = ok
        self.left = np.where(ok, False, np.asarray(brain.azimuth)[: len(visual)] < 0)  # unplaced: which eye
        self.left[ok] = left
        self.placed = int(ok.sum())
        # column spacing in pixels (one R8 per column) -> acceptance angle
        gaps = []
        for e in "LR":
            m = (pos["eye"] == e) & (pos["types"] == "R8")
            p = np.c_[(0.5 * pos["front"][m] if e == "L" else 1 - 0.5 * pos["front"][m]) * (width - 1),
                      (1 - pos["up"][m]) * (height - 1)]
            gaps.append(cKDTree(p).query(p, 2)[0][:, 1])
        self.spacing = float(np.median(np.concatenate(gaps)))
        self.sigma = self.spacing / 2.355                                  # FWHM = spacing

    def drive(self, rgb: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        light = srgb_to_light(rgb)
        seen = gaussian_filter(light, (self.sigma, self.sigma, 0), mode="nearest")      # each ommatidium's patch
        px = seen[self.y, self.x]                                          # (n, 3)
        d = np.einsum("nc,nc->n", px, SPECTRAL[self.kind]).astype(np.float32)
        if not self.ok.all():
            half = rgb.shape[1] // 2
            mean = [np.einsum("c,c->", light[:, s].reshape(-1, 3).mean(0), SPECTRAL[0]) for s in
                    (slice(0, half), slice(half, None))]
            d[~self.ok] = np.where(self.left[~self.ok], mean[0], mean[1])
        return d
