"""Physiology check: does the fly's own wiring see motion like a real fly?

    python physio.py              # graded optic lobe (hybrid.py), photoreceptors only, nothing of ours

Real T4/T5 cells (the fly's motion detectors) are tested with moving gratings and edges. Known results
in flies: in each eye, subtype a prefers front-to-back motion, b back-to-front, c upward, d downward
(Maisak et al. 2013 Nature 500:212); T4 responds to bright (ON) edges, T5 to dark (OFF) edges.
Here the stimuli go only into the photoreceptors, at their real eye columns. Direction selectivity
index per subtype: DSI = (preferred - opposite) / (preferred + opposite), responses above rest.
"""
from __future__ import annotations

import sys

import numpy as np

from flyears import lamina
from flyears.hybrid import HybridBrain

W, H = 160, 90
DIRS = {"right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1)}       # in the frame (y down)


SPEED = 24.0                       # px/s (period 24 px: 1 Hz)


def grating(direction, t, period=24, speed=None):
    """Square-wave grating, period in px, speed px/s (about 1 Hz), 0..1."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = DIRS[direction]
    phase = (xx * dx + yy * dy - (speed or SPEED) * t) / period
    return (np.floor(phase * 2) % 2).astype(np.float32) * 0.8 + 0.1


def edge(direction, t, bright, speed=40.0):
    """A bright edge (ON) or a dark edge (OFF) sweeping across, starting at the far side."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = DIRS[direction]
    pos = (xx - (W if dx < 0 else 0)) * dx if dx else (yy - (H if dy < 0 else 0)) * dy
    covered = pos < speed * t
    return np.where(covered, 0.9 if bright else 0.1, 0.1 if bright else 0.9).astype(np.float32)


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "auto"
    brain = HybridBrain(device=device)
    xp = brain.xp
    pos = lamina.build_eye_positions()
    ct = np.asarray(brain.cell_type).astype(str)
    # photoreceptors at their eye columns (front of each eye = frame centre)
    where = {int(c): k for k, c in enumerate(pos["cells"])}
    vis = np.asarray(brain.visual)
    k = np.array([where.get(int(c), -1) for c in vis])
    ok = k >= 0
    left = pos["eye"][k[ok]] == "L"
    x = np.where(left, 0.5 * pos["front"][k[ok]], 1 - 0.5 * pos["front"][k[ok]])      # front of each eye = centre
    px = np.full(len(vis), W // 2)
    py = np.full(len(vis), H // 2)
    px[ok] = np.clip(np.round(x * (W - 1)), 0, W - 1)
    py[ok] = np.clip(np.round((1 - pos["up"][k[ok]]) * (H - 1)), 0, H - 1)
    print(f"photoreceptors placed on the eye: {ok.sum()} of {len(vis)}")
    drive = lambda img: img[py, px]                                          # noqa: E731
    grey = np.full((H, W), 0.5, np.float32)
    brain.calibrate(drive(grey))
    print(f"calibrated: graded neurons rest at {float(brain.out[brain._g].mean()):.3f} (target 0.3)")
    groups = {}
    for e in "LR":
        for kind in ("T4", "T5"):
            for s in "abcd":
                m = (pos["types"] == f"{kind}{s}") & (pos["eye"] == e)
                groups[f"{kind}{s} {e}"] = xp.asarray(pos["cells"][m])
    def run(make, seconds=2.0, measure="amplitude"):
        """Adapt to the stimulus's first frame (1 s), then present it moving. amplitude: each cell's
        temporal standard deviation (how strongly it follows the passing pattern), averaged over the
        cells of a group; mean: the change of the mean output from the adapted level."""
        first = drive(make(0.0))
        for _ in range(50):
            brain.step(first)
        rest = {g: float(brain.out[i].mean()) for g, i in groups.items()}
        s1 = {g: xp.zeros(len(i), xp.float32) for g, i in groups.items()}
        s2 = {g: xp.zeros(len(i), xp.float32) for g, i in groups.items()}
        base = {g: brain.out[i].copy() for g, i in groups.items()}           # each cell's adapted level
        rect = {g: xp.zeros(len(i), xp.float32) for g, i in groups.items()}
        n = int(seconds / 0.02)
        for s in range(n):
            brain.step(drive(make(s * 0.02)))
            if s >= 10:
                for g, i in groups.items():
                    o = brain.out[i]
                    s1[g] += o
                    s2[g] += o * o
                    rect[g] += xp.maximum(o - base[g], 0)
        m = n - 10
        if measure == "rectified":                                           # depolarisation, as calcium imaging sees it
            return {g: float((rect[g] / m).mean()) for g in groups}
        if measure == "mean":
            return {g: float((s1[g] / m).mean()) - rest[g] for g in groups}
        return {g: float(xp.sqrt(xp.maximum(s2[g] / m - (s1[g] / m) ** 2, 0)).mean()) for g in groups}
    resp = {d: run(lambda t, d=d: grating(d, t)) for d in DIRS}
    rect = {d: run(lambda t, d=d: grating(d, t), measure="rectified") for d in DIRS}
    # preferred direction of each subtype in each eye, in frame terms (front of the eye = frame centre)
    pref = {("a", "L"): "left", ("b", "L"): "right", ("a", "R"): "right", ("b", "R"): "left",
            ("c", "L"): "up", ("c", "R"): "up", ("d", "L"): "down", ("d", "R"): "down"}
    opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}
    print("\ngratings (response above rest, graded output):")
    print(f"  {'cell':7s} {'right':>8s} {'left':>8s} {'up':>8s} {'down':>8s}   expected best   DSI")
    good = 0
    rows = []
    for g in groups:
        kind_s, e = g.split()
        p = pref[(kind_s[2], e)]
        r = {d: resp[d][g] for d in DIRS}
        a, b = r[p], r[opposite[p]]
        dsi = (a - b) / (a + b) if a + b > 1e-6 else 0.0
        best = max(r, key=r.get)
        good += best == p
        rows.append(dsi)
        print(f"  {g:7s} " + " ".join(f"{r[d]:8.4f}" for d in DIRS) + f"   {p:6s} {'ok' if best == p else '--'}   {dsi:+.2f}")
    print(f"\n  best direction = the real fly's for {good} of {len(groups)} subtype x eye; mean DSI {np.mean(rows):+.2f}")
    rd = []
    for g in groups:                                                         # the same, on depolarisation only
        kind_s, e = g.split()
        p = pref[(kind_s[2], e)]
        a, b = rect[p][g], rect[opposite[p]][g]
        rd.append((a - b) / (a + b) if a + b > 1e-9 else 0.0)
    print(f"  on depolarisation (as calcium imaging measures it): mean DSI {np.mean(rd):+.2f}, "
          f"positive in {sum(x > 0 for x in rd)} of {len(rd)}")
    on = run(lambda t: edge("right", t, True), measure="mean")
    off = run(lambda t: edge("right", t, False), measure="mean")
    t4 = np.mean([on[g] for g in groups if g.startswith("T4")]), np.mean([off[g] for g in groups if g.startswith("T4")])
    t5 = np.mean([on[g] for g in groups if g.startswith("T5")]), np.mean([off[g] for g in groups if g.startswith("T5")])
    print("\nedges moving right (response above rest):        bright (ON)   dark (OFF)")
    print(f"  T4 (should prefer bright)                      {t4[0]:+.4f}      {t4[1]:+.4f}")
    print(f"  T5 (should prefer dark)                        {t5[0]:+.4f}      {t5[1]:+.4f}")


if __name__ == "__main__":
    main()
