"""Pick how the graded optic lobe talks to the spiking brain, on gratings and rest only.

    python calibrate_gs.py

Two physiological constraints, fixed in advance (the looming / escape test never takes part):
  1. the lobula visual projection neurons (LPLC2, LC4, LPLC1, LC10a) fire 5-15 spikes/s during gratings;
  2. the giant fiber (DNp01, the escape neuron) is silent at rest (< 0.5 spikes/s), as in real flies.
Searched: hybrid.GS (graded -> spiking strength) and hybrid.RATE_TAU (how smoothly a graded neuron reads a
spiking partner's spikes). Picked: the first setting meeting both, smallest GS first.
"""
import numpy as np

import physio
from flyears import hybrid
from flyears.hybrid import HybridBrain
from flyears.senses import ColumnEyes

to_rgb = lambda img: (np.repeat(img[..., None], 3, 2) * 255).astype(np.uint8)   # noqa: E731


def measure(gs, tau):
    hybrid.GS, hybrid.RATE_TAU = gs, tau
    brain = HybridBrain()
    eyes = ColumnEyes(brain, physio.W, physio.H)
    grey = eyes.drive(to_rgb(np.full((physio.H, physio.W), 0.5, np.float32)))
    brain.calibrate(grey)
    lc, gf = brain.cells(["LPLC2", "LC4", "LPLC1", "LC10a"]), brain.cells(["DNp01"])
    brain.reset(1)
    for _ in range(100):
        brain.step(grey)
    rest = sum(np.isin(gf, brain.step(grey)).sum() for _ in range(150)) / (len(gf) * 150 * 0.02)
    n = 0
    for d in physio.DIRS:
        for s in range(75):
            n += np.isin(lc, brain.step(eyes.drive(to_rgb(physio.grating(d, s * 0.02))))).sum()
    return n / (len(lc) * 4 * 75 * 0.02), rest


def main():
    pick = None
    for tau in (0.02, 0.1, 0.3):
        for gs in (4, 8, 16):
            vp, gfr = measure(gs, tau)
            ok = 5 <= vp <= 15 and gfr < 0.5
            print(f"  rate tau {tau * 1000:4.0f} ms, strength {gs:2d}: visual projection {vp:6.2f}/s during gratings,"
                  f" giant fiber at rest {gfr:5.2f}/s  {'OK' if ok else ''}", flush=True)
            if ok and pick is None:
                pick = (gs, tau)
    print("pick:", pick)


if __name__ == "__main__":
    main()
