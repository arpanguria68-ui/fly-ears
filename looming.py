"""Looming check: with fly-own vision, do LPLC2 and the escape neuron respond more to an approaching disc
than to sliding bars? Test video out/testvideo/test_events.mp4 (bars 0-4 s, approaching disc 4-6 s, still
6-7 s), 5 noise seeds. Nothing of this program detects looming; the light goes only into photoreceptors.

    python looming.py
"""
from pathlib import Path

import numpy as np

import watch
from flyears import vision

P = {"bars": (0.2, 4), "looming": (4.2, 6), "still": (6.2, 7)}
GROUPS = ("LPLC2 (looming)", "escape", "T4/T5 (motion)")


def main() -> None:
    watch.FPS, watch.STEPS_PER_FRAME = 25, 2
    rgb = watch.colour_frames(Path("out/testvideo/test_events.mp4"), 0, 7)
    frames = watch.grey(rgb)
    env = np.zeros((len(frames) * 2, 8), np.float32)
    brain = watch.make_brain("fly-own")
    orig = brain.reset
    res = []
    for seed in range(1, 6):
        brain.reset = lambda s=None, _o=orig, _seed=seed: _o(_seed)
        names, rates, rest, *_ = watch.simulate(brain, vision.Eyes(brain, fps=25, mode="fly-own"), frames, env, False,
                                                rgb=rgb)
        t = np.arange(len(rates)) * 0.02
        row = {n: {k: rates[(t >= a) & (t < b), names.index(n)].mean() for k, (a, b) in P.items()}
               | {"rest": rest[names.index(n)]} for n in GROUPS}
        res.append(row)
        print(f"seed {seed}: " + " | ".join(f"{n.split()[0]} rest {r['rest']:.2f} bars {r['bars']:.2f} "
                                            f"loom {r['looming']:.2f} still {r['still']:.2f}" for n, r in row.items()))
    for n in GROUPS[:2]:
        print(f"{n}: looming > bars in {sum(r[n]['looming'] > r[n]['bars'] for r in res)} of 5; "
              f"mean loom {np.mean([r[n]['looming'] for r in res]):.2f} bars {np.mean([r[n]['bars'] for r in res]):.2f} "
              f"rest {np.mean([r[n]['rest'] for r in res]):.2f} still {np.mean([r[n]['still'] for r in res]):.2f}")


if __name__ == "__main__":
    main()
