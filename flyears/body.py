"""The fly's body and internal states, read from the connectome's own neurons.

Body: the MaleCNS includes the ventral nerve cord (the fly's spinal cord) and its motor neurons.
  wings     flight power muscles (DLM, DVM) and steering muscles (b1-3, hg1-4, i1-2, iii, tp, ps, hDVM)
  legs      leg motor neurons, named by muscle (tibia, femur, trochanter, tarsus, sternal, pleural...),
            assigned to front / middle / hind leg by their position along the nerve cord (three clear
            clusters: the T1, T2, T3 segments; the one nearest the brain is the front leg) and to a side
  jump      TTMn (tergotrochanteral muscle: the escape jump)
  neck, abdomen
Behaviour: command (descending) neurons with known roles: walking forward (DNg100), turning (DNa02,
  per side), escape (DNp01, the giant fiber), backing up (MDN), grooming (DNg11), courtship song (pIP10).
States ("emotions"): populations that carry a fly's internal states. They are readouts of this
  model's spiking, not measured feelings, and the words are shorthand:
  fear        giant-fiber escape circuit (DNp01, GFC1-4)
  pleasure    PAM dopamine neurons (reward)
  distress    PPL1 dopamine neurons (punishment)
  excitement  octopamine neurons (arousal)
  desire      pC1 neurons (courtship drive)
  anger       aIPg neurons (aggression)
  mood        serotonin neurons
  appetite    NPF neurons (hunger)
"""
from __future__ import annotations

import numpy as np

LEG_MUSCLES = ("Ti ", "Tr ", "Fe ", "Ta ", "Sternal", "Pleural", "Acc.", "Sternotrochanter", "Tergopleural", "ltm")
WING_POWER = ("DLMn", "DVMn")
WING_STEER = ("b1 ", "b2 ", "b3 ", "hg", "i1 ", "i2 ", "iii", "hi", "tp", "ps", "hDVM")
NECK = ("MNnm", "CEM", "ADNM", "FNM")

BEHAVIOUR = {"walk": ["DNg100"], "back up": ["MDN"], "escape": ["DNp01"], "groom": ["DNg11"], "song": ["pIP10"]}
STATES = {
    "fear": ("DNp01", "GFC"), "pleasure": ("PAM",), "distress": ("PPL1",), "excitement": ("OA-",),
    "desire": ("pC1",), "anger": ("aIPg",), "mood": ("5-HT",), "appetite": ("NPF",),
}
STATE_SOURCE = {"fear": "giant-fiber escape circuit", "pleasure": "PAM dopamine (reward)",
                "distress": "PPL1 dopamine (punishment)", "excitement": "octopamine (arousal)",
                "desire": "pC1 (courtship drive)", "anger": "aIPg (aggression)", "mood": "serotonin",
                "appetite": "NPF (hunger)"}


def _by_prefix(ct: np.ndarray, prefixes) -> np.ndarray:
    pre = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
    return np.flatnonzero([t.startswith(pre) for t in ct])


def leg_segments(brain, legs: np.ndarray) -> dict[int, str]:
    """front / middle / hind for each leg motor neuron, from three clusters along the nerve cord."""
    pos = np.asarray(brain.positions, np.float64)
    p = pos[legs]
    ax = int(np.argmax(np.nanstd(p, 0)))
    v = p[:, ax]
    s = np.sort(v[np.isfinite(v)])
    gaps = np.argsort(np.diff(s))[-2:]                    # the two biggest gaps split T1 / T2 / T3
    cuts = np.sort(s[gaps] + np.diff(s)[gaps] / 2)
    seg = np.digitize(v, cuts)                            # 0, 1, 2 along the axis
    brain_c = np.nanmean(pos[np.asarray(brain.superclass).astype(str) == "cb_intrinsic"][:, ax])
    near_brain_first = abs(np.nanmean(v[seg == 0]) - brain_c) < abs(np.nanmean(v[seg == 2]) - brain_c)
    names = ("front", "middle", "hind") if near_brain_first else ("hind", "middle", "front")
    return {int(i): names[k] for i, k in zip(legs, seg)}


def groups(brain) -> dict[str, np.ndarray]:
    """Body parts, behaviour commands and states: name -> neuron indices (fixed in advance)."""
    ct = np.asarray(brain.cell_type).astype(str)
    side = np.asarray(brain.side).astype(str)
    g: dict[str, np.ndarray] = {}
    for s in "LR":
        g[f"wing power {s}"] = np.intersect1d(_by_prefix(ct, WING_POWER), np.flatnonzero(side == s))
        g[f"wing steer {s}"] = np.intersect1d(_by_prefix(ct, WING_STEER), np.flatnonzero(side == s))
        g[f"neck {s}"] = np.intersect1d(_by_prefix(ct, NECK), np.flatnonzero(side == s))
    legs = _by_prefix(ct, LEG_MUSCLES)
    if len(legs) and getattr(brain, "positions", None) is not None:
        seg = leg_segments(brain, legs)
        for s in "LR":
            for part in ("front", "middle", "hind"):
                g[f"leg {part} {s}"] = np.array([i for i in legs if seg[int(i)] == part and side[i] == s], int)
    g["jump"] = _by_prefix(ct, "TTMn")
    g["abdomen"] = _by_prefix(ct, "MNad")
    for name, types in BEHAVIOUR.items():
        g[name] = brain.cells(types)
    for s in "LR":
        g[f"turn {s}"] = brain.cells(["DNa02"], side=s)
    for name, prefixes in STATES.items():
        g[name] = _by_prefix(ct, prefixes)
    return {k: v for k, v in g.items() if len(v)}
