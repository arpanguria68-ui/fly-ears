"""Play clips to the connectome and record how it responds.

Each clip is one fly in a batch (same wiring, own voltages and noise). The brain gets 0.5 s of
silence to settle, then the clip through its ears. Recorded per clip: spike rates of a fixed set
of neuron groups, chosen before any song was heard (the ear, the auditory relay, every
superclass), summarised the same way as the ear's own input (see features.response_stats).

Brains: the real MaleCNS connectome, or a degree-preserving rewired copy (each synapse keeps its
sender and weight, each neuron keeps how many synapses it receives, partners are shuffled).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from . import ear

SETTLE = 25                     # steps of silence before the clip (0.5 s)


def import_flybrain():
    """flybrain comes from the fly.ai code (FLYBRAIN_SRC, else ../firefly)."""
    try:
        import flybrain  # noqa: F401
    except ImportError:
        src = Path(os.environ.get("FLYBRAIN_SRC") or Path(__file__).resolve().parents[2] / "firefly")
        sys.path.insert(0, str(src))
    from flybrain import FlyBrain
    return FlyBrain


def make_brain(batch: int, device: str = "auto", seed: int = 7, rewired: bool = False):
    FlyBrain = import_flybrain()
    brain = FlyBrain(device=device, batch=batch, seed=seed, sensory_input=False)
    if rewired:
        rewire(brain, seed=1000 + seed)
    return brain


def rewire(brain, seed: int) -> None:
    """Degree-preserving scramble (as fly.ai's flytalk.py)."""
    rng = np.random.default_rng(seed)
    brain.indices = rng.permutation(brain.indices).astype(brain.indices.dtype)
    if not brain.sensory_input:
        sensory = np.char.find(brain.superclass.astype(str), "sensory") >= 0
        brain.weights = np.where(sensory[brain.indices], 0, brain.weights).astype(brain.weights.dtype)
    if brain.device == "cuda":
        from cupyx.scipy import sparse as cusparse
        W = sparse.csc_matrix((brain.weights, brain.indices, brain.indptr), shape=(brain.n, brain.n))
        W.sum_duplicates()
        brain._W = cusparse.csr_matrix(W.tocsr().astype(np.float32))


def groups(brain) -> dict[str, np.ndarray]:
    """The recorded neuron groups, fixed in advance: the ear, the auditory relay, every superclass."""
    ct = brain.cell_type.astype(str)
    g = {
        "ear JO-A/B": brain.cells(sorted({t for t in ct if t.startswith(("JO-A", "JO-B"))})),
        "other JO": brain.cells(sorted({t for t in ct if t.startswith("JO-") and not t.startswith(("JO-A", "JO-B"))})),
        "AMMC": brain.cells(sorted({t for t in ct if t.startswith("AMMC")})),
        "WED": brain.cells(sorted({t for t in ct if t.startswith("WED") and not t.startswith("WEDPN")})),
        "WEDPN": brain.cells(sorted({t for t in ct if t.startswith("WEDPN")})),
    }
    for sc in sorted(set(brain.superclass.astype(str))):
        if sc and sc != "nan":
            g[sc] = brain.cells([sc])
    return {k: v for k, v in g.items() if len(v)}


def play(brain, envs: list[np.ndarray], mode: str = "tonotopic", seed: int = 0, log=print) -> tuple[np.ndarray, list[str]]:
    """Spike counts per group per step for each clip: (clips, steps, groups)."""
    cells, bands = ear.ear_map(brain, mode)
    G = groups(brain)
    names = list(G)
    aud = np.full(brain.n, -1)                            # auditory groups (they overlap superclasses,
    sup = np.full(brain.n, -1)                            # so each neuron has one code in each)
    for j, k in enumerate(names):
        (aud if j < 5 else sup)[G[k]] = j
    K = len(names)
    B = brain.batch
    steps = min(len(e) for e in envs)
    out = np.zeros((len(envs), steps, len(names)), np.float32)
    t0 = time.perf_counter()
    for start in range(0, len(envs), B):
        batch = envs[start:start + B]
        env = np.zeros((steps, ear.N_BANDS, B), np.float32)
        for b, e in enumerate(batch):
            env[:, :, b] = e[:steps]
        brain.reset(seed * 100_000 + start)
        for _ in range(SETTLE):
            brain.step()
        for s in range(steps):
            fired = brain.step(inject=ear.injections(cells, bands, env[s]))
            fired = [fired] if B == 1 else fired
            for b in range(len(batch)):
                a, c = aud[fired[b]], sup[fired[b]]
                out[start + b, s] = np.bincount(a[a >= 0], minlength=K) + np.bincount(c[c >= 0], minlength=K)
        log(f"  played {min(start + B, len(envs))}/{len(envs)} clips "
            f"({(time.perf_counter() - t0) / min(start + B, len(envs)):.1f} s/clip)")
    sizes = np.array([len(G[k]) for k in names], np.float32)
    return out / sizes / brain.dt, names                  # spikes per neuron per second
