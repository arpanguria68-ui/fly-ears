"""A fly brain closer to a real one: graded optic lobe, spiking central brain, one connectome.

In a real fly the photoreceptors and most optic-lobe neurons do not spike: they signal with graded
voltage around a resting level (so an inhibitory input can lower their output, which the ON pathway needs),
and cell types differ in speed (direction selectivity comes from fast and slow inputs meeting at T4/T5).
The central brain and the descending and motor neurons spike. This simulator does both on the MaleCNS
wiring, with nothing added that detects anything:

graded neurons (superclass ol_intrinsic and ol_sensory: photoreceptors, lamina, medulla, lobula, T4/T5):
    tau_type * dv/dt = -v + bias + K * sum_j W_ij a_j,   output = clip(v, 0, 1)
    W is the connectome (each neuron's inputs are signed shares that sum to 1); a_j is the output of a
    graded partner, or a spiking partner's rate (low-passed, 50 spikes/s -> 1). bias sets every graded
    neuron's resting output to REST under a plain grey view (homeostatic, label-free: no stimulus or
    result is used). Photoreceptor output follows the light at its eye column (tau 5 ms).
spiking neurons (everything else): fly.ai's leaky integrate-and-fire model, at a 5 ms step; a graded
    partner adds its change from rest at its rate's worth (output 1 = 50 spikes/s): its resting release
    is part of the spiking neuron's own resting level, so the spiking brain rests as in fly.ai's model.

Speeds (time constants), approximations of published measurements (Behnia et al. 2014 Nature 512:427;
Arenz et al. 2017 Curr Biol 27:929): photoreceptors 5 ms; L1-L5 and the fast medulla inputs Mi1, Tm3,
Tm1, Tm2, Tm4 10 ms; the slow ones Mi4, Mi9, Tm9 50 ms; every other graded neuron (incl. T4/T5) 20 ms.

Readout: graded activity is turned into events at its equivalent rate (output 1 = 50 per second) by a
deterministic counter, so rates, rasters and brain maps read both kinds of neuron the same way.
The dynamics never see these events.
"""
from __future__ import annotations

import numpy as np

from . import listen

SUB = 4                     # 5 ms sub-steps per 20 ms step
REST = 0.3                  # graded resting output under a grey view
K = 2.0                     # graded input gain: a full swing of input moves the output across its range
FULL_RATE = 50.0            # spikes/s that a graded output of 1 stands for
RATE_TAU = 0.020            # s: how a graded neuron reads a spiking partner (its spikes, low-passed)
GS = 4.0                    # graded -> spiking strength (calibrate_gs.py; chosen so the giant fiber is ~silent at rest)
TAU = {"photo": 0.005, "fast": 0.010, "slow": 0.050, "other": 0.020}
FAST = ("L1", "L2", "L3", "L4", "L5", "Mi1", "Tm3", "Tm1", "Tm2", "Tm4")
SLOW = ("Mi4", "Mi9", "Tm9")


class HybridBrain:
    """The parts of FlyBrain's interface fly-ears uses, with a graded optic lobe. Batch 1."""

    def __init__(self, device: str = "auto", seed: int = 7, rewired: bool = False):
        FlyBrain = listen.import_flybrain()
        b = FlyBrain(device=device, batch=1, seed=seed, sensory_input=False, dt=0.020 / SUB)
        if rewired:
            listen.rewire(b, seed=1000 + seed)
        self._b = b
        self.device, self.xp = b.device, b.xp
        self.n, self.batch, self.dt = b.n, 1, 0.020
        self.cell_type, self.side, self.superclass = b.cell_type, b.side, b.superclass
        self.positions, self.visual, self.azimuth, self.groups = b.positions, b.visual, b.azimuth, b.groups
        self.sensory_input = False
        self.indices, self.indptr, self.weights = b.indices, b.indptr, b.weights
        xp = self.xp
        sc = np.asarray(b.superclass).astype(str)
        ct = np.asarray(b.cell_type).astype(str)
        graded = np.isin(sc, ["ol_intrinsic", "ol_sensory"])
        tau = np.full(self.n, TAU["other"], np.float32)
        tau[np.isin(ct, FAST)] = TAU["fast"]
        tau[np.isin(ct, SLOW)] = TAU["slow"]
        tau[np.asarray(b.visual)] = TAU["photo"]
        sub_dt = 0.020 / SUB
        self._g = xp.asarray(graded)
        self._alpha = xp.asarray((sub_dt / tau).astype(np.float32))       # Euler step per sub-step
        self._photo = xp.asarray(np.asarray(b.visual))
        self._bias = xp.full(self.n, REST, xp.float32)
        self._rest_out = xp.full(self.n, REST, xp.float32)                  # each graded neuron's own resting output
        self._spike_decay = np.float32(np.exp(-sub_dt / RATE_TAU))
        if b.device == "cuda":
            self._W = b._W
        else:
            from scipy import sparse
            self._W = sparse.csc_matrix((b.weights, b.indices, b.indptr), shape=(b.n, b.n)).tocsr()
        self.calibrated = False
        self.reset(seed)

    # ---------------------------------------------------------------- FlyBrain-like interface
    def cells(self, types, side=None):
        return self._b.cells(types, side)

    def reset(self, seed=None):
        xp = self.xp
        self._b.reset(seed)
        self.v_g = xp.where(self._g, xp.float32(REST), xp.float32(0))       # graded voltages
        self.rate = xp.zeros(self.n, xp.float32)                            # spiking partners' low-passed rate
        self.out = xp.where(self._g, xp.float32(REST), xp.float32(0))
        self.acc = xp.zeros(self.n, xp.float32)                             # readout counters

    def _matvec(self, x):
        return self._W @ x

    def step(self, eye_drive=None, inject=()):
        """One 20 ms step (4 sub-steps). eye_drive: 0..1 per photoreceptor. Returns the indices of
        neurons that spiked or, for graded ones, crossed their readout counter."""
        xp, b = self.xp, self._b
        g = self._g
        photo_target = None
        if eye_drive is not None:
            photo_target = xp.asarray(eye_drive, dtype=xp.float32)
        events = []
        for _ in range(SUB):
            spikes = xp.zeros(self.n, xp.float32)
            spikes[b.fired] = 1.0
            self.rate = self.rate * self._spike_decay + spikes / (FULL_RATE * 0.020 / SUB) * (1 - self._spike_decay)
            act = xp.where(g, self.out, self.rate)                         # what each neuron passes on
            graded_in = self._matvec(act)                                  # signed shares, -1..1
            # spiking neurons: fly.ai's model; a graded partner adds its change from rest, at its rate's worth
            # (its resting release is part of the spiking neuron's own resting level, as in a real fly)
            ev = xp.where(g, (self.out - self._rest_out) * xp.float32(GS * FULL_RATE * 0.020 / SUB), spikes)
            current = self._matvec(ev) * b.gain
            v = b.v[:, 0]
            v *= b.decay
            v += xp.where(g, 0, current + b.tonic)
            v += xp.where(g, 0, (b.rng.random(self.n) < b.noise_hz * b.dt) * np.float32(b.noise_amp))
            before = self.v_g[self._photo].copy()
            self.v_g += self._alpha * (-self.v_g + self._bias + K * graded_in)
            if photo_target is not None:                                   # photoreceptors follow the light
                self.v_g[self._photo] = before + self._alpha[self._photo] * (photo_target - before)
            self.out = xp.where(g, xp.clip(self.v_g, 0, 1), 0)
            fired = xp.flatnonzero((~g) & (v >= 1.0))
            v[fired] = 0.0
            b.fired = fired
            self.acc += xp.where(g, self.out * xp.float32(FULL_RATE * 0.020 / SUB), 0)
            gev = xp.flatnonzero(self.acc >= 1.0)
            self.acc[gev] -= 1.0
            events.append(fired)
            events.append(gev)
        for idx, amount in inject:                                         # receptor input, once per 20 ms step
            b.v[xp.asarray(idx), 0] += xp.float32(amount)
        b.steps += 1
        allev = xp.unique(xp.concatenate(events))
        return allev if xp is np else allev.get()

    # ---------------------------------------------------------------- label-free resting levels
    def calibrate(self, eye_drive, rounds: int = 6, steps: int = 15):
        """Set each graded neuron's bias so it rests at REST under a plain grey view (no stimulus, no
        result used): repeat (settle, measure its input, bias = REST - K * input)."""
        xp = self.xp
        for _ in range(rounds):
            acc = xp.zeros(self.n, xp.float32)
            for _ in range(steps):
                self.step(eye_drive)
                acc += self._matvec(xp.where(self._g, self.out, self.rate))
            mean_in = acc / steps
            self._bias = xp.where(self._g, xp.float32(REST) - K * mean_in, self._bias)
        acc = xp.zeros(self.n, xp.float32)                                  # then record each one's resting output:
        for _ in range(steps):                                              # its change from this is what a spiking
            self.step(eye_drive)                                            # partner receives
            acc += self.out
        self._rest_out = xp.where(self._g, acc / steps, xp.float32(0))
        self.calibrated = True
