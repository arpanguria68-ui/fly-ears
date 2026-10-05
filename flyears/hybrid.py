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
Tm1, Tm2, Tm4 10 ms; the slow ones Mi4, Mi9, Tm9 150 ms (ds_timing.py); every other graded neuron (incl. T4/T5) 20 ms.

Photoreceptors adapt, as real ones do: each signals its light relative to its own recent mean (time
constant ADAPT_TAU), around the resting level, so the optic lobe responds to contrast and change, not to
how bright a scene is overall.

Readout: a graded cell is turned into events while it is depolarised above its own resting output (0.1
above rest = 5 per second), by a deterministic counter, so rates, rasters and maps show responses, not
the steady resting level (hyperpolarisation, e.g. the lamina's answer to light, is not shown as events).
The dynamics never see these events.
"""
from __future__ import annotations

import numpy as np

from . import listen

SUB = 4                     # 5 ms sub-steps per 20 ms step
REST = 0.3                  # graded resting output under a grey view
K = 2.0                     # graded input gain: a full swing of input moves the output across its range
FULL_RATE = 50.0            # spikes/s that a graded output of 1 stands for
ADAPT_TAU = 1.0             # s: photoreceptors adapt to the recent mean light (real ones signal contrast)
ADAPT_GAIN = 1.0            # contrast -> output around the resting level (0.5 at the adapted mean)
SHUNT = 0.0                 # shunting (divisive) inhibition: inhibitory input also divides a graded
                            # cell's response, as conductance-based synapses do (set by tune_shunt.py)
RATE_TAU = 0.020            # s: how a graded neuron reads a spiking partner (its spikes, low-passed)
GS = 4.0                    # graded -> spiking strength (calibrate_gs.py; chosen so the giant fiber is ~silent at rest)
TAU = {"photo": 0.005, "fast": 0.010, "slow": 0.150, "other": 0.020}   # slow 150 ms: see ds_timing.py
FAST = ("L1", "L2", "L3", "L4", "L5", "Mi1", "Tm3", "Tm1", "Tm2", "Tm4")
SLOW = ("Mi4", "Mi9", "Tm9")


_CSR_SRC = r'''
extern "C" __global__ void csr_matvec(const int n, const int* indptr, const int* indices, const float* data,
                                      const float* x, float* y) {
    // one warp (32 threads) per row; each lane sums a fixed stride, then a fixed tree: same order every run
    int row = (blockDim.x * blockIdx.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    float s = 0.0f;
    for (int k = indptr[row] + lane; k < indptr[row + 1]; k += 32) s += data[k] * x[indices[k]];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
    if (lane == 0) y[row] = s;
}'''


class _DeterministicCSR:
    """y = W @ x on the GPU with each row summed in its stored order: the same result every run."""

    def __init__(self, xp, m):
        self.xp, self.n = xp, m.shape[0]
        self.indptr = xp.asarray(m.indptr.astype(np.int32))
        self.indices = xp.asarray(m.indices.astype(np.int32))
        self.data = xp.asarray(m.data.astype(np.float32))
        self.kernel = xp.RawKernel(_CSR_SRC, "csr_matvec")

    def __matmul__(self, x):
        xp = self.xp
        x = xp.ascontiguousarray(x, dtype=xp.float32)
        y = xp.empty(self.n, xp.float32)
        self.kernel(((self.n * 32 + 255) // 256,), (256,),(np.int32(self.n), self.indptr, self.indices, self.data, x, y))
        return y


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
        self._alpha = xp.asarray((sub_dt / tau).astype(np.float32))       # photoreceptor relaxation per sub-step
        self._rate = self._alpha                                           # dt / tau, for the exact graded step
        self._photo = xp.asarray(np.asarray(b.visual))
        self._bias = xp.full(self.n, REST, xp.float32)
        self._rest_out = xp.full(self.n, REST, xp.float32)                  # each graded neuron's own resting output
        self._spike_decay = np.float32(np.exp(-sub_dt / RATE_TAU))
        if b.device == "cuda":
            # Own CSR product, one thread per neuron summing its inputs in a fixed order, so a run can be
            # repeated bit for bit (the library product sums in a varying order, and the spiking network turns
            # that rounding into different spikes).
            from scipy import sparse as _sp
            Wn = _sp.csc_matrix((b.weights, b.indices, b.indptr), shape=(b.n, b.n)).tocsr()
            Wn.sort_indices()
            self._W = _DeterministicCSR(xp, Wn.astype(np.float32))
            self._Winh = _DeterministicCSR(xp, abs(Wn.minimum(0)).tocsr().astype(np.float32))  # inhibitory part only
        else:
            from scipy import sparse
            self._W = sparse.csc_matrix((b.weights, b.indices, b.indptr), shape=(b.n, b.n)).tocsr()
            self._Winh = abs(self._W.minimum(0)).tocsr().astype(np.float32)
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
        self._adapt = None                                                  # photoreceptors' adapted light level

    def _matvec(self, x):
        return self._W @ x

    def step(self, eye_drive=None, inject=()):
        """One 20 ms step (4 sub-steps). eye_drive: 0..1 per photoreceptor. Returns the indices of
        neurons that spiked or, for graded ones, crossed their readout counter."""
        xp, b = self.xp, self._b
        g = self._g
        photo_target = None
        if eye_drive is not None:                                          # light -> contrast against the
            light = xp.asarray(eye_drive, dtype=xp.float32)               # photoreceptor's adapted level
            if self._adapt is None:
                self._adapt = light.copy()
            self._adapt += (light - self._adapt) * xp.float32(1 - np.exp(-0.020 / ADAPT_TAU))
            photo_target = xp.clip(0.5 + ADAPT_GAIN * (light - self._adapt), 0, 1)
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
            # exact exponential step towards the steady state (stable for any input): with shunting
            # (conductance) inhibition the steady state is (bias + K * input) / (1 + SHUNT * inhibition)
            # and the cell relaxes faster by the same factor
            leak = 1 + SHUNT * (self._Winh @ act) if SHUNT else 1
            target = (self._bias + K * graded_in) / leak
            self.v_g = target + (self.v_g - target) * xp.exp(-self._rate * leak)
            if photo_target is not None:                                   # photoreceptors follow the light
                self.v_g[self._photo] = before + self._alpha[self._photo] * (photo_target - before)
            self.out = xp.where(g, xp.clip(self.v_g, 0, 1), 0)
            fired = xp.flatnonzero((~g) & (v >= 1.0))
            v[fired] = 0.0
            b.fired = fired
            # readout only: a graded cell counts while it is depolarised above its own rest (0.1 above = 5/s)
            self.acc += xp.where(g, xp.maximum(self.out - self._rest_out, 0) * xp.float32(FULL_RATE * 0.020 / SUB), 0)
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
            acc_i = xp.zeros(self.n, xp.float32)
            for _ in range(steps):
                self.step(eye_drive)
                a = xp.where(self._g, self.out, self.rate)
                acc += self._matvec(a)
                if SHUNT:
                    acc_i += self._Winh @ a
            mean_in, mean_inh = acc / steps, acc_i / steps
            # steady state v = (bias + K * input) / (1 + SHUNT * inhibition): solve for v = REST
            self._bias = xp.where(self._g, xp.float32(REST) * (1 + SHUNT * mean_inh) - K * mean_in, self._bias)
        acc = xp.zeros(self.n, xp.float32)                                  # then record each one's resting output:
        for _ in range(steps):                                              # its change from this is what a spiking
            self.step(eye_drive)                                            # partner receives
            acc += self.out
        self._rest_out = xp.where(self._g, acc / steps, xp.float32(0))
        self.calibrated = True

    def save_calibration(self, path) -> None:
        """Each graded neuron's bias and resting output, so a run can be reproduced exactly."""
        get = (lambda a: a) if self.xp is np else (lambda a: a.get())
        np.savez_compressed(path, bias=get(self._bias), rest_out=get(self._rest_out))

    def load_calibration(self, path) -> None:
        z = np.load(path)
        self._bias, self._rest_out = self.xp.asarray(z["bias"]), self.xp.asarray(z["rest_out"])
        self.calibrated = True
