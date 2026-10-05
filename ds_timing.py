"""How T4/T5 direction selectivity depends on timing: the slow inputs' time constant and the grating speed.

    python ds_timing.py
"""
import contextlib
import io
import re

import physio
from flyears import hybrid

DSI = re.compile(r"mean DSI ([+-][\d.]+)")
BEST = re.compile(r"for (\d+) of 16")


def measure(tau_slow, hz):
    hybrid.TAU["slow"] = tau_slow
    physio.SPEED = 24.0 * hz
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        physio.main()
    out = buf.getvalue()
    return float(DSI.search(out).group(1)), int(BEST.search(out).group(1))


def main():
    for tau in (0.05, 0.15):
        for hz in (1, 3):
            dsi, best = measure(tau, hz)
            print(f"slow tau {tau * 1000:4.0f} ms, grating {hz} Hz: mean DSI {dsi:+.2f}, real direction best {best}/16",
                  flush=True)


if __name__ == "__main__":
    main()
