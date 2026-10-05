"""Shunting inhibition strength, chosen on physiology only (gratings and edges), never on looming.

    python tune_shunt.py

For each strength: the T4/T5 direction selectivity on gratings (mean DSI over the 16 subtype x eye groups,
preferred = the real fly's direction, Maisak et al. 2013) and the ON/OFF split on edges. Rule, fixed in
advance: the smallest strength whose mean DSI reaches 0.3 (the low end of real T4/T5) with the ON/OFF
split intact; if none does, the one with the highest mean DSI, reported as such.
"""
import io
import contextlib
import re

import physio
from flyears import hybrid


def run(strength):
    hybrid.SHUNT = strength
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        physio.main()
    out = buf.getvalue()
    dsi = float(re.search(r"mean DSI ([+-][\d.]+)", out).group(1))
    best = int(re.search(r"for (\d+) of 16", out).group(1))
    t4 = re.search(r"T4 \(should prefer bright\)\s+([+-][\d.]+)\s+([+-][\d.]+)", out)
    t5 = re.search(r"T5 \(should prefer dark\)\s+([+-][\d.]+)\s+([+-][\d.]+)", out)
    onoff = float(t4.group(1)) > float(t4.group(2)) and float(t5.group(2)) > float(t5.group(1))
    return dsi, best, onoff, out


def main():
    rows = []
    for strength in (0, 2, 5, 10, 20, 40):
        dsi, best, onoff, out = run(strength)
        rows.append((strength, dsi, best, onoff))
        print(f"  shunt {strength:3d}: mean DSI {dsi:+.2f}, real direction best in {best}/16, ON/OFF {'ok' if onoff else 'BROKEN'}",
              flush=True)
    ok = [r for r in rows if r[1] >= 0.3 and r[3]]
    pick = ok[0] if ok else max((r for r in rows if r[3]), key=lambda r: r[1])
    print("pick:", pick[0], "(meets the rule)" if ok else "(none reached 0.3: highest DSI with ON/OFF intact)")


if __name__ == "__main__":
    main()
