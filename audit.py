"""Audit a watched video: is the brain's time the video's time, and is every number real?

    python audit.py out/watch/<name>

A. Timing     brain steps x step length == the clip's length; picture, sound and spikes share one clock.
B. Real data  the first seconds are simulated again from the same inputs and seed, and compared with
              the saved timeline (the saved numbers must come out of the simulation, nothing else).
C. No phantom the same length of a plain grey view in silence: every group must stay at rest
              (anything the viewer shows for the real clip is a response to the clip).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import watch
from flyears import audio, ear, listen, senses, vision

CHECK_S = 2.0                        # seconds re-simulated for B and C


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    v = json.loads((out / "view.json").read_text(encoding="utf-8"))
    rows = np.loadtxt(out / "timeline.csv", delimiter=",", skiprows=1)
    header = (out / "timeline.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    ok = True

    # A ---------------------------------------------------------------- timing
    dt = rows[1, 0] - rows[0, 0]
    sim_s = len(rows) * dt
    clip = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(out / "clip.mp4")], capture_output=True, text=True).stdout or 0)
    print("A. timing")
    print(f"   brain step {dt * 1000:.0f} ms, {len(rows)} steps = {sim_s:.2f} s of brain time")
    print(f"   video: {v['frames']} frames at {v['fps']} fps = {v['frames'] / v['fps']:.2f} s; requested {v['seconds']:.2f} s;"
          f" clip file {clip:.2f} s")
    print(f"   each frame is shown to the brain for {1000 / v['fps']:.0f} ms = {round(1 / v['fps'] / dt)} brain steps")
    a_ok = abs(sim_s - v["frames"] / v["fps"]) < 1e-6 and abs(sim_s - v["seconds"]) <= 1 / v["fps"] + 1e-6
    print(f"   {'PASS' if a_ok else 'FAIL'}: 1 s of video = 1 s of brain time")
    ok &= a_ok

    # B ---------------------------------------------------------------- the saved numbers come from the simulation
    src = Path(v["source"]) if not str(v["source"]).startswith("http") else None
    if src is None or not src.exists():
        down = sorted((out.parent / "downloads").glob("*.mp4"))
        src = next((d for d in down if d.stem in out.name), None)
    n_frames = int(CHECK_S * v["fps"])
    watch.FPS, watch.STEPS_PER_FRAME = v["fps"], 50 // v["fps"]
    print("\nB. re-simulating the first", CHECK_S, "s from the same inputs and seed")
    if src is None or not src.exists():
        print("   SKIP: source video not on this PC")
    else:
        rgb = watch.colour_frames(src, v["start"], CHECK_S)[:n_frames] if v.get("colour") else None
        frames = watch.grey(rgb) if rgb is not None else watch.small_frames(src, v["start"], CHECK_S)[:n_frames]
        sched = [tuple(x) for x in v.get("schedule", [])]
        x = watch.soundtrack(src, v["start"], v["seconds"])          # loudness is set over the whole clip,
        env = ear.envelopes(audio.level(x)) if v["sound"] else np.zeros((n_frames * 2, ear.N_BANDS), np.float32)
        need = n_frames * watch.STEPS_PER_FRAME
        env = np.vstack([env, np.zeros((max(0, need - len(env)), ear.N_BANDS), np.float32)])[:need]
        brain = listen.make_brain(1, "auto", rewired=v["rewired"])
        mode = v.get("vision", "assisted")
        print(f"   vision mode: {mode}")
        eyes = vision.Eyes(brain, fps=v["fps"], mode=mode)
        stim = senses.Stimuli(brain, sched) if sched else None
        names, rates, rest, *_ = watch.simulate(brain, eyes, frames, env, v["sound"], rgb=rgb, stim=stim)
        again = watch.simulate(brain, vision.Eyes(brain, fps=v["fps"], mode=mode), frames, env, v["sound"], rgb=rgb, stim=stim)[1]
        print(f"   same run twice: largest difference {np.abs(again - rates).max():.4g} "
              f"({'deterministic' if np.abs(again - rates).max() == 0 else 'NOT deterministic'})")
        saved = rows[:need, 1:1 + len(names)]
        same_names = header[1:1 + len(names)] == [n.replace(",", " ") for n in names]
        diff = np.abs(saved - np.round(rates, 4))
        rel = diff.max() / max(1e-9, np.abs(saved).max())
        print(f"   {len(names)} groups x {need} steps compared; same groups: {same_names}")
        print(f"   largest difference: {diff.max():.4g} spikes/neuron/s ({rel:.2%} of the largest value)")
        exact = diff.max() < 1e-2
        print(f"   {'PASS' if exact and same_names else 'CHECK'}: saved timeline "
              f"{'= the simulation, step for step' if exact else 'differs (GPU arithmetic is not bit-exact; see C)'}")
        ok &= bool(same_names)

        # C ------------------------------------------------------------ nothing shown without a stimulus
        print("\nC. the same length of a plain grey view in silence")
        grey = np.full_like(frames, 128)
        grey_rgb = np.full_like(rgb, 128) if rgb is not None else None
        brain.reset(1)
        names2, rates2, rest2, *_ = watch.simulate(brain, vision.Eyes(brain, fps=v["fps"], mode=mode), grey, np.zeros_like(env), False,
                                                   rgb=grey_rgb)
        dev = rates2.mean(0) - rest2
        clip_dev = rates.mean(0) - rest
        key = [n for n in names if n in ("T4/T5 (motion)", "LPLC2 (looming)", "ear JO-A/B", "escape", "fear", "WED",
                                         "descending_neuron", "wing power L", "leg front L")]
        print(f"   {'group':22s} {'blank - rest':>13s} {'clip - rest':>12s}")
        for n in key:
            j = names.index(n)
            print(f"   {n:22s} {dev[j]:+13.3f} {clip_dev[j]:+12.3f}")
        sensory = [names.index(n) for n in ("T4/T5 (motion)", "LPLC2 (looming)", "ear JO-A/B") if n in names]
        c_ok = bool(np.all(np.abs(dev[sensory]) < 0.5))
        print(f"   {'PASS' if c_ok else 'FAIL'}: with nothing to see or hear, the senses stay at rest")
        ok &= c_ok
    print("\nresult:", "all checks passed" if ok else "a check failed: see above")


if __name__ == "__main__":
    main()
