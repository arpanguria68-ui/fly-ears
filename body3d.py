"""The fly's body in 3D: NeuroMechFly (EPFL), posed by the connectome's motor neurons.

    D:\\fly-body\\.venv\\Scripts\\python.exe body3d.py out\\watch\\<name>      -> out/watch/<name>/body3d.mp4

NeuroMechFly (flygym, Apache-2.0; Lobato-Rios et al. 2022 Nature Methods; Wang-Chen et al. 2024 Nature
Methods) is an anatomically accurate fly body from a micro-CT scan of a real fly: 6 legs with all their
segments, wings, halteres, head, antennae, proboscis, a 5-segment abdomen. It runs in its own Python
environment (D:\\fly-body\\.venv, Python 3.12); this script only reads the watched video's view.json.

How the pose is set, frame by frame, from the brain's spikes (view.json, the same numbers as the viewer):
each part moves away from NeuroMechFly's neutral (resting) pose in proportion to its motor neurons'
activity above rest, on the fixed scale (FULL spikes/neuron/s above rest = the full movement):
    legs      each leg flexes (femur-tibia) and lifts with its own leg motor neurons (front/middle/hind, L/R)
    jump      the jump muscle neurons (TTMn) extend the middle legs, as the real jump muscle does
    wings     spread from folded with the flight power neurons, tilt with the steering ones (per side)
    head      turns toward the side whose neck motor neurons are more active
    abdomen   bends with the abdominal motor neurons
This is kinematic posing (no muscles, no physics, no gait generator: a real gait would need a rhythm the
connectome's motor neurons do not produce in this model, so none is invented), the same rule as the 2D
drawing, on the real anatomy.
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import mujoco as mj                                         # noqa: E402
from flygym.anatomy import AxisOrder, JointPreset, Skeleton  # noqa: E402
from flygym.compose.fly.neuromechfly import NeuroMechFly    # noqa: E402
from flygym.compose.pose import KinematicPosePreset         # noqa: E402

FULL = 10.0                 # spikes/neuron/s above rest = the full movement (same fixed scale as the viewer)
SMOOTH = 0.3                # per-frame smoothing of activity (as the viewer)
W, H = 640, 480

# joint offsets (radians) at full activity; signs checked by rendering each one (pose_check.png)
LEG = {"femur": ("{leg}_coxa-{leg}_trochanterfemur-pitch", -0.6),
       "tibia": ("{leg}_trochanterfemur-{leg}_tibia-pitch", 0.9),
       "coxa": ("c_thorax-{leg}_coxa-pitch", 0.35)}
JUMP_TIBIA = -1.0           # middle legs extend
WING_SPREAD, WING_TILT = 1.1, 0.3   # roll spreads a wing out to the side (flight position); pitch tilts it
HEAD_YAW = 0.45
ABDOMEN_PITCH = 0.12        # per abdominal joint


def build():
    fly = NeuroMechFly()
    fly.add_joints(Skeleton(axis_order=AxisOrder.PITCH_ROLL_YAW, joint_preset=JointPreset.ALL_BIOLOGICAL),
                   neutral_pose=KinematicPosePreset.NEUTRAL)
    try:
        fly.colorize()
    except Exception:                                      # noqa: BLE001  (plain grey if the textures fail)
        pass
    model, data = fly.compile()
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    adr = {n: model.jnt_qposadr[j] for j, n in enumerate(names)}
    neutral = {j.name: a for j, a in fly.jointdof_to_neutralangle.items()}
    return model, data, adr, neutral


def pose(data, adr, neutral, act):
    """act: name -> 0..1 activity of each body group above rest."""
    q = dict(neutral)
    def add(name, v):
        if name in adr:
            q[name] = q.get(name, 0.0) + v
    for side, s in (("l", "L"), ("r", "R")):
        for pos, word in (("f", "front"), ("m", "middle"), ("h", "hind")):
            a = act.get(f"leg {word} {s}", 0.0)
            leg = side + pos
            for name, k in LEG.values():
                add(name.format(leg=leg), k * a)
            if pos == "m":
                add(f"{leg}_trochanterfemur-{leg}_tibia-pitch", JUMP_TIBIA * act.get("jump", 0.0))
        wp, ws = act.get(f"wing power {s}", 0.0), act.get(f"wing steer {s}", 0.0)
        add(f"c_thorax-{side}_wing-roll", WING_SPREAD * wp)
        add(f"c_thorax-{side}_wing-pitch", WING_TILT * ws)
    add("c_thorax-c_head-yaw", HEAD_YAW * (act.get("neck L", 0.0) - act.get("neck R", 0.0)))
    for seg in ("c_thorax-c_abdomen12", "c_abdomen12-c_abdomen3", "c_abdomen3-c_abdomen4", "c_abdomen4-c_abdomen5"):
        add(f"{seg}-pitch", ABDOMEN_PITCH * act.get("abdomen", 0.0))
    for n, a in adr.items():
        data.qpos[a] = q.get(n, 0.0)


PARTS = ["leg front L", "leg middle L", "leg hind L", "leg front R", "leg middle R", "leg hind R", "jump",
         "wing power L", "wing power R", "wing steer L", "wing steer R", "neck L", "neck R", "abdomen"]


def main():
    out = Path(sys.argv[1]).resolve()
    v = json.loads((out / "view.json").read_text(encoding="utf-8"))
    idx = {n: i for i, n in enumerate(v["names"])}
    rates, rest = np.asarray(v["rates"], np.float32), np.asarray(v["rest"], np.float32)
    model, data, adr, neutral = build()
    r = mj.Renderer(model, H, W)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    pose(data, adr, neutral, {})
    mj.mj_forward(model, data)
    cam.lookat[:] = data.xpos.mean(0)
    cam.distance, cam.azimuth, cam.elevation = 6.5, 125.0, -40.0
    smooth = {p: 0.0 for p in PARTS}
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(v["fps"]), "-i", "pipe:0", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                            str(out / "body3d.part.mp4")], stdin=subprocess.PIPE)
    for i in range(v["frames"]):
        act = {}
        for p in PARTS:
            if p in idx:
                x = float(np.clip((rates[i, idx[p]] - rest[idx[p]]) / FULL, 0, 1))
                smooth[p] = smooth[p] * (1 - SMOOTH) + x * SMOOTH
                act[p] = smooth[p]
        pose(data, adr, neutral, act)
        mj.mj_forward(model, data)
        r.update_scene(data, camera=cam)
        enc.stdin.write(r.render().tobytes())
    enc.stdin.close()
    enc.wait()
    (out / "body3d.part.mp4").replace(out / "body3d.mp4")
    print(f"done: {out / 'body3d.mp4'}")


if __name__ == "__main__":
    main()
