"""The fly watches a video: every frame through its eyes, the soundtrack through its ears.

    python watch.py "https://www.youtube.com/watch?v=..."      # needs yt-dlp (pip install yt-dlp)
    python watch.py my_video.mp4 --start 30 --seconds 60
    python watch.py my_video.mp4 --rewired                      # a scrambled brain, for comparison

Each video frame (25 a second) is two brain steps (20 ms each). Out (out/watch/<name>/):
  fly_watching.mp4   the video with the fly's brain beside it, live, with the original sound
  timeline.csv       every 20 ms: spike rates of each neuron group, looming, ear bands
  summary.json       how strongly each group responded to the video compared with rest

Use videos you have the right to download and use; downloads stay in out/watch/downloads.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from flyears import audio, body, ear, listen, senses, vision

HERE = Path(__file__).resolve().parent
FPS = 25                                 # set per video in main(): 50 when the source has 50+ frames a second,
STEPS_PER_FRAME = 2                      # so each 20 ms brain step gets its own frame; else 25 (40 ms = 2 steps)
FULL = 10.0                              # spikes/neuron/s above rest that fills a meter (fixed, every video)
VISION = "fly-own"                       # set in main()
REST_SETTLE, REST_MEASURE = 100, 100     # steps: 2 s to settle, then 2 s of rest measured
VW, VH = 768, 432                        # the video in the output
OUT_W, OUT_H = 1280, 720
FONT = Path("C:/Windows/Fonts/consola.ttf")
CASE, SCREEN, AMBER, GREEN, RED, CYAN, DIM = ((203, 199, 190), (21, 22, 25), (255, 178, 62), (108, 240, 138),
                                               (255, 90, 90), (62, 216, 255), (60, 62, 68))


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- getting the video
def fetch(source: str, folder: Path) -> Path:
    if not re.match(r"https?://", source):
        p = Path(source)
        if not p.exists():
            raise SystemExit(f"no such file: {p}")
        return p
    folder.mkdir(parents=True, exist_ok=True)
    try:
        import yt_dlp
    except ImportError:
        yt_dlp = None
    opts = {"format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]/b",
            "outtmpl": str(folder / "%(id)s.%(ext)s"), "merge_output_format": "mp4", "quiet": True,
            "noplaylist": True}
    if yt_dlp is not None:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(source, download=True)
            return Path(y.prepare_filename(info)).with_suffix(".mp4")
    if shutil.which("yt-dlp"):
        subprocess.run(["yt-dlp", "-f", opts["format"], "--merge-output-format", "mp4", "--no-playlist",
                        "-o", opts["outtmpl"], source], check=True)
        return max(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    raise SystemExit("YouTube links need yt-dlp: pip install yt-dlp (or download the video and give the file)")


def duration(src: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def source_fps(src: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
                        "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip()
    try:
        a, _, b = r.partition("/")
        return float(a) / float(b or 1)
    except (ValueError, ZeroDivisionError):
        return 25.0


def has_audio(src: Path) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def colour_frames(src: Path, start: float, seconds: float) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(seconds), "-i", str(src), "-vf",
                          f"fps={FPS},scale={vision.FLOW_W}:{vision.FLOW_H},format=rgb24", "-f", "rawvideo", "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, vision.FLOW_H, vision.FLOW_W, 3)


def grey(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.uint8)


def small_frames(src: Path, start: float, seconds: float) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(seconds), "-i", str(src), "-vf",
                          f"fps={FPS},scale={vision.FLOW_W}:{vision.FLOW_H},format=gray", "-f", "rawvideo", "pipe:1"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, vision.FLOW_H, vision.FLOW_W)


def soundtrack(src: Path, start: float, seconds: float) -> np.ndarray:
    if not has_audio(src):
        return np.zeros(int(seconds * audio.EAR_SR), np.float32)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(seconds), "-i", str(src), "-ac", "1",
                          "-ar", str(audio.EAR_SR), "-f", "f32le", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


# ---------------------------------------------------------------- the fly watching
def recorded_groups(brain, eyes) -> dict[str, np.ndarray]:
    g = {"T4/T5 (motion)": eyes.cells if eyes.has_motion else np.array([], int),
         "LPLC2 (looming)": np.concatenate(list(eyes.lplc2.values())),
         "photoreceptors": np.asarray(brain.visual)}
    g.update(listen.groups(brain))
    g.update(body.groups(brain))                           # body parts, behaviour commands, states
    g.update(senses.systems(brain))                        # colour, smell, taste, ... and memory, compass, clock
    side = np.asarray(brain.side).astype(str)              # left vs right for the main pathways
    if eyes.has_motion:
        g["T4/T5 L"], g["T4/T5 R"] = eyes.cells[eyes.left], eyes.cells[~eyes.left]
    g["LPLC2 L"], g["LPLC2 R"] = eyes.lplc2["L"], eyes.lplc2["R"]
    ear_cells, _ = ear.ear_map(brain)
    for s in "LR":
        g[f"ear {s}"] = ear_cells[side[ear_cells] == s]
        for key, short in (("WED", "WED"), ("descending_neuron", "descending")):
            if key in g:
                g[f"{short} {s}"] = g[key][side[g[key]] == s]
    return {k: v for k, v in g.items() if len(v)}


RASTER = ("ear JO-A/B", "AMMC", "WED", "T4/T5 (motion)", "LPLC2 (looming)", "photoreceptors", "escape", "walk",
          "turn L", "turn R", "back up", "groom", "fear", "pleasure", "distress", "excitement", "desire", "anger",
          "mood", "wing power L", "wing power R", "leg front L", "leg front R", "descending_neuron")
RASTER_PER_GROUP = 4                    # real neurons per group shown in the browser's spike raster


def code_arrays(n: int, groups: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Groups can overlap (T4/T5 are also 'visual_projection'...): each neuron gets one code per array."""
    arrays: list[np.ndarray] = []
    for j, idx in enumerate(groups.values()):
        for a in arrays:
            if np.all(a[idx] == -1):
                a[idx] = j
                break
        else:
            a = np.full(n, -1)
            a[idx] = j
            arrays.append(a)
    return arrays


def make_brain(mode: str, device: str = "auto", rewired: bool = False, calibration: Path | None = None):
    """fly-own: the graded optic lobe + spiking brain (hybrid.py), resting levels set under a grey view;
    assisted / lamina: fly.ai's all-spiking brain."""
    if mode == "fly-own":
        from flyears.hybrid import HybridBrain
        brain = HybridBrain(device=device, rewired=rewired)
        if calibration is not None and calibration.exists():          # reproduce a saved run exactly
            brain.load_calibration(calibration)
            return brain
        eyes = senses.ColumnEyes(brain, vision.FLOW_W, vision.FLOW_H)
        brain.calibrate(eyes.drive(np.full((vision.FLOW_H, vision.FLOW_W, 3), 128, np.uint8)))
        return brain
    return listen.make_brain(1, device, rewired=rewired)


def simulate(brain, eyes, frames: np.ndarray, env: np.ndarray, hear: bool, rgb: np.ndarray | None = None,
             stim: "senses.Stimuli | None" = None):
    G = recorded_groups(brain, eyes)
    names = list(G)
    codes = code_arrays(brain.n, G)
    K = len(names)
    sizes = np.array([len(v) for v in G.values()], np.float32)
    cells, bands = ear.ear_map(brain)
    steps = len(frames) * STEPS_PER_FRAME
    rates = np.zeros((steps, K), np.float32)
    spikes, motion, loom = [], [], np.zeros((len(frames), 2), np.float32)
    rng = np.random.default_rng(0)                         # the raster's neurons, picked in advance
    raster_rows, raster_ids = [], []
    for gname in RASTER:
        if gname in G:
            ids = [int(x) for x in G[gname] if int(x) not in raster_ids]
            pick = ids if len(ids) <= RASTER_PER_GROUP else sorted(rng.choice(ids, RASTER_PER_GROUP, replace=False))
            raster_rows += [gname] * len(pick)
            raster_ids += [int(x) for x in pick]
    colour = None
    if rgb is not None:                                    # each photoreceptor at its eye column when the
        colour = senses.ColumnEyes(brain, vision.FLOW_W, vision.FLOW_H) if eyes.mode == "fly-own" \
            else senses.ColourEyes(brain, vision.FLOW_W)  # graded brain sees; else by azimuth
    kc = G.get("mushroom body (Kenyon cells)", np.array([], int))
    is_kc = np.zeros(brain.n, bool)
    is_kc[kc] = True
    epg = G.get("compass (EPG)", np.array([], int))
    epg_index = np.full(brain.n, -1)
    epg_index[epg] = np.arange(len(epg))
    kc_active = np.zeros(len(frames), np.float32)          # share of Kenyon cells firing in each frame
    tindex = np.full(brain.n, -1)                          # the eye maps show T4/T5 cells' own spikes
    if eyes.has_motion:
        tindex[eyes.cells] = np.arange(len(eyes.cells))
    t45 = np.zeros((len(frames), len(eyes.cells) if eyes.has_motion else 0), np.uint8)
    epg_bits = np.zeros((len(frames), len(epg)), bool)
    stim_on: list[list[str]] = []
    rindex = np.full(brain.n, -1)
    rindex[raster_ids] = np.arange(len(raster_ids))
    raster = np.zeros((len(frames), len(raster_ids)), bool)
    light = np.zeros(len(frames), np.float32)
    rest = np.zeros(K, np.float32)
    grey = np.full((vision.FLOW_H, vision.FLOW_W), 128, np.uint8)
    brain.reset(1)
    rest_view = colour.drive(np.full((vision.FLOW_H, vision.FLOW_W, 3), 128, np.uint8)) if colour else \
        eyes.photoreceptors(grey)
    for k in range(REST_SETTLE + REST_MEASURE):          # rest: a plain grey view, silence; the brain
        f = brain.step(eye_drive=rest_view)               # settles first, then rest is measured
        if k >= REST_SETTLE:
            rest += sum(np.bincount(c[f][c[f] >= 0], minlength=K) for c in codes) / REST_MEASURE
    eyes.prev = None
    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        photo, inject = eyes.see(frame)
        if colour is not None:
            photo = colour.drive(rgb[i])                   # R1-6 brightness, R8 green, R7 blue
        fired_frame = []
        for k in range(STEPS_PER_FRAME):
            s = i * STEPS_PER_FRAME + k
            sound = ear.injections(cells, bands, env[s][:, None]) if hear and s < len(env) else []
            sound = [(idx, float(np.ravel(a)[0])) for idx, a in sound]
            smell = stim.at(s * brain.dt) if stim is not None else []
            f = brain.step(eye_drive=photo, inject=inject + sound + smell)
            rates[s] = sum(np.bincount(c[f][c[f] >= 0], minlength=K) for c in codes)
            fired_frame.append(f)
            r = rindex[f]
            raster[i, r[r >= 0]] = True
            r = tindex[f]
            t45[i, r[r >= 0]] += 1
        spikes.append(np.concatenate(fired_frame))
        both = spikes[-1]
        if len(kc):
            kc_active[i] = len(np.unique(both[is_kc[both]])) / len(kc)
        e = epg_index[both]
        epg_bits[i, e[e >= 0]] = True
        stim_on.append(stim.on(i / FPS) if stim is not None else [])
        motion.append(t45[i])                              # spikes of each T4/T5 cell this frame (0..steps)
        loom[i] = eyes.last["loom"]
        light[i] = float(photo.mean())
        if (i + 1) % (FPS * 10) == 0 or i + 1 == len(frames):
            log(f"  watched {(i + 1) / FPS:.0f} s of {len(frames) / FPS:.0f} s "
                f"({(time.perf_counter() - t0) / (i + 1) * FPS:.1f} s per video second)")
    to_rate = 1 / (sizes * brain.dt)
    extra = {"raster": raster, "raster_rows": raster_rows, "light": light, "kc_active": kc_active,
             "epg": epg, "epg_bits": epg_bits, "stim_on": stim_on}
    return names, rates * to_rate, rest * to_rate, spikes, motion, loom, extra


# ---------------------------------------------------------------- the picture
def rect(img, x, y, w, h, color):
    img[y:y + h, x:x + w] = color


def screen(img, x, y, w, h):
    rect(img, x - 3, y - 3, w + 6, h + 6, (11, 11, 13))
    rect(img, x, y, w, h, SCREEN)


def meter(img, x, y, w, h, value, color):
    rect(img, x, y, w, h, (13, 14, 16))
    n = int(np.clip(value, 0, 1) * w)
    if n:
        seg = img[y:y + h, x:x + n]
        seg[:] = color
        seg[:, (np.arange(n) % 9) >= 7] = (13, 14, 16)


class Panel:
    BRAIN = (812, 72, 444, 300)
    EYES = ((812, 400, 214, 120), (1042, 400, 214, 120))
    METERS = (812, 556, 444, 140)
    LIST = ("ears (JO-A/B)", "ear JO-A/B"), ("auditory relay", "WED"), ("motion (T4/T5)", "T4/T5 (motion)"), \
        ("looming (LPLC2)", "LPLC2 (looming)"), ("descending neurons", "descending_neuron")

    def __init__(self, brain, eyes, names, rates, rest):
        self.names = names
        self.rest = rest
        base = np.zeros((OUT_H, OUT_W, 3), np.uint8)
        base[:] = CASE
        rect(base, 0, 0, OUT_W, 40, (37, 37, 40))
        rect(base, 0, 40, OUT_W, 6, (22, 22, 24))
        screen(base, 24, 72, VW, VH)
        bx, by, bw, bh = self.BRAIN
        screen(base, bx, by, bw, bh)
        for (x, y, w, h) in self.EYES:
            screen(base, x, y, w, h)
        mx, my, mw, mh = self.METERS
        screen(base, mx, my, mw, mh)
        screen(base, 24, 556, VW, 140)
        pos = np.asarray(brain.positions, np.float64)[:, :2]
        known = np.all(np.isfinite(pos), 1)
        pos = np.where(known[:, None], pos, np.nanmean(pos[known], 0))     # (a few neurons have no position)
        self.known = known
        lo, hi = pos[known].min(0), pos[known].max(0)
        sc = min((bw - 20) / (hi[0] - lo[0]), (bh - 20) / (hi[1] - lo[1]))
        xy = (pos - lo) * sc
        off = ((bw - (hi[0] - lo[0]) * sc) / 2, (bh - (hi[1] - lo[1]) * sc) / 2)
        self.bx = np.clip(xy[:, 0] + off[0], 0, bw - 1).astype(int)
        self.by = np.clip(xy[:, 1] + off[1], 0, bh - 1).astype(int)
        dots = np.zeros((bh, bw), np.float32)
        np.add.at(dots, (self.by[known], self.bx[known]), 1)
        sub = base[by:by + bh, bx:bx + bw].astype(np.float32)
        sub += (np.clip(dots / 6, 0, 1)[..., None] * 70)
        base[by:by + bh, bx:bx + bw] = np.clip(sub, 0, 255).astype(np.uint8)
        self.base = base
        self.glow = np.zeros((bh, bw), np.float32)
        self.eye_px = None
        if eyes.has_motion:
            ex = []
            for (x, y, w, h), side in zip(self.EYES, (True, False)):
                m = eyes.left == side
                front = np.where(side, 1 - (eyes.px[m] / (vision.FLOW_W - 1)) * 2, eyes.px[m] / (vision.FLOW_W - 1) * 2 - 1)
                px = np.clip((eyes.px[m] / (vision.FLOW_W - 1) - (0 if side else 0.5)) * 2 * (w - 8) + 4, 0, w - 2).astype(int)
                py = np.clip(eyes.py[m] / (vision.FLOW_H - 1) * (h - 8) + 4, 0, h - 2).astype(int)
                ex.append((m, px, py))
                base[y + py, x + px] = (44, 46, 52)                 # every cell, dim: the eye's shape
                del front
            self.eye_px = ex


    def draw(self, frame_rgb, i, rates_frame, spikes, motion, loom, env):
        img = self.base.copy()
        img[72:72 + VH, 24:24 + VW] = frame_rgb
        bx, by, bw, bh = self.BRAIN
        self.glow *= 0.55
        spikes = spikes[self.known[spikes]] if len(spikes) else spikes
        if len(spikes):
            np.add.at(self.glow, (self.by[spikes], self.bx[spikes]), 1.0)
        g = np.clip(self.glow / 2, 0, 1)[..., None]
        sub = img[by:by + bh, bx:bx + bw].astype(np.float32)
        img[by:by + bh, bx:bx + bw] = np.clip(sub * (1 - g) + np.array(AMBER) * g, 0, 255).astype(np.uint8)
        if self.eye_px is not None and motion is not None:
            for (x, y, w, h), (m, px, py) in zip(self.EYES, self.eye_px):
                d = np.clip(motion[m] / STEPS_PER_FRAME, 0, 1)
                on = d > 0.02
                for dx in (0, 1):
                    for dy in (0, 1):
                        img[y + py[on] + dy, x + px[on] + dx] = (np.array(GREEN)[None] * d[on, None] +
                                                                  np.array(DIM)[None] * (1 - d[on, None])).astype(np.uint8)
        mx, my, mw, mh = self.METERS
        for r, (label, key) in enumerate(self.LIST):
            if key in self.names:
                j = self.names.index(key)
                meter(img, mx + 190, my + 14 + r * 25, mw - 204, 14, (rates_frame[j] - self.rest[j]) / FULL,
                      (RED, CYAN, GREEN, (255, 90, 210), AMBER)[r])
        for b in range(ear.N_BANDS):                       # what the ears get, low to high
            h = int(np.clip(env[b], 0, 1) * 110)
            rect(img, 40 + b * 40, 556 + 125 - h, 30, h, CYAN)
        for s, v in enumerate(loom):
            meter(img, 420 + s * 180, 676, 160, 10, v / vision.CAP, (255, 90, 210))
        return img


def labels(seconds: float, title: str, rewired: bool) -> str:
    if not FONT.exists():
        return "null"
    f = str(FONT).replace("\\", "/").replace(":", "\\:")
    t = lambda s, x, y, size=16, color="0x2a2925": (  # noqa: E731
        f"drawtext=fontfile='{f}':text='{s}':x={x}:y={y}:fontsize={size}:fontcolor={color}")
    items = [t("FLYBRAIN", 22, 9, 22, "0xc41f29"), t("THE FLY IS WATCHING" + (" (REWIRED BRAIN)" if rewired else "") +
               (" - ASSISTED VISION" if VISION == "assisted" else " - FLY-OWN VISION"), 170, 13, 16, "0xebe5d7"),
             t(title[:60].replace("'", "").replace(":", " "), 520, 13, 15, "0x8f897d"),
             t("01 WHAT IT SEES AND HEARS", 24, 52, 14), t("02 ALL 166,700 NEURONS (AMBER = FIRING)", 812, 52, 14),
             t("03 LEFT EYE T4/T5 SPIKES", 812, 380, 14), t("RIGHT EYE T4/T5 SPIKES", 1042, 380, 14),
             t("04 RESPONSE", 812, 536, 14), t("05 EARS 80 Hz - 1.2 kHz", 24, 536, 14),
             t("LOOMING  L", 420, 660, 13, "0xebe5d7"), t("R", 600, 660, 13, "0xebe5d7")]
    for r, (label, _) in enumerate(Panel.LIST):
        items.append(t(label, 824, 566 + r * 25, 14, "0xebe5d7"))
    items.append(f"drawtext=fontfile='{f}':text='%{{pts\\:hms}}':x=1150:y=13:fontsize=16:fontcolor=0xebe5d7")
    return ",".join(items)


def render(src: Path, start: float, seconds: float, out: Path, panel: Panel, sim, env_frames, title, rewired, sound):
    names, rates, rest, spikes, motion, loom = sim[:6]
    n = len(spikes)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(seconds), "-i", str(src), "-vf",
                            f"fps={FPS},scale={VW}:{VH}:force_original_aspect_ratio=decrease,"
                            f"pad={VW}:{VH}:(ow-iw)/2:(oh-ih)/2", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                           stdout=subprocess.PIPE)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}",
           "-r", str(FPS), "-i", "pipe:0"]
    if sound:
        cmd += ["-ss", str(start), "-t", str(seconds), "-i", str(src), "-map", "0:v", "-map", "1:a", "-c:a", "aac"]
    cmd += ["-vf", labels(seconds, title, rewired), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-shortest", str(out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    per = rates.reshape(-1, STEPS_PER_FRAME, rates.shape[1]).mean(1)
    for i in range(n):
        buf = dec.stdout.read(VW * VH * 3)
        if len(buf) < VW * VH * 3:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(VH, VW, 3)
        enc.stdin.write(panel.draw(frame, i, per[i], spikes[i], motion[i], loom[i], env_frames[i]).tobytes())
    enc.stdin.close()
    enc.wait()
    dec.kill()


MAX_SPIKES = 3000                         # spikes kept per frame for the browser's brain map


def save_view(out: Path, brain, sim, src: Path, start: float, seconds: float, source: str, rewired: bool, hear: bool,
              env_frames: np.ndarray, eyes, schedule: list):
    """Data for the browser viewer: per-frame rates of every group, sampled spikes, eye maps, the ear's
    bands, a spike raster of real neurons, and a clean clip."""
    names, rates, rest, spikes, motion, loom = sim[:6]
    extra = sim[6]
    if eyes.has_motion:                                    # T4/T5 drive per frame, 0..255
        blank = np.zeros(len(eyes.cells), np.uint8)
        m = np.stack([blank if d is None else np.round(np.clip(d / STEPS_PER_FRAME, 0, 1) * 255).astype(np.uint8)
                      for d in motion])
        (out / "motion.bin").write_bytes(m.tobytes())
        eyes_file = out.parent / "eyes.json"
        if not eyes_file.exists():
            cols = vision.load_columns()
            hue = (np.degrees(np.arctan2(cols["pref"][:, 1], cols["pref"][:, 0])) % 360).round().astype(int)
            eyes_file.write_text(json.dumps({"left": eyes.left.astype(int).tolist(),
                                             "front": np.round(cols["front"], 3).tolist(),
                                             "up": np.round(cols["up"], 3).tolist(), "hue": hue.tolist()}))
    (out / "raster.bin").write_bytes(np.packbits(extra["raster"], axis=1).tobytes())
    epg = extra["epg"]
    epg_angle = []
    if len(epg) and getattr(brain, "positions", None) is not None:
        p = np.asarray(brain.positions, np.float64)[epg]
        p = p - np.nanmean(p, 0)                          # the ring of the ellipsoid body: its two widest axes
        _, _, vt = np.linalg.svd(np.nan_to_num(p), full_matrices=False)
        xy = np.nan_to_num(p) @ vt[:2].T
        epg_angle = np.round(np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) % 360).astype(int).tolist()
        (out / "compass.bin").write_bytes(np.packbits(extra["epg_bits"], axis=1).tobytes())
    per = rates.reshape(-1, STEPS_PER_FRAME, rates.shape[1]).mean(1)
    rng = np.random.default_rng(0)
    kept = [s if len(s) <= MAX_SPIKES else np.sort(rng.choice(s, MAX_SPIKES, replace=False)) for s in spikes]
    offsets = np.cumsum([0] + [len(s) for s in kept]).astype(np.uint32)
    (out / "spikes.bin").write_bytes(offsets.tobytes() + np.concatenate(kept).astype(np.uint32).tobytes())
    pos_file = out.parent / "positions.bin"
    if not pos_file.exists():
        pos = np.asarray(brain.positions, np.float64)[:, :2]
        known = np.all(np.isfinite(pos), 1)
        lo, hi = pos[known].min(0), pos[known].max(0)
        xy = np.where(known[:, None], (pos - lo) / (hi - lo).max() * 1000, -1)
        pos_file.write_bytes(xy.astype(np.int16).tobytes())
    state_rows = [n for n in body.STATES if n in names]
    view = {"source": source, "start": start, "seconds": seconds, "rewired": rewired, "sound": hear, "fps": FPS,
            "frames": len(per), "names": names, "rest": [round(float(x), 4) for x in rest],
            "rates": [[round(float(x), 3) for x in row] for row in per], "loom": np.round(loom, 3).tolist(),
            "states": state_rows, "state_source": body.STATE_SOURCE, "neurons": int(brain.n),
            "sizes": [int(len(x)) for x in recorded_groups(brain, eyes).values()],
            "ear": np.round(env_frames, 3).tolist(), "bands": np.round(ear.BAND_EDGES).astype(int).tolist(),
            "light": np.round(extra["light"], 3).tolist(), "has_motion": bool(eyes.has_motion),
            "eye_cells": int(len(eyes.cells)) if eyes.has_motion else 0,
            "raster_rows": extra["raster_rows"], "raster_bytes": int((len(extra["raster_rows"]) + 7) // 8),
            "kc_active": np.round(extra["kc_active"], 4).tolist(), "epg_angle": epg_angle,
            "epg_bytes": int((len(epg_angle) + 7) // 8), "stim_on": extra["stim_on"], "schedule": schedule,
            "stimuli": {k: {"sense": v[2], "what": v[3]} for k, v in senses.STIMULI.items()}, "colour": True,
            "vision": eyes.mode}
    (out / "view.json").write_text(json.dumps(view), encoding="utf-8")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(seconds), "-i", str(src), "-vf",
                    "scale=854:-2", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-c:a", "aac",
                    str(out / "clip.mp4")], check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="YouTube (or other) link, or a video file")
    p.add_argument("--start", type=float, default=0.0, help="start at this second")
    p.add_argument("--seconds", type=float, default=60.0, help="how much to watch")
    p.add_argument("--rewired", action="store_true", help="a degree-preserving scrambled brain")
    p.add_argument("--no-sound", action="store_true", help="eyes only")
    p.add_argument("--vision", default="fly-own", choices=["fly-own", "assisted", "lamina"],
                   help="fly-own (default): light into the photoreceptors at their eye columns, a graded optic lobe "
                        "(as in a real fly) and the spiking brain; nothing of ours detects anything. assisted: the "
                        "all-spiking brain plus this program's optical flow into T4/T5 and looming detector into LPLC2. "
                        "lamina: the all-spiking brain with darkening into L2/L3 (the earlier workaround)")
    p.add_argument("--stim", default="", help="smells, tastes, wind, temperature, humidity, touch on a schedule, "
                   "e.g. 'vinegar:5-15,heat:20-30' (seconds into the clip); names: " + ", ".join(senses.STIMULI))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--out", default=None, help="output folder (default out/watch/<name>)")
    a = p.parse_args()
    global FPS, STEPS_PER_FRAME, VISION
    src = fetch(a.source, HERE / "out" / "watch" / "downloads")
    FPS = 50 if source_fps(src) >= 48 else 25       # a real frame for every 20 ms step when the video has one
    STEPS_PER_FRAME = 50 // FPS
    VISION = a.vision
    total = duration(src)
    seconds = max(0.5, min(a.seconds, total - a.start)) if total else a.seconds
    whole = a.start == 0 and (not total or seconds >= total - 0.5)
    part = "" if whole else f"_{int(a.start // 60)}m{int(a.start % 60):02d}s_{seconds:g}s"   # parts kept apart
    tag = "_" + re.sub(r"[^\w]+", "-", a.stim.replace(":", "")).strip("-")[:40] if a.stim else ""
    name = re.sub(r"[^\w-]+", "_", src.stem)[:60] + part + tag + ("" if a.vision == "fly-own" else "_" + a.vision) + \
        ("_rewired" if a.rewired else "")
    out = Path(a.out) if a.out else HERE / "out" / "watch" / name
    out.mkdir(parents=True, exist_ok=True)
    log(f"{src.name}: watching {seconds:.1f} s from {a.start:.1f} s")
    try:
        schedule = senses.parse_schedule(a.stim)
    except ValueError as e:
        raise SystemExit(str(e))
    rgb = colour_frames(src, a.start, seconds)
    frames = grey(rgb)
    x = soundtrack(src, a.start, seconds)
    hear = bool(not a.no_sound and np.abs(x).max() > 1e-4)
    env = ear.envelopes(audio.level(x)) if hear else np.zeros((len(frames) * STEPS_PER_FRAME, ear.N_BANDS), np.float32)
    need = len(frames) * STEPS_PER_FRAME
    env = np.vstack([env, np.zeros((max(0, need - len(env)), ear.N_BANDS), np.float32)])[:need]
    brain = make_brain(a.vision, a.device, rewired=a.rewired)
    if hasattr(brain, "save_calibration"):                # kept with the run, so it can be reproduced
        brain.save_calibration(out / "calibration.npz")
    eyes = vision.Eyes(brain, fps=FPS, mode=a.vision)
    if not eyes.has_motion:
        log("  (no columns.npz in the fly data: motion detectors off, photoreceptors and looming only)")
    log(f"  vision: {a.vision}" + {"fly-own": " (photoreceptors at their eye columns, graded optic lobe, nothing of ours)",
                                       "assisted": " (adds this program's motion and looming detectors)",
                                       "lamina": " (all-spiking brain, darkening into L2/L3)"}[a.vision])
    log(f"  {len(frames)} frames at {FPS} fps ({1000 // FPS} ms each = {STEPS_PER_FRAME} brain step"
        f"{'s' if STEPS_PER_FRAME > 1 else ''}), sound: {'yes' if hear else 'no'}, brain on {brain.device}")
    stim = senses.Stimuli(brain, schedule) if schedule else None
    if schedule:
        log("  stimuli: " + ", ".join(f"{n} {s0:g}-{s1:g} s" for n, s0, s1 in schedule))
    sim = simulate(brain, eyes, frames, env, hear, rgb=rgb, stim=stim)
    names, rates, rest = sim[0], sim[1], sim[2]
    header = "time_s," + ",".join(n.replace(",", " ") for n in names) + ",loom_L,loom_R," + \
             ",".join(f"ear_band_{b}" for b in range(ear.N_BANDS))
    loom_steps = np.repeat(sim[5], STEPS_PER_FRAME, 0)
    t = np.arange(len(rates)) * brain.dt
    np.savetxt(out / "timeline.csv", np.column_stack([t, rates, loom_steps, env]), delimiter=",", header=header,
               comments="", fmt="%.4g")
    mean = rates.mean(0)
    summary = sorted(({"group": n, "rest": float(rest[j]), "watching": float(mean[j]),
                       "change": float(mean[j] - rest[j])} for j, n in enumerate(names)),
                     key=lambda r: -abs(r["change"]))
    (out / "summary.json").write_text(json.dumps({"source": a.source, "start": a.start, "seconds": seconds,
                                                  "rewired": a.rewired, "sound": hear, "groups": summary}, indent=1))
    log("  biggest changes (spikes per neuron per second, at rest -> while watching):")
    for r in summary[:8]:
        log(f"    {r['group']:22s} {r['rest']:7.3f} -> {r['watching']:7.3f}  ({r['change']:+.3f})")
    env_frames = env.reshape(-1, STEPS_PER_FRAME, ear.N_BANDS).mean(1)
    save_view(out, brain, sim, src, a.start, seconds, a.source, a.rewired, hear, env_frames, eyes, schedule)
    panel = Panel(brain, eyes, names, rates, rest)
    log("  drawing the video...")
    part = out / "fly_watching.part.mp4"              # finished videos only: a stop mid-way leaves no broken file
    render(src, a.start, seconds, part, panel, sim[:6], env_frames, src.stem, a.rewired, sound=has_audio(src))
    part.replace(out / "fly_watching.mp4")
    body_py = Path(os.environ.get("FLY_BODY_PY") or "D:/fly-body/.venv/Scripts/python.exe")
    if body_py.exists():                              # the 3D body (NeuroMechFly), in its own environment
        log("  posing the 3D body (NeuroMechFly)...")
        r = subprocess.run([str(body_py), str(HERE / "body3d.py"), str(out)], capture_output=True, text=True)
        log("  " + (r.stdout.strip().splitlines() or ["(no output)"])[-1] if r.returncode == 0
            else f"  3D body skipped: {(r.stderr.strip().splitlines() or ['error'])[-1]}")
    log(f"done: {out / 'fly_watching.mp4'}")


if __name__ == "__main__":
    main()
