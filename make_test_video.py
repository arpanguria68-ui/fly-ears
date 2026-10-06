"""Make out/testvideo/test_events.mp4, the test video with known events: 0-2 s a dark bar moves right, 2-4 s it moves left,
4-6 s a dark disc rushes toward the camera (looming), 6-7 s still. 300 Hz beeps 0.5-3.5 s.

    python make_test_video.py
"""
import subprocess
from pathlib import Path
import numpy as np
from scipy.io import wavfile
Path("out/testvideo").mkdir(parents=True, exist_ok=True)
W, H, FPS, T = 640, 360, 25, 7.0
n = int(T * FPS)
yy, xx = np.mgrid[0:H, 0:W]
rng = np.random.default_rng(0)
tex = (rng.random((H // 8, W // 8)) * 60 + 150).repeat(8, 0).repeat(8, 1)   # mild texture background
frames = []
for i in range(n):
    t = i / FPS
    img = tex.copy()
    if t < 2:
        x = 40 + (t / 2) * 560
        img[np.abs(xx - x) < 30] = 20
    elif t < 4:
        x = 600 - ((t - 2) / 2) * 560
        img[np.abs(xx - x) < 30] = 20
    elif t < 6:
        r = 10 * np.exp((t - 4) * 1.9)
        img[(xx - W / 2) ** 2 + (yy - H / 2) ** 2 < r ** 2] = 15
    frames.append(np.repeat(img[..., None], 3, 2).astype(np.uint8))
sr = 22050
ts = np.arange(int(T * sr)) / sr
beep = (np.sin(2 * np.pi * 300 * ts) * ((ts % 0.5) < 0.25) * ((ts > 0.5) & (ts < 3.5)) * 0.5).astype(np.float32)
wavfile.write("out/testvideo/beeps.wav", sr, beep)
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                      "-i", "pipe:0", "-i", "out/testvideo/beeps.wav", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                      "-shortest", "out/testvideo/test_events.mp4"], stdin=subprocess.PIPE)
p.stdin.write(np.stack(frames).tobytes()); p.stdin.close(); p.wait()
print("made out/testvideo/test_events.mp4")
