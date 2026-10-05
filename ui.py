"""fly-ears in the browser: give the fly a video (link or file) or run the music test, see the results.

    python ui.py              # then open http://127.0.0.1:8790/

One job at a time (the brain uses the GPU). Jobs run watch.py / run.py, so the browser and the
command line give the same results. Everything stays on this PC.
"""
from __future__ import annotations

import json
import mimetypes
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
WATCH = HERE / "out" / "watch"
UPLOADS = WATCH / "uploads"
PORT = 8790
VIDEO_TYPES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


class Job:
    """The one running (or last) job: its command, its log, where its result is."""

    def __init__(self):
        self.lock = threading.Lock()
        self.kind = None
        self.state = "idle"            # idle | running | done | error
        self.lines: list[str] = []
        self.result = None
        self.started = 0.0
        self.proc = None
        self.stopping = False

    def start(self, kind: str, cmd: list[str]) -> bool:
        with self.lock:
            if self.state == "running":
                return False
            self.kind, self.state, self.lines, self.result, self.started = kind, "running", [], None, time.time()
            self.stopping = False
        threading.Thread(target=self._run, args=(cmd,), daemon=True).start()
        return True

    def _run(self, cmd):
        try:
            self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", bufsize=1)
            for line in self.proc.stdout:
                line = line.rstrip()
                if not line or "UserWarning" in line or "warnings.warn" in line:
                    continue
                with self.lock:
                    self.lines.append(line)
                    self.lines = self.lines[-300:]
                m = re.match(r"(done|report): (.+)", line)
                if m:
                    self.result = Path(m.group(2).strip())
            code = self.proc.wait()
            self.state = "stopped" if self.stopping else "done" if code == 0 else "error"
        except Exception as e:                            # noqa: BLE001
            self.lines.append(f"error: {e}")
            self.state = "error"

    def status(self) -> dict:
        with self.lock:
            res = None
            if self.result is not None:
                try:
                    res = self.result.parent.relative_to(WATCH if self.kind == "watch" else HERE / "out").as_posix()
                except ValueError:
                    res = self.result.parent.name
            prog = None
            for line in reversed(self.lines):
                m = re.search(r"watched (\d+) s of (\d+) s", line)
                if m:
                    prog = int(m.group(1)) / max(1, int(m.group(2)))
                    break
            return {"kind": self.kind, "state": self.state, "log": self.lines[-60:], "result": res,
                    "progress": prog, "drawing": any("drawing the video" in x for x in self.lines),
                    "seconds": round(time.time() - self.started) if self.started else 0}


JOB = Job()


def library() -> list[dict]:
    out = []
    if not WATCH.is_dir():
        return out
    for d in WATCH.iterdir():
        mp4, summ = d / "fly_watching.mp4", d / "summary.json"
        if mp4.exists() and summ.exists():
            s = json.loads(summ.read_text(encoding="utf-8"))
            out.append({"name": d.name, "view": (d / "view.json").exists(), "source": s.get("source"), "start": s.get("start"), "seconds": s.get("seconds"),
                        "rewired": s.get("rewired"), "sound": s.get("sound"), "when": mp4.stat().st_mtime,
                        "top": s.get("groups", [])[:6]})
    return sorted(out, key=lambda r: -r["when"])


def safe_child(base: Path, rel: str) -> Path | None:
    p = (base / rel).resolve()
    return p if p.is_file() and base.resolve() in p.parents else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj).encode(), "application/json", status)

    def _file(self, path: Path):
        """Serve a file, with byte ranges so the video player can seek."""
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if a and b else size - 1
            end = min(end, size - 1)
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    return
                left -= len(chunk)

    def do_HEAD(self):
        """Whether a result file exists (the viewer asks before showing the 3D body)."""
        path = unquote(urlparse(self.path).path)
        f = safe_child(WATCH, path[len("/media/"):]) if path.startswith("/media/") else None
        self.send_response(200 if f else 404)
        if f:
            self.send_header("Content-Length", str(f.stat().st_size))
            self.send_header("Content-Type", mimetypes.guess_type(str(f))[0] or "application/octet-stream")
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        if path == "/":
            return self._send((HERE / "ui.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/view":
            return self._send((HERE / "view.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/retro.css":
            return self._send((HERE / "retro.css").read_bytes(), "text/css; charset=utf-8")
        if path == "/job":
            return self._json(JOB.status())
        if path == "/library":
            return self._json(library())
        if path.startswith("/media/"):
            f = safe_child(WATCH, path[len("/media/"):])
            return self._file(f) if f else self.send_error(404)
        if path == "/report":
            name = parse_qs(url.query).get("name", [""])[0]
            f = safe_child(HERE / "out", f"{name}/report.md")
            return self._send(f.read_bytes(), "text/plain; charset=utf-8") if f else self.send_error(404)
        if path == "/reports":
            items = [p.parent.relative_to(HERE / "out").as_posix() for p in (HERE / "out").rglob("report.md")]
            return self._json(sorted(items))
        self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if url.path == "/upload":                           # a video from this PC, streamed to disk
            name = Path(parse_qs(url.query).get("name", ["video.mp4"])[0]).name
            if Path(name).suffix.lower() not in VIDEO_TYPES:
                return self._json({"ok": False, "error": "not a video file"}, 400)
            UPLOADS.mkdir(parents=True, exist_ok=True)
            target = UPLOADS / name
            left = length
            with open(target, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            return self._json({"ok": True, "path": str(target)})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._json({"ok": False, "error": "could not read the request"}, 400)
        if url.path == "/watch":
            src = str(body.get("source") or "").strip().strip('"')
            if not src:
                return self._json({"ok": False, "error": "give a link or a video file"}, 400)
            if not re.match(r"https?://", src) and not (Path(src) if Path(src).is_absolute() else HERE / src).is_file():
                return self._json({"ok": False, "error": f"no such file: {src}"}, 400)
            if not re.match(r"https?://", src) and not Path(src).is_absolute():
                src = str(HERE / src)
            cmd = [sys.executable, "-u", "-W", "ignore", "watch.py", src, "--start", str(float(body.get("start") or 0)),
                   "--seconds", str(float(body.get("seconds") or 60))]
            if body.get("rewired"):
                cmd.append("--rewired")
            if body.get("assisted"):
                cmd += ["--vision", "assisted"]
            if body.get("stim"):
                from flyears import senses
                try:
                    senses.parse_schedule(str(body["stim"]))
                except ValueError as e:
                    return self._json({"ok": False, "error": str(e)}, 400)
                cmd += ["--stim", str(body["stim"])]
            ok = JOB.start("watch", cmd)
            return self._json({"ok": ok, "error": None if ok else "the fly is busy with another job"}, 200 if ok else 409)
        if url.path == "/music":
            folder = Path(str(body.get("folder") or "songs").strip().strip('"'))
            if not folder.is_absolute():                     # relative to this project, not the server's cwd
                folder = HERE / folder
            if not body.get("synthetic") and not folder.is_dir():
                return self._json({"ok": False, "error": f"no such folder: {folder}"}, 400)
            cmd = [sys.executable, "-u", "-W", "ignore", "run.py", str(folder)]
            if body.get("synthetic"):
                cmd = [sys.executable, "-u", "-W", "ignore", "run.py", "--synthetic"]
            if body.get("quick"):
                cmd.append("--quick")
            if body.get("same_codec") and not body.get("synthetic"):
                cmd.append("--same-codec")
            ok = JOB.start("music", cmd)
            return self._json({"ok": ok, "error": None if ok else "the fly is busy with another job"}, 200 if ok else 409)
        if url.path == "/stop":
            if JOB.proc and JOB.state == "running":
                JOB.stopping = True
                JOB.proc.kill()
                JOB.lines.append("stopped")
            return self._json({"ok": True})
        self.send_error(404)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{PORT}/"
    print(f"fly-ears: {url}", flush=True)
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
