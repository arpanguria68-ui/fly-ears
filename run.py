"""fly-ears: does a real fly brain respond differently to human-made and AI-generated music?

    python run.py songs                      # songs/human/*.mp3 and songs/ai/*.mp3 (your own files)
    python run.py songs --same-codec         # first push every file through one MP3 codec
    python run.py --synthetic                # check the method on synthetic songs (null + signal)
    python run.py songs --quick              # no rewired brain, fewer permutations (a first look)

Writes out/<name>/report.md and results.json. See README.md for the questions, decided in advance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from flyears import audio, ear, evaluate, features, listen, synth

HERE = Path(__file__).resolve().parent
CLASSES = ("human", "ai")                   # label 0, label 1


def log(*a):
    print(*a, flush=True)


def load(folder: Path, seconds: float, per_song: int, same_codec: bool, seed: int):
    found = audio.songs_in(folder)
    missing = [c for c in CLASSES if c not in found]
    if missing:
        raise SystemExit(f"{folder} needs subfolders {CLASSES} with audio files (missing: {missing})")
    rng = np.random.default_rng(seed)
    n = min(len(found[c]) for c in CLASSES)
    used = {c: sorted(rng.choice(found[c], n, replace=False).tolist()) for c in CLASSES}   # balance classes
    dropped = {c: len(found[c]) - n for c in CLASSES}
    songs, labels, clip_song, ear_clips, full_clips = [], [], [], [], []
    for lab, c in enumerate(CLASSES):
        for f in used[c]:
            x8 = audio.decode(f, audio.EAR_SR, same_codec)
            x22 = audio.decode(f, audio.FULL_SR, same_codec)
            a = audio.clips(x8, audio.EAR_SR, seconds, per_song)
            b = audio.clips(x22, audio.FULL_SR, seconds, per_song)
            if not a:
                log(f"  skipped (shorter than {seconds} s): {f.name}")
                continue
            sid = len(songs)
            songs.append(str(f))
            labels.append(lab)
            for c8, c22 in zip(a, b):
                ear_clips.append(audio.level(c8))
                full_clips.append(audio.level(c22))
                clip_song.append(sid)
    return songs, np.array(labels), np.array(clip_song), ear_clips, full_clips, dropped


def analyse(folder: Path, out: Path, args) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    log(f"\n== {folder}")
    songs, labels, clip_song, ear_clips, full_clips, dropped = load(folder, args.seconds, args.clips, args.same_codec,
                                                                      args.seed)
    if np.bincount(labels, minlength=2).min() < 4:
        raise SystemExit("need at least 4 songs (long enough) in each class")
    log(f"{len(songs)} songs ({np.sum(labels == 0)} human, {np.sum(labels == 1)} ai), {len(clip_song)} clips; "
        f"dropped to balance: {dropped}")
    envs = [ear.envelopes(x) for x in ear_clips]
    sets = {"audio (full band)": np.stack([features.audio_stats(x) for x in full_clips]),
            "ear input": np.stack([features.response_stats(e) for e in envs])}

    key = hashlib.sha1(json.dumps([songs, args.seconds, args.clips, args.same_codec]).encode()).hexdigest()[:10]
    brains = [("brain (real)", False, "tonotopic")]
    if not args.quick:
        brains += [("brain (rewired)", True, "tonotopic"), ("brain (flat ear)", False, "flat")]
    responses = {}
    for name, rewired, mode in brains:
        cache = out / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}_{key}.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            rates, names = d["rates"], list(d["names"])
        else:
            log(f"{name}: playing {len(envs)} clips + silence")
            brain = listen.make_brain(args.batch, args.device, seed=args.seed, rewired=rewired)
            silent = np.zeros_like(envs[0])
            rates, names = listen.play(brain, envs + [silent], mode=mode, seed=args.seed, log=log)
            np.savez_compressed(cache, rates=rates, names=np.array(names))
            del brain
        responses[name] = (rates, names)
        sets[name] = np.stack([features.response_stats(r) for r in rates[:-1]])

    results = {}
    for name, X in sets.items():
        t0 = time.perf_counter()
        r = evaluate.assess(X, clip_song, labels, perms=args.perms, seed=args.seed)
        results[name] = r
        log(f"  {name:20s} AUC {r['auc']:.3f}  null {r['null_mean']:.3f} (95th {r['null_95']:.3f})  p {r['p']:.4f}"
            f"  [{time.perf_counter() - t0:.0f} s]")
    boot = evaluate.bootstrap({k: v["scores"] for k, v in results.items()}, labels, seed=args.seed)
    comps = {}
    for a, b in (("brain (real)", "ear input"), ("brain (real)", "brain (rewired)"), ("brain (real)", "brain (flat ear)"),
                 ("audio (full band)", "brain (real)")):
        if a in results and b in results:
            comps[f"{a} vs {b}"] = evaluate.paired(boot["draws"], a, b)
    response = describe(responses["brain (real)"], clip_song, labels)
    report = write_report(out, folder, songs, labels, dropped, results, boot["ci"], comps, response, args)
    return report


def describe(resp, clip_song, labels) -> list[dict]:
    """How the brain responds: each group's rate during music vs silence, and human vs ai (per song)."""
    rates, names = resp
    clip_mean = rates[:-1].mean(1)                     # (clips, groups)
    silence = rates[-1].mean(0)
    n_songs = len(labels)
    song_mean = np.stack([clip_mean[clip_song == s].mean(0) for s in range(n_songs)])
    rows = []
    for j, g in enumerate(names):
        h, a = song_mean[labels == 0, j], song_mean[labels == 1, j]
        sd = np.sqrt((h.var(ddof=1) + a.var(ddof=1)) / 2) + 1e-9
        p = stats.ttest_ind(h, a, equal_var=False).pvalue if sd > 1e-6 else 1.0
        rows.append({"group": g, "silence": float(silence[j]), "music": float(clip_mean[:, j].mean()),
                     "human": float(h.mean()), "ai": float(a.mean()), "d": float((a.mean() - h.mean()) / sd),
                     "p": float(p)})
    order = np.argsort([r["p"] for r in rows])                         # Holm correction over groups
    m = len(rows)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[i]["p"]))
        rows[i]["p_holm"] = running
    return rows


def verdict(ok: bool) -> str:
    return "**yes**" if ok else "no"


def write_report(out, folder, songs, labels, dropped, results, ci, comps, response, args) -> dict:
    R = {k: {kk: vv for kk, vv in v.items() if kk != "scores"} | {"ci": ci[k]} for k, v in results.items()}
    lines = [f"# fly-ears report: {folder.name}", "",
             f"{len(songs)} songs ({int(np.sum(labels == 0))} human, {int(np.sum(labels == 1))} ai), "
             f"{args.clips} clips of {args.seconds:g} s per song, same codec: {args.same_codec}, "
             f"dropped to balance: {dropped}. Unit of analysis: the song.", "",
             "| features | AUC | 95% CI | null mean | null 95th | p (permutation) |", "|---|---|---|---|---|---|"]
    for k, r in R.items():
        lines.append(f"| {k} | {r['auc']:.3f} | {r['ci'][0]:.2f}-{r['ci'][1]:.2f} | {r['null_mean']:.3f} | "
                     f"{r['null_95']:.3f} | {r['p']:.4f} |")
    lines += ["", "| comparison (paired bootstrap over songs) | AUC difference | 95% CI |", "|---|---|---|"]
    for k, c in comps.items():
        lines.append(f"| {k} | {c['diff']:+.3f} | {c['ci'][0]:+.3f} to {c['ci'][1]:+.3f} |")
    br = R.get("brain (real)")
    q = ["", "## The questions (decided before running)", ""]
    q.append(f"1. Plain audio separates the classes (reference): {verdict(R['audio (full band)']['p'] < 0.05)}")
    q.append(f"2. The fly brain's response separates them above chance: {verdict(br is not None and br['p'] < 0.05)}")
    c = comps.get("brain (real) vs ear input")
    q.append(f"3. The brain adds something beyond what its ear received: {verdict(bool(c and c['ci'][0] > 0))}")
    c = comps.get("brain (real) vs brain (rewired)")
    q.append(f"4. The real wiring matters (real beats rewired): " + (verdict(c['ci'][0] > 0) if c else "not run (--quick)"))
    lines += q
    lines += ["", "## How the brain responds", "",
              "Spikes per neuron per second, real brain. *music* is the mean over all clips; human/ai are means over "
              "songs; d is (ai - human) / pooled SD; p is Welch's t over songs, Holm-corrected across groups.", "",
              "| group | silence | music | human | ai | d | p (Holm) |", "|---|---|---|---|---|---|---|"]
    for r in sorted(response, key=lambda r: r["p_holm"]):
        lines.append(f"| {r['group']} | {r['silence']:.3f} | {r['music']:.3f} | {r['human']:.3f} | {r['ai']:.3f} | "
                     f"{r['d']:+.2f} | {r['p_holm']:.3f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"songs": songs, "labels": labels.tolist(), "results": R, "comparisons": comps, "response": response,
              "scores": {k: v["scores"].tolist() for k, v in results.items()}}
    (out / "results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log("\n" + "\n".join(q[2:]))
    log(f"report: {out / 'report.md'}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("songs", nargs="?", help="folder with human/ and ai/ subfolders of audio files")
    p.add_argument("--synthetic", action="store_true", help="run the null and signal synthetic checks")
    p.add_argument("--same-codec", action="store_true", help="push every file through one MP3 codec first")
    p.add_argument("--seconds", type=float, default=8.0, help="clip length")
    p.add_argument("--clips", type=int, default=3, help="clips per song (fixed positions)")
    p.add_argument("--perms", type=int, default=200, help="label permutations for the null")
    p.add_argument("--quick", action="store_true", help="real brain only, 50 permutations")
    p.add_argument("--batch", type=int, default=32, help="clips played at once (flies per batch)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    if args.quick:
        args.perms = min(args.perms, 50)
    if args.synthetic:
        base = HERE / "out" / "synthetic"
        for kind in ("null", "signal"):
            folder = synth.make(base / "songs", kind, per_class=12, seed=1 if kind == "null" else 2)
            analyse(folder, base / kind, args)
        return
    if not args.songs:
        p.error("give a songs folder, or --synthetic")
    folder = Path(args.songs).resolve()
    analyse(folder, HERE / "out" / folder.name, args)


if __name__ == "__main__":
    main()
