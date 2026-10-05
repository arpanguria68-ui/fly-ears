"""fly-ears tests: the method's parts, without the brain (fast, no GPU, no audio files).

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyears import audio, ear, evaluate, features  # noqa: E402


# ---------------------------------------------------------------- statistics
def test_auc_matches_known_cases():
    y = np.array([0, 0, 1, 1])
    assert evaluate.auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert evaluate.auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert evaluate.auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5          # ties count half


def test_folds_are_stratified_by_song():
    labels = np.array([0] * 10 + [1] * 10)
    fold = evaluate.song_folds(labels, 5, np.random.default_rng(0))
    for f in range(5):
        assert np.bincount(labels[fold == f], minlength=2).tolist() == [2, 2]


def _songs(n_per_class=12, clips=3):
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    clip_song = np.repeat(np.arange(len(labels)), clips)
    return labels, clip_song


def test_clear_signal_is_found():
    rng = np.random.default_rng(0)
    labels, clip_song = _songs()
    X = rng.normal(size=(len(clip_song), 5))
    X[:, 0] += 2.0 * labels[clip_song]
    r = evaluate.assess(X, clip_song, labels, perms=50, seed=0)
    assert r["auc"] > 0.9 and r["p"] < 0.05


def test_no_signal_gives_chance():
    rng = np.random.default_rng(1)
    labels, clip_song = _songs()
    aucs = [evaluate.auc(evaluate.cv_scores(rng.normal(size=(len(clip_song), 5)), clip_song, labels, seed=s), labels)
            for s in range(20)]
    assert abs(np.mean(aucs) - 0.5) < 0.08


def test_song_fingerprints_do_not_leak_across_folds():
    """Each song has its own random fingerprint shared by its clips, and labels are random. If clips of a
    song reached both training and test, the model could learn the fingerprint -> label and score high."""
    rng = np.random.default_rng(2)
    aucs = []
    for s in range(20):
        labels, clip_song = _songs()
        labels = rng.permutation(labels)
        finger = rng.normal(size=(len(labels), 20))
        X = finger[clip_song] + 0.01 * rng.normal(size=(len(clip_song), 20))
        aucs.append(evaluate.auc(evaluate.cv_scores(X, clip_song, labels, seed=s), labels))
    assert np.mean(aucs) < 0.62


def test_bootstrap_paired_difference_sign():
    labels = np.array([0] * 15 + [1] * 15)
    good = labels + np.random.default_rng(3).normal(0, 0.3, 30)
    bad = np.random.default_rng(4).normal(size=30)
    b = evaluate.bootstrap({"good": good, "bad": bad}, labels, n=500)
    assert evaluate.paired(b["draws"], "good", "bad")["ci"][0] > 0


# ---------------------------------------------------------------- audio and ear
def test_clips_fixed_positions_and_level():
    x = np.random.default_rng(0).normal(size=audio.EAR_SR * 30).astype(np.float32) * 3
    cs = audio.clips(x, audio.EAR_SR, 8, 3)
    assert len(cs) == 3 and all(len(c) == 8 * audio.EAR_SR for c in cs)
    assert audio.clips(x[: audio.EAR_SR * 5], audio.EAR_SR, 8, 3) == []
    assert abs(np.sqrt(np.mean(audio.level(cs[0]) ** 2)) - audio.TARGET_RMS) < 1e-4


def _tone(f, seconds=2.0):
    t = np.arange(int(seconds * audio.EAR_SR)) / audio.EAR_SR
    return audio.level(np.sin(2 * np.pi * f * t).astype(np.float32))


def test_tones_land_in_their_band():
    low, high = ear.envelopes(_tone(150)).mean(0), ear.envelopes(_tone(900)).mean(0)
    assert np.argmax(low) in ear.LOW_BANDS and np.argmax(high) in ear.HIGH_BANDS
    assert 0 < low.max() <= 1 and 0 < high.max() <= 1


def test_out_of_range_sound_is_not_heard():
    assert ear.envelopes(_tone(3000)).max() < 0.15        # 3 kHz: above the fly's ear here


def _fake_brain():
    types = np.array(["JO-A"] * 6 + ["JO-B1_a"] * 4 + ["JO-B2"] * 4 + ["JO-CM"] * 3 + ["WED001"] * 2)
    return SimpleNamespace(cell_type=types, cells=lambda ts: np.flatnonzero(np.isin(types, ts)))


def test_ear_map_tonotopic_and_flat():
    cells, bands = ear.ear_map(_fake_brain())
    types = _fake_brain().cell_type[cells]
    assert len(cells) == 14                                # JO-A and JO-B only
    assert all(b in ear.LOW_BANDS for b, t in zip(bands, types) if t.startswith("JO-B"))
    assert all(b in ear.HIGH_BANDS for b, t in zip(bands, types) if t.startswith("JO-A"))
    env = np.random.default_rng(0).random((ear.N_BANDS, 3))
    assert ear.drive(env, bands).shape == (14, 3)
    _, flat = ear.ear_map(_fake_brain(), "flat")
    d = ear.drive(env, flat)
    assert np.allclose(d, env.mean(0) * ear.EAR_CAP)


def test_feature_shapes():
    series = np.random.default_rng(0).random((400, 7))
    assert features.response_stats(series).shape == (28,)
    x = np.random.default_rng(1).normal(size=audio.FULL_SR * 4).astype(np.float32) * 0.1
    assert np.all(np.isfinite(features.audio_stats(x)))


def test_injections_match_per_neuron_drive():
    cells, bands = ear.ear_map(_fake_brain())
    env = np.random.default_rng(5).random((ear.N_BANDS, 3))
    full = dict(zip(cells.tolist(), ear.drive(env, bands)))
    for idx, amount in ear.injections(cells, bands, env):
        for c in idx:
            assert np.allclose(full[int(c)], amount)


# ---------------------------------------------------------------- vision
from flyears import vision  # noqa: E402


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    big = ndimage_zoom(rng.random((30, 54)), 6)
    return (big[: vision.FLOW_H + 20, : vision.FLOW_W + 20] * 255).astype(np.uint8)


def ndimage_zoom(a, k):
    from scipy import ndimage
    return ndimage.zoom(a, k, order=1)


def test_flow_finds_rightward_motion():
    tex = _texture()
    a = tex[10:10 + vision.FLOW_H, 10:10 + vision.FLOW_W]
    b = tex[10:10 + vision.FLOW_H, 8:8 + vision.FLOW_W]              # content moved 2 px right
    u, v, _ = vision.flow(a, b)
    mid = (slice(15, -15), slice(15, -15))
    assert 1.0 < np.median(u[mid]) < 3.0 and abs(np.median(v[mid])) < 0.5


def _fake_eye_brain():
    types = np.array(["LPLC2"] * 4 + ["R1-6"] * 6)
    side = np.array(["L", "L", "R", "R"] + ["L"] * 6)
    def cells(ts, side_=None, **kw):
        s = kw.get("side", side_)
        m = np.isin(types, ts)
        return np.flatnonzero(m & (side == s) if s else m)
    return SimpleNamespace(cell_type=types, azimuth=np.linspace(-1, 1, 6), cells=cells)


def _cols():
    # 4 cells: left/right eye x (front-to-back preferring "backward", back-to-front preferring "forward")
    return {"cells": np.array([100, 101, 102, 103]), "eye": np.array(["L", "L", "R", "R"]),
            "front": np.full(4, 0.5, np.float32), "up": np.full(4, 0.5, np.float32),
            "pref": np.array([[-1, 0], [1, 0], [-1, 0], [1, 0]], np.float32),
            "types": np.array(["T4a", "T4b", "T4a", "T4b"])}


def test_rightward_motion_drives_the_right_cells():
    """Moving right: the left eye sees back-to-front (forward) motion, the right eye front-to-back."""
    eyes = vision.Eyes(_fake_eye_brain(), _cols())
    tex = _texture(1)
    eyes.see(tex[10:10 + vision.FLOW_H, 10:10 + vision.FLOW_W])
    eyes.see(tex[10:10 + vision.FLOW_H, 8:8 + vision.FLOW_W])
    d = eyes.last["motion"]
    assert d[1] > d[0]          # left eye: forward-preferring cell
    assert d[2] > d[3]          # right eye: backward-preferring cell


def test_photoreceptors_follow_brightness():
    eyes = vision.Eyes(_fake_eye_brain(), _cols())
    frame = np.zeros((vision.FLOW_H, vision.FLOW_W), np.uint8)
    frame[:, : vision.FLOW_W // 2] = 255                              # bright on the left
    p = eyes.photoreceptors(frame)
    assert p[0] > 0.9 and p[-1] < 0.1


def test_level_sets_match_drive():
    drive = np.array([0.0, 0.1, 0.4, 0.8, 0.8], np.float32)
    cells = np.arange(5)
    got = np.zeros(5)
    for idx, amt in vision.level_sets(cells, drive):
        got[idx] = amt
    assert np.max(np.abs(got - drive)) <= vision.CAP / (2 * vision.LEVELS) + 1e-6


def test_looming_needs_expansion_not_sliding():
    H, W = vision.FLOW_H, vision.FLOW_W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = 36                                                            # an LPLC2 receptive-field centre
    expand_u, expand_v = (xx - cx) * 0.05, (yy - H / 2) * 0.05          # everything flows away from a point
    slide_u, slide_v = np.full((H, W), 1.5, np.float32), np.zeros((H, W), np.float32)
    assert vision.looming(expand_u, expand_v, "L") > 0.3
    assert vision.looming(slide_u, slide_v, "L") == 0.0
