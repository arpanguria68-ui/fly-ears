"""Can a feature set tell the two classes apart, and is that more than chance?

The unit is the SONG, not the clip: clips of one song are never split between training and
testing, a song's score is the mean over its clips, and every statistic counts songs.
  - stratified group k-fold over songs, repeated with different splits
  - L2 logistic regression, fixed strength, standardised inside each training fold only
    (nothing is tuned on the test songs)
  - AUC over songs
  - null: shuffle the labels between songs and rerun the whole cross-validation
  - 95% CI: bootstrap over songs; two feature sets are compared on the same resamples (paired)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

L2 = 1.0                        # fixed before seeing any data


def fit_logistic(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    n, d = X.shape

    def loss(wb):
        w, b = wb[:d], wb[d]
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        ll = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * L2 * w @ w / n
        g = X.T @ (p - y) / n + L2 * w / n
        return ll, np.r_[g, np.mean(p - y)]

    r = minimize(loss, np.zeros(d + 1), jac=True, method="L-BFGS-B")
    return r.x[:d], r.x[d]


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC (ties count half)."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.r_[pos, neg], kind="mergesort")
    ranks = np.empty(len(order))
    vals = np.r_[pos, neg][order]
    i = 0
    while i < len(vals):                                  # average ranks for ties
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def song_folds(song_labels: np.ndarray, k: int, rng) -> np.ndarray:
    """Fold number per song, stratified by class."""
    fold = np.empty(len(song_labels), int)
    for c in np.unique(song_labels):
        idx = rng.permutation(np.flatnonzero(song_labels == c))
        fold[idx] = np.arange(len(idx)) % k
    return fold


def cv_scores(X: np.ndarray, clip_song: np.ndarray, song_labels: np.ndarray, k: int = 5, repeats: int = 5,
              seed: int = 0) -> np.ndarray:
    """Out-of-fold score per song, averaged over repeated splits."""
    n_songs = len(song_labels)
    k = min(k, int(np.bincount(song_labels).min()))
    total = np.zeros(n_songs)
    rng = np.random.default_rng(seed)
    y_clip = song_labels[clip_song]
    for _ in range(repeats):
        fold = song_folds(song_labels, k, rng)
        for f in range(k):
            test_song = fold == f
            tr, te = ~test_song[clip_song], test_song[clip_song]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            w, b = fit_logistic((X[tr] - mu) / sd, y_clip[tr])
            z = ((X[te] - mu) / sd) @ w + b
            songs = clip_song[te]
            s = np.bincount(songs, weights=z, minlength=n_songs) / np.maximum(np.bincount(songs, minlength=n_songs), 1)
            total[test_song] += s[test_song]
    return total / repeats


def assess(X, clip_song, song_labels, perms: int = 200, seed: int = 0) -> dict:
    scores = cv_scores(X, clip_song, song_labels, seed=seed)
    obs = auc(scores, song_labels)
    rng = np.random.default_rng(seed + 1)
    null = np.array([auc(cv_scores(X, clip_song, (lab := rng.permutation(song_labels)), repeats=1, seed=seed + 2 + i),
                         lab) for i in range(perms)])
    return {"auc": obs, "p": float((1 + np.sum(null >= obs)) / (1 + perms)),
            "null_mean": float(null.mean()), "null_95": float(np.quantile(null, 0.95)), "scores": scores}


def bootstrap(scores: dict[str, np.ndarray], labels: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """AUC 95% CI per feature set, and paired differences, on the same song resamples."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    draws = {k: [] for k in scores}
    for _ in range(n):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        for k, s in scores.items():
            draws[k].append(auc(s[idx], labels[idx]))
    draws = {k: np.array(v) for k, v in draws.items()}
    ci = {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in draws.items()}
    return {"ci": ci, "draws": draws}


def paired(draws: dict, a: str, b: str) -> dict:
    d = draws[a] - draws[b]
    return {"diff": float(np.mean(d)), "ci": (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))),
            "p_a_not_better": float(np.mean(d <= 0))}
