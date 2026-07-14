"""Fast connectome-constrained rate model + the shuffled-wiring null.

The scientific core of Enactome: turn a weight matrix into a simulatable rate model,
test what it computes with a cross-validated linear decoder, and — critically —
compare every decodable structure against a degree/weight-preserving shuffle null,
because sparse convergence alone produces apparent selectivity by chance.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def divisive_norm(x: np.ndarray, m: float = 0.05, sigma: float = 0.1) -> np.ndarray:
    """Olsen-Wilson gain control: per-stimulus divisive normalization."""
    denom = sigma + m * x.sum(axis=-1, keepdims=True)
    return x / denom


def rate_forward(x: np.ndarray, W: np.ndarray, normalize: bool = True) -> np.ndarray:
    """One feedforward layer: (divisive norm) -> linear mix by W -> ReLU."""
    if normalize:
        x = divisive_norm(x)
    y = x @ W
    return np.maximum(y, 0.0)


def participation_ratio(X: np.ndarray) -> float:
    """Representational dimensionality: (Σλ)² / Σλ². Lower = more compressed."""
    X = X - X.mean(0, keepdims=True)
    C = np.cov(X, rowvar=False)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 1e-12]
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if len(ev) else 0.0


def decode_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> tuple[float, float]:
    """Cross-validated balanced accuracy of a linear readout. Returns (mean, std)."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=min(n_splits, np.bincount(y).min()), shuffle=True, random_state=seed)
    s = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
    return float(s.mean()), float(s.std())


def shuffle_null(W: np.ndarray, Xin: np.ndarray, y: np.ndarray, forward_fn,
                 n: int = 200, seed: int = 0) -> dict:
    """Degree/weight-preserving column-shuffle null for a decoding score.

    Returns observed score, null distribution, z, and one-sided p (obs >= null).
    """
    rng = np.random.default_rng(seed)
    obs, _ = decode_cv(forward_fn(Xin, W), y)
    null = np.empty(n)
    for i in range(n):
        perm = rng.permutation(W.shape[0])
        null[i], _ = decode_cv(forward_fn(Xin, W[perm]), y)
    z = (obs - null.mean()) / (null.std() + 1e-9)
    p = float((null >= obs).mean())
    return {"observed": float(obs), "null_mean": float(null.mean()),
            "null_std": float(null.std()), "z": float(z), "p": p, "null": null.tolist()}
