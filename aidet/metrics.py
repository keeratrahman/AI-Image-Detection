"""Small metric helpers (no sklearn dependency)."""
import numpy as np


def roc_auc(labels, scores) -> float:
    y = np.asarray(labels).reshape(-1) > 0.5
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    _, inverse, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0  # average 1-based rank per tie group
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = avg_rank[inverse]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
