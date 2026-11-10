from __future__ import annotations

import numpy as np


def precision_recall_f1(overlap: int, candidate_size: int, target_size: int) -> tuple[float, float, float]:
    overlap = int(overlap)
    candidate_size = int(candidate_size)
    target_size = int(target_size)
    if candidate_size <= 0 or target_size <= 0 or overlap <= 0:
        return 0.0, 0.0, 0.0
    precision = overlap / candidate_size
    recall = overlap / target_size
    f1 = 2.0 * overlap / (candidate_size + target_size)
    return float(precision), float(recall), float(f1)


def f1_from_counts(overlap: np.ndarray, candidate_sizes: np.ndarray, target_size: int) -> np.ndarray:
    overlap = np.asarray(overlap, dtype=np.float64)
    candidate_sizes = np.asarray(candidate_sizes, dtype=np.float64)
    denom = candidate_sizes + float(target_size)
    out = np.zeros_like(overlap, dtype=np.float64)
    valid = (overlap > 0) & (candidate_sizes > 0) & (denom > 0)
    out[valid] = 2.0 * overlap[valid] / denom[valid]
    return out
