"""Baselines every model must beat."""

import numpy as np


def seasonal_naive(history: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the last observed week."""
    last = history[:, -season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last, reps)[:, :horizon]


def tsb(history: np.ndarray, horizon: int, alpha_demand: float = 0.1, alpha_prob: float = 0.1) -> np.ndarray:
    """Teunter-Syntetos-Babai for intermittent demand, vectorised across series."""
    first = history[:, 0]
    prob = (first > 0).astype(np.float64)
    size = np.where(first > 0, first, np.nan)
    for t in range(1, history.shape[1]):
        y = history[:, t]
        occurred = y > 0
        prob = prob + alpha_prob * (occurred - prob)
        size = np.where(
            occurred,
            np.where(np.isnan(size), y, size + alpha_demand * (y - size)),
            size,
        )
    level = np.nan_to_num(prob * size)
    return np.repeat(level[:, None], horizon, axis=1)
