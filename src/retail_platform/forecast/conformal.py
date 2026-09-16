"""Mondrian split-conformal quantiles for cumulative demand.

Series are grouped by sales velocity. For each group and each window length h, the calibration set gives the
empirical quantile of (actual cumulative demand - forecast cumulative demand). Adding it to a new cumulative
forecast gives an upper bound with approximately the requested coverage.
"""

from dataclasses import dataclass

import numpy as np

N_BUCKETS = 10


def velocity(history: np.ndarray, window: int = 28) -> np.ndarray:
    return history[:, -window:].mean(axis=1)


@dataclass
class CumulativeConformal:
    edges: np.ndarray  # bucket edges on velocity
    residuals: np.ndarray  # (n_buckets, horizon, n_calibration_series_in_bucket) stored as list per bucket
    _sorted: list[np.ndarray]

    @classmethod
    def fit(cls, history: np.ndarray, forecast: np.ndarray, actual: np.ndarray) -> "CumulativeConformal":
        vel = velocity(history)
        edges = np.unique(np.quantile(vel[vel > 0], np.linspace(0, 1, N_BUCKETS + 1)[1:-1]))
        bucket = np.searchsorted(edges, vel, side="right")
        err = np.cumsum(actual, axis=1) - np.cumsum(forecast, axis=1)
        per_bucket = [np.sort(err[bucket == b], axis=0) for b in range(len(edges) + 1)]
        return cls(edges=edges, residuals=err, _sorted=per_bucket)

    def buckets(self, history: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, velocity(history), side="right")

    def quantile_offset(self, buckets: np.ndarray, window: np.ndarray, level: float) -> np.ndarray:
        """Residual quantile per series for its bucket and cumulative window length (1-based)."""
        out = np.zeros(len(buckets), dtype=np.float64)
        horizon = self._sorted[0].shape[1]
        for b in np.unique(buckets):
            errs = self._sorted[b]
            rows = buckets == b
            h = np.clip(window[rows], 1, horizon) - 1
            if len(errs) == 0:
                continue
            k = min(len(errs) - 1, int(np.ceil(level * (len(errs) + 1))) - 1)
            out[rows] = errs[k, h]
        return out

    def coverage(
        self, history: np.ndarray, forecast: np.ndarray, actual: np.ndarray, level: float, window: int
    ) -> float:
        b = self.buckets(history)
        upper = forecast[:, :window].sum(axis=1) + self.quantile_offset(b, np.full(len(b), window), level)
        return float((actual[:, :window].sum(axis=1) <= upper).mean())
