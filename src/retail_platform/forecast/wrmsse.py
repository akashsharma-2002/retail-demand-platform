"""WRMSSE, the official M5 accuracy metric, for a single-state subset.

The 12 M5 aggregation levels are kept. With one state, the state-level levels equal their
all-state counterparts, which is exactly what the official formula gives when restricted to one state.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse

LEVELS: dict[str, list[str]] = {
    "total": [],
    "state": ["state_id"],
    "store": ["store_id"],
    "cat": ["cat_id"],
    "dept": ["dept_id"],
    "state_cat": ["state_id", "cat_id"],
    "state_dept": ["state_id", "dept_id"],
    "store_cat": ["store_id", "cat_id"],
    "store_dept": ["store_id", "dept_id"],
    "item": ["item_id"],
    "item_state": ["item_id", "state_id"],
    "item_store": ["item_id", "store_id"],
}


def aggregation_matrix(ids: pl.DataFrame, keys: list[str]) -> sparse.csr_matrix:
    n = ids.height
    if not keys:
        return sparse.csr_matrix(np.ones((1, n), dtype=np.float32))
    group = ids.select(pl.struct(keys).rank("dense").cast(pl.Int64) - 1).to_series().to_numpy()
    return sparse.csr_matrix((np.ones(n, dtype=np.float32), (group, np.arange(n))), shape=(int(group.max()) + 1, n))


def naive_scale(history: np.ndarray) -> np.ndarray:
    """Mean squared one-step naive error, measured from each series' first non-zero sale."""
    started = np.maximum.accumulate(history > 0, axis=1)
    diffs = np.diff(history, axis=1) ** 2
    valid = started[:, :-1]
    count = valid.sum(axis=1)
    scale = np.where(valid, diffs, 0).sum(axis=1) / np.maximum(count, 1)
    return np.where(scale > 0, scale, 1.0)


@dataclass
class WRMSSEEvaluator:
    ids: pl.DataFrame  # one row per series, same order as the arrays
    sales: np.ndarray  # (n_series, n_days) full observed history, column j = day j+1
    revenue: np.ndarray  # (n_series, n_days) sales * price, nan-free

    def score(self, forecast: np.ndarray, start_day: int) -> dict[str, float]:
        """forecast has shape (n_series, h) and covers days start_day .. start_day + h - 1."""
        h = forecast.shape[1]
        s = start_day - 1
        actual = self.sales[:, s : s + h]
        history = self.sales[:, :s]
        rev28 = self.revenue[:, s - 28 : s].sum(axis=1)

        out: dict[str, float] = {}
        total = 0.0
        for name, keys in LEVELS.items():
            agg = aggregation_matrix(self.ids, keys)
            a_hist = np.asarray(agg @ history)
            a_act = np.asarray(agg @ actual)
            a_fc = np.asarray(agg @ forecast)
            weights = np.asarray(agg @ rev28).ravel()
            weights = weights / weights.sum()
            rmsse = np.sqrt(((a_act - a_fc) ** 2).mean(axis=1) / naive_scale(a_hist))
            level_score = float((weights * rmsse).sum())
            out[name] = level_score
            total += level_score
        out["wrmsse"] = total / len(LEVELS)
        return out


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.abs(actual - forecast).sum() / max(actual.sum(), 1e-9))


def bias_pct(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float((forecast.sum() - actual.sum()) / max(actual.sum(), 1e-9) * 100)
