"""Middle-out MinT reconciliation.

Base forecasts for every node of the store x department hierarchy come from an aggregate model that sees
recent sales. MinT-shrink makes them coherent, and item forecasts are then scaled so they add up exactly to
the reconciled store x department totals.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Hierarchy:
    labels: list[str]  # one per node, bottom nodes last
    summing: np.ndarray  # S, shape (n_nodes, n_bottom)
    item_to_bottom: np.ndarray  # bottom index per item series
    bottom_labels: list[str]

    @property
    def n_bottom(self) -> int:
        return self.summing.shape[1]


def build_hierarchy(ids: pl.DataFrame) -> Hierarchy:
    bottom_key = ids.select(pl.concat_str(["store_id", "dept_id"], separator="/")).to_series()
    bottom_labels = sorted(bottom_key.unique().to_list())
    index = {label: i for i, label in enumerate(bottom_labels)}
    item_to_bottom = np.array([index[k] for k in bottom_key.to_list()])

    stores = sorted({b.split("/")[0] for b in bottom_labels})
    depts = sorted({b.split("/")[1] for b in bottom_labels})
    cats = sorted({d.rsplit("_", 1)[0] for d in depts})

    rows: list[tuple[str, np.ndarray]] = []

    def add(label: str, predicate) -> None:
        rows.append((label, np.array([predicate(b) for b in bottom_labels], dtype=np.float64)))

    add("total", lambda b: True)
    for s in stores:
        add(f"store:{s}", lambda b, s=s: b.startswith(s + "/"))
    for c in cats:
        add(f"cat:{c}", lambda b, c=c: b.split("/")[1].rsplit("_", 1)[0] == c)
    for d in depts:
        add(f"dept:{d}", lambda b, d=d: b.split("/")[1] == d)
    for s in stores:
        for c in cats:
            add(
                f"store_cat:{s}/{c}",
                lambda b, s=s, c=c: b.startswith(s + "/") and b.split("/")[1].rsplit("_", 1)[0] == c,
            )
    for i, b in enumerate(bottom_labels):
        rows.append((f"bottom:{b}", np.eye(len(bottom_labels))[i]))

    return Hierarchy(
        labels=[r[0] for r in rows],
        summing=np.vstack([r[1] for r in rows]),
        item_to_bottom=item_to_bottom,
        bottom_labels=bottom_labels,
    )


def bottom_series(values: np.ndarray, hierarchy: Hierarchy) -> np.ndarray:
    """Sum item-level rows (n_items, T) into bottom nodes (n_bottom, T)."""
    out = np.zeros((hierarchy.n_bottom, values.shape[1]), dtype=np.float64)
    np.add.at(out, hierarchy.item_to_bottom, values)
    return out


def shrink_covariance(residuals: np.ndarray) -> np.ndarray:
    """Schaefer-Strimmer shrinkage of the residual covariance towards its diagonal."""
    r = residuals - residuals.mean(axis=1, keepdims=True)
    t = r.shape[1]
    cov = r @ r.T / t
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(std, std)
    std_r = r / std[:, None]
    w = np.einsum("it,jt->ijt", std_r, std_r)
    var_corr = t / (t - 1) ** 3 * ((w - w.mean(axis=2, keepdims=True)) ** 2).sum(axis=2)
    off = ~np.eye(len(corr), dtype=bool)
    lam = float(np.clip(var_corr[off].sum() / (corr[off] ** 2).sum(), 0, 1))
    shrunk = corr * (1 - lam)
    np.fill_diagonal(shrunk, 1.0)
    return shrunk * np.outer(std, std)


def mint(base: np.ndarray, residuals: np.ndarray, hierarchy: Hierarchy) -> np.ndarray:
    """Reconciled bottom forecasts (n_bottom, h) from base forecasts for all nodes (n_nodes, h)."""
    s = hierarchy.summing
    w_inv = np.linalg.pinv(shrink_covariance(residuals))
    projection = np.linalg.solve(s.T @ w_inv @ s, s.T @ w_inv)
    return projection @ base


def allocate_to_items(item_forecast: np.ndarray, bottom_target: np.ndarray, hierarchy: Hierarchy) -> np.ndarray:
    """Scale item forecasts so each bottom node's items sum to its reconciled total, day by day."""
    current = bottom_series(item_forecast, hierarchy)
    scale = np.divide(bottom_target, current, out=np.ones_like(bottom_target), where=current > 1e-9)
    return (item_forecast * np.clip(scale, 0, None)[hierarchy.item_to_bottom]).astype(np.float32)
