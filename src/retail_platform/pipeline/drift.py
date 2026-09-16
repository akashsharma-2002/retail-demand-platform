"""Feature and residual drift report using the Population Stability Index (PSI).

PSI < 0.1: stable, 0.1-0.25: moderate shift, > 0.25: significant shift. Drift is reported, not auto-acted on.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

DRIFT_FEATURES = ["roll_mean_28", "roll_std_28", "lag_28", "sell_price", "price_rel_52w", "snap"]
BINS = 10


def psi(reference: np.ndarray, current: np.ndarray, bins: int = BINS) -> float:
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: compare category shares directly
        values = np.union1d(np.unique(reference), np.unique(current))
        ref = np.array([(reference == v).mean() for v in values])
        cur = np.array([(current == v).mean() for v in values])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        ref = np.histogram(reference, edges)[0] / len(reference)
        cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def status(value: float) -> str:
    if np.isnan(value):
        return "insufficient_data"
    return "stable" if value < 0.1 else "moderate" if value < 0.25 else "significant"


def drift_report(
    features: pl.DataFrame, reference: tuple[int, int], current: tuple[int, int], sample: int = 400_000, seed: int = 3
) -> dict:
    """reference/current are inclusive day ranges (d)."""

    def window(lo: int, hi: int) -> pl.DataFrame:
        rows = features.filter((pl.col("d") >= lo) & (pl.col("d") <= hi) & pl.col("sales").is_not_null())
        return rows.sample(n=min(sample, rows.height), seed=seed)

    ref, cur = window(*reference), window(*current)
    per_feature: dict[str, dict[str, float | str]] = {}
    for col in DRIFT_FEATURES:
        value = psi(ref[col].to_numpy().astype(float), cur[col].to_numpy().astype(float))
        per_feature[col] = {"psi": round(value, 4), "status": status(value)}
    return {
        "reference_days": list(reference),
        "current_days": list(current),
        "features": per_feature,
        "any_significant": any(f["status"] == "significant" for f in per_feature.values()),
    }


def write(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    from retail_platform.config import get_settings
    from retail_platform.forecast.context import ForecastContext

    settings = get_settings()
    ctx = ForecastContext(settings)
    last = settings.last_day
    result = drift_report(ctx.features, (ctx.train_start_d, last - 28), (last - 27, last))
    write(result, settings.artifacts_dir / "reports" / "drift.json")
    print(json.dumps(result, indent=2))
