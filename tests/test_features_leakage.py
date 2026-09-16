from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from retail_platform.features.build import MIN_LAG, build_features, feature_columns


def synthetic_long(n_series: int = 3, n_days: int = 500, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_series):
        for d in range(1, n_days + 1):
            rows.append(
                {
                    "series_idx": s,
                    "d": d,
                    "sales": float(rng.poisson(3 + s)),
                    "item_id": f"FOODS_3_{s:03d}",
                    "dept_id": "FOODS_3",
                    "cat_id": "FOODS",
                    "store_id": "CA_1",
                    "date": date(2014, 1, 1) + timedelta(days=d - 1),
                    "wm_yr_wk": d // 7,
                    "wday": (d % 7) + 1,
                    "month": 1,
                    "year": 2014,
                    "event_type_1": None if d % 50 else "Sporting",
                    "snap": int(d % 10 < 3),
                    "sell_price": 2.0 + (d // 90) * 0.1 + s,
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("series_idx").cast(pl.Int32),
        pl.col("d").cast(pl.Int16),
        pl.col("sales").cast(pl.Float32),
        *[pl.col(c).cast(pl.Categorical) for c in ["item_id", "dept_id", "cat_id", "store_id"]],
    )


@pytest.mark.parametrize("min_lag", [14, 28])
@pytest.mark.parametrize("cutoff", [380, 420, 460])
def test_features_within_min_lag_do_not_use_sales_after_cutoff(cutoff, min_lag):
    full = synthetic_long()
    truncated = full.with_columns(pl.when(pl.col("d") > cutoff).then(None).otherwise(pl.col("sales")).alias("sales"))
    window = (pl.col("d") > cutoff) & (pl.col("d") <= cutoff + min_lag)
    cols = feature_columns(min_lag)
    a = build_features(full, min_lag).filter(window).select(cols).to_numpy()
    b = build_features(truncated, min_lag).filter(window).select(cols).to_numpy()
    np.testing.assert_allclose(a, b, equal_nan=True)


def test_feature_29_days_ahead_would_change_so_the_test_can_fail():
    full = synthetic_long()
    cutoff = 400
    truncated = full.with_columns(pl.when(pl.col("d") > cutoff).then(None).otherwise(pl.col("sales")).alias("sales"))
    day = pl.col("d") == cutoff + MIN_LAG + 1
    a = build_features(full).filter(day)["lag_28"].to_numpy()
    b = build_features(truncated).filter(day)["lag_28"].to_numpy()
    assert not np.allclose(a, b, equal_nan=True)
