"""Leakage-safe features.

Every feature derived from sales is shifted by at least `min_lag` days, so a model built with min_lag = m can
forecast days 1..m after a cut-off using only sales observed up to that cut-off. The platform uses two models:
min_lag 14 for days 1-14 (sees more recent sales) and min_lag 28 for days 15-28.
"""

import polars as pl

MIN_LAG = 28
ROLL_WINDOWS = [7, 28, 56]
CATEGORICAL = ["item_code", "dept_code", "cat_code", "store_code", "event_code"]
STATIC_COLS = [
    "sell_price",
    "price_rel_52w",
    "price_change_7d",
    "price_rel_dept",
    "wday",
    "month",
    "mday",
    "week",
    "snap",
    *CATEGORICAL,
]


def sales_lags(min_lag: int) -> list[int]:
    return [min_lag, min_lag + 7, min_lag + 14, min_lag + 21, min_lag + 28, 364]


def feature_columns(min_lag: int = MIN_LAG) -> list[str]:
    return [
        *[f"lag_{lag}" for lag in sales_lags(min_lag)],
        *[f"roll_mean_{w}" for w in ROLL_WINDOWS],
        *[f"roll_std_{w}" for w in ROLL_WINDOWS],
        "dow_mean_4w",
        "item_roll_mean_28",
        "store_dept_roll_mean_28",
        *STATIC_COLS,
    ]


FEATURE_COLS = feature_columns(MIN_LAG)


def _dense_code(col: str) -> pl.Expr:
    return (pl.col(col).cast(pl.String).rank("dense") - 1).cast(pl.Int32)


def build_features(long: pl.DataFrame, min_lag: int = MIN_LAG) -> pl.DataFrame:
    """`long` must be sorted by (series_idx, d) with contiguous days per series."""
    by_series = "series_idx"
    lags = sales_lags(min_lag)
    shifted = pl.col("sales").shift(min_lag)
    cols = feature_columns(min_lag)
    return (
        long.with_columns(
            *[pl.col("sales").shift(lag).over(by_series).alias(f"lag_{lag}") for lag in lags],
            *[shifted.rolling_mean(w).over(by_series).alias(f"roll_mean_{w}") for w in ROLL_WINDOWS],
            *[shifted.rolling_std(w).over(by_series).alias(f"roll_std_{w}") for w in ROLL_WINDOWS],
            (pl.col("sell_price") / pl.col("sell_price").rolling_mean(364, min_samples=1).over(by_series)).alias(
                "price_rel_52w"
            ),
            (pl.col("sell_price") / pl.col("sell_price").shift(7).over(by_series) - 1).alias("price_change_7d"),
            (pl.col("sell_price") / pl.col("sell_price").mean().over(["dept_id", "store_id", "d"])).alias(
                "price_rel_dept"
            ),
            pl.col("date").dt.day().alias("mday"),
            pl.col("date").dt.week().alias("week"),
            _dense_code("item_id").alias("item_code"),
            _dense_code("dept_id").alias("dept_code"),
            _dense_code("cat_id").alias("cat_code"),
            _dense_code("store_id").alias("store_code"),
            (pl.col("event_type_1").fill_null("none").rank("dense") - 1).cast(pl.Int32).alias("event_code"),
        )
        .with_columns(
            (pl.sum_horizontal([pl.col(f"lag_{lag}") for lag in lags[:4]]) / 4).alias("dow_mean_4w"),
            pl.col("roll_mean_28").mean().over(["item_id", "d"]).alias("item_roll_mean_28"),
            pl.col("roll_mean_28").mean().over(["store_id", "dept_id", "d"]).alias("store_dept_roll_mean_28"),
        )
        .with_columns(pl.col(c).cast(pl.Float32) for c in cols if c not in CATEGORICAL)
    )
