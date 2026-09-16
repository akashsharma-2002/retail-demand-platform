"""Global LightGBM model with a Tweedie objective."""

import lightgbm as lgb
import numpy as np
import polars as pl

from retail_platform.features.build import CATEGORICAL, MIN_LAG, feature_columns

DEFAULT_PARAMS: dict[str, object] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.06,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "num_threads": 8,
    "verbose": -1,
    "seed": 42,
}
NUM_ROUNDS = 700


def train(
    features: pl.DataFrame,
    train_start_d: int,
    cutoff: int,
    params: dict[str, object] | None = None,
    num_rounds: int = NUM_ROUNDS,
    min_lag: int = MIN_LAG,
) -> lgb.Booster:
    cols = feature_columns(min_lag)
    rows = features.filter((pl.col("d") >= train_start_d) & (pl.col("d") <= cutoff) & pl.col("sales").is_not_null())
    data = lgb.Dataset(
        rows.select(cols).to_numpy(),
        label=rows["sales"].to_numpy(),
        feature_name=cols,
        categorical_feature=CATEGORICAL,
        free_raw_data=True,
    )
    return lgb.train({**DEFAULT_PARAMS, **(params or {})}, data, num_boost_round=num_rounds)


def predict_wide(model: lgb.Booster, features: pl.DataFrame, start_day: int, horizon: int, n_series: int) -> np.ndarray:
    """Predict days start_day..start_day+horizon-1 and return (n_series, horizon); unlisted series get 0."""
    rows = features.filter((pl.col("d") >= start_day) & (pl.col("d") < start_day + horizon))
    preds = np.clip(model.predict(rows.select(model.feature_name()).to_numpy()), 0, None)
    out = np.zeros((n_series, horizon), dtype=np.float32)
    out[rows["series_idx"].to_numpy(), rows["d"].to_numpy() - start_day] = preds
    return out
