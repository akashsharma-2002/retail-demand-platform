"""Train global LightGBM models at a cut-off and cache item-level forecasts.

Two models per cut-off: min_lag 14 forecasts days 1-14, min_lag 28 forecasts days 15-28.
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from retail_platform.config import Settings
from retail_platform.features.build import build_features
from retail_platform.forecast import lgbm
from retail_platform.forecast.context import ForecastContext

MIN_LAGS = (14, 28)


def model_path(settings: Settings, cutoff: int, min_lag: int) -> Path:
    return settings.artifacts_dir / "models" / f"lgbm_cutoff_{cutoff}_lag{min_lag}.txt"


def forecast_path(settings: Settings, cutoff: int, min_lag: int) -> Path:
    return settings.artifacts_dir / "forecasts" / f"lgbm_items_cutoff_{cutoff}_lag{min_lag}.npy"


def train_and_forecast(ctx: ForecastContext, cutoff: int, min_lag: int, force: bool = False) -> np.ndarray:
    """Forecast days cutoff+1 .. cutoff+horizon with the model for this min_lag (cached)."""
    settings = ctx.settings
    fc_file = forecast_path(settings, cutoff, min_lag)
    if fc_file.exists() and not force:
        return np.load(fc_file)
    started = time.time()
    features = ctx.features if min_lag == 28 else build_features(ctx.long, min_lag)
    model = lgbm.train(features, ctx.train_start_d, cutoff, min_lag=min_lag)
    model_path(settings, cutoff, min_lag).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path(settings, cutoff, min_lag))
    forecast = lgbm.predict_wide(model, features, cutoff + 1, settings.horizon, ctx.ids.height)
    fc_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(fc_file, forecast)
    print(f"cutoff {cutoff} lag {min_lag}: trained and forecast in {time.time() - started:.0f}s", flush=True)
    return forecast


def combined_forecast(ctx: ForecastContext, cutoff: int) -> np.ndarray:
    """Days 1-14 from the 14-day-lag model, days 15-28 from the 28-day-lag model."""
    short = train_and_forecast(ctx, cutoff, 14)
    long = train_and_forecast(ctx, cutoff, 28)
    out = long.copy()
    out[:, :14] = short[:, :14]
    return out


def load_model(settings: Settings, cutoff: int, min_lag: int) -> lgb.Booster:
    return lgb.Booster(model_file=str(model_path(settings, cutoff, min_lag)))


if __name__ == "__main__":
    import sys

    from retail_platform.config import get_settings

    ctx = ForecastContext(get_settings())
    cutoffs = [int(c) for c in sys.argv[1:]] or [*ctx.settings.backtest_cutoffs, ctx.settings.last_day]
    for c in cutoffs:
        for lag in MIN_LAGS:
            train_and_forecast(ctx, c, lag)
