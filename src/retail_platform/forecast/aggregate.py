"""Aggregate-level statistical forecasts for every hierarchy node (they see the most recent sales)."""

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from retail_platform.forecast.reconcile import Hierarchy, bottom_series

HISTORY_DAYS = 728


def node_history(sales: np.ndarray, cutoff: int, hierarchy: Hierarchy) -> np.ndarray:
    bottom = bottom_series(sales[:, cutoff - HISTORY_DAYS : cutoff], hierarchy)
    return hierarchy.summing @ bottom


def ets_forecast(history: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (forecast (n_nodes, h), in-sample residuals (n_nodes, T))."""
    n, t = history.shape
    frame = pd.DataFrame(
        {
            "unique_id": np.repeat(np.arange(n), t),
            "ds": np.tile(pd.date_range("2000-01-01", periods=t, freq="D"), n),
            "y": history.reshape(-1),
        }
    )
    sf = StatsForecast(models=[AutoETS(season_length=7)], freq="D", n_jobs=1)
    fc = sf.forecast(df=frame, h=horizon, fitted=True)
    fitted = sf.forecast_fitted_values()
    forecast = fc.sort_values(["unique_id", "ds"])["AutoETS"].to_numpy().reshape(n, horizon)
    fitted = fitted.sort_values(["unique_id", "ds"])
    residuals = (fitted["y"] - fitted["AutoETS"]).to_numpy().reshape(n, t)
    return np.clip(forecast, 0, None), np.nan_to_num(residuals)
