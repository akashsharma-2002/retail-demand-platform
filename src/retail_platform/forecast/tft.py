"""Temporal Fusion Transformer challenger on store x department series (NeuralForecast)."""

import logging

import numpy as np
import pandas as pd
import polars as pl

from retail_platform.forecast.reconcile import Hierarchy, bottom_series

HISTORY_DAYS = 728


def tft_bottom_forecast(
    sales: np.ndarray,
    calendar: pl.DataFrame,
    cutoff: int,
    hierarchy: Hierarchy,
    horizon: int = 28,
    max_steps: int = 800,
    seed: int = 1,
) -> np.ndarray:
    """Forecast each store x department series for days cutoff+1..cutoff+horizon. Returns (n_bottom, horizon)."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    history = bottom_series(sales[:, cutoff - HISTORY_DAYS : cutoff], hierarchy)
    cal = calendar.to_pandas().set_index("d")
    past_days = np.arange(cutoff - HISTORY_DAYS + 1, cutoff + 1)
    future_days = np.arange(cutoff + 1, cutoff + horizon + 1)

    def frame(days: np.ndarray, values: np.ndarray | None) -> pd.DataFrame:
        rows = []
        for b, label in enumerate(hierarchy.bottom_labels):
            part = pd.DataFrame(
                {
                    "unique_id": label,
                    "ds": pd.to_datetime(cal.loc[days, "date"].to_numpy()),
                    "wday": cal.loc[days, "wday"].to_numpy(),
                    "snap": cal.loc[days, "snap_CA"].to_numpy(),
                    "event": cal.loc[days, "event_type_1"].notna().astype(int).to_numpy(),
                }
            )
            if values is not None:
                part["y"] = values[b]
            rows.append(part)
        return pd.concat(rows, ignore_index=True)

    model = TFT(
        h=horizon,
        input_size=4 * horizon,
        hidden_size=64,
        max_steps=max_steps,
        learning_rate=1e-3,
        scaler_type="robust",
        futr_exog_list=["wday", "snap", "event"],
        random_seed=seed,
        batch_size=32,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="cpu",
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=frame(past_days, history))
    pred = nf.predict(futr_df=frame(future_days, None))
    pred = pred.sort_values(["unique_id", "ds"])
    by_label = {k: g["TFT"].to_numpy() for k, g in pred.groupby("unique_id")}
    return np.clip(np.vstack([by_label[label] for label in hierarchy.bottom_labels]), 0, None)


def main() -> None:
    """Run in a separate process: PyTorch and LightGBM each load their own OpenMP runtime, which can crash when
    both live in one process on macOS."""
    import sys

    from retail_platform.config import get_settings
    from retail_platform.data.load import load_calendar, load_sales_wide
    from retail_platform.forecast.reconcile import build_hierarchy

    cutoff, out = int(sys.argv[1]), sys.argv[2]
    settings = get_settings()
    ids, sales = load_sales_wide(settings.raw_dir, settings.state)
    forecast = tft_bottom_forecast(
        sales,
        load_calendar(settings.raw_dir),
        cutoff,
        build_hierarchy(ids),
        settings.horizon,
        max_steps=int(sys.argv[3]) if len(sys.argv) > 3 else 400,
    )
    np.save(out, forecast)


if __name__ == "__main__":
    main()
