"""Shared inputs for forecasting stages: long data, features, wide sales and the evaluator."""

from dataclasses import dataclass
from functools import cached_property
from typing import cast

import numpy as np
import polars as pl

from retail_platform.config import Settings
from retail_platform.data.load import load_sales_wide
from retail_platform.features.build import build_features
from retail_platform.forecast.wrmsse import WRMSSEEvaluator


@dataclass
class ForecastContext:
    settings: Settings

    @cached_property
    def long(self) -> pl.DataFrame:
        return pl.read_parquet(self.settings.processed_dir / f"sales_{self.settings.state.lower()}.parquet")

    @cached_property
    def features(self) -> pl.DataFrame:
        return build_features(self.long)

    @cached_property
    def _wide(self) -> tuple[pl.DataFrame, np.ndarray]:
        return load_sales_wide(self.settings.raw_dir, self.settings.state)

    @property
    def ids(self) -> pl.DataFrame:
        return self._wide[0]

    @property
    def sales(self) -> np.ndarray:
        return self._wide[1]

    @cached_property
    def prices(self) -> np.ndarray:
        """(n_series, n_calendar_days) sell price, 0 where the item is not on sale."""
        n_days = cast(int, self.long["d"].max())
        out = np.zeros((self.ids.height, n_days), dtype=np.float32)
        out[self.long["series_idx"].to_numpy(), self.long["d"].to_numpy() - 1] = self.long["sell_price"].to_numpy()
        return out

    @cached_property
    def evaluator(self) -> WRMSSEEvaluator:
        n_obs = self.sales.shape[1]
        return WRMSSEEvaluator(self.ids, self.sales, self.sales * self.prices[:, :n_obs])

    @cached_property
    def train_start_d(self) -> int:
        start = pl.lit(self.settings.train_start).str.to_date()
        first = self.long.filter(pl.col("date") >= start)["d"].min()
        assert isinstance(first, int)
        return first
