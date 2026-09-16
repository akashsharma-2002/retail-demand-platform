"""Pipeline steps. Each step is a plain function so it can be tested and orchestrated (Prefect) separately."""

import hashlib
import json
import subprocess  # nosec B404 - fixed module invocation, no shell
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from retail_platform.config import Settings
from retail_platform.data.load import load_calendar, save_processed
from retail_platform.data.supply import build_supply
from retail_platform.features.build import build_features
from retail_platform.forecast import lgbm
from retail_platform.forecast.aggregate import ets_forecast, node_history
from retail_platform.forecast.baselines import seasonal_naive, tsb
from retail_platform.forecast.conformal import CumulativeConformal
from retail_platform.forecast.context import ForecastContext
from retail_platform.forecast.reconcile import allocate_to_items, bottom_series, build_hierarchy, mint
from retail_platform.forecast.train_global import combined_forecast, load_model, train_and_forecast
from retail_platform.forecast.wrmsse import bias_pct, wape
from retail_platform.inventory.simulator import (
    PlatformPolicy,
    SupplyArrays,
    rule_of_thumb,
    simulate,
    weekly_budgets,
)

CANDIDATES = ["lgbm_lag28", "lgbm_two_stage", "lgbm_two_stage_mint", "tft_middle_out"]
COVERAGE_WINDOWS = (7, 14, 21)
SERVICE_LEVELS = (0.8, 0.9, 0.95)


def data_fingerprint(settings: Settings) -> str:
    digest = hashlib.sha256()
    with open(settings.raw_dir / "sales_train_evaluation.csv", "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def ingest(settings: Settings, force: bool = False) -> Path:
    out = settings.processed_dir / f"sales_{settings.state.lower()}.parquet"
    return out if out.exists() and not force else save_processed(settings)


@dataclass
class BacktestResult:
    folds: dict[int, dict[str, dict[str, float]]] = field(default_factory=dict)
    forecasts: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)

    def mean(self, model: str, metric: str = "wrmsse") -> float:
        return float(np.mean([self.folds[c][model][metric] for c in self.folds]))

    def table(self) -> dict[str, dict[str, float]]:
        models = next(iter(self.folds.values())).keys()
        return {
            m: {
                "wrmsse_mean": round(self.mean(m), 4),
                "wape_mean": round(self.mean(m, "wape"), 4),
                "bias_pct_mean": round(self.mean(m, "bias_pct"), 2),
                **{f"wrmsse_{c}": round(self.folds[c][m]["wrmsse"], 4) for c in self.folds},
            }
            for m in models
        }


def candidate_forecasts(ctx: ForecastContext, cutoff: int, include_tft: bool) -> dict[str, np.ndarray]:
    sales, h = ctx.sales, build_hierarchy(ctx.ids)
    out = {"lgbm_lag28": train_and_forecast(ctx, cutoff, 28), "lgbm_two_stage": combined_forecast(ctx, cutoff)}
    two_stage = out["lgbm_two_stage"]
    ets, resid = ets_forecast(node_history(sales, cutoff, h), ctx.settings.horizon)
    blend = 0.5 * ets + 0.5 * (h.summing @ bottom_series(two_stage, h))
    out["lgbm_two_stage_mint"] = allocate_to_items(two_stage, mint(blend, resid, h), h)
    if include_tft:
        cache = ctx.settings.artifacts_dir / "forecasts" / f"tft_bottom_cutoff_{cutoff}.npy"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-m", "retail_platform.forecast.tft", str(cutoff), str(cache)], check=True
            )
        tft_bottom = np.load(cache)
        out["tft_middle_out"] = allocate_to_items(two_stage, tft_bottom, h)
    return out


def backtest(ctx: ForecastContext, include_tft: bool = True) -> BacktestResult:
    result = BacktestResult()
    horizon = ctx.settings.horizon
    for cutoff in ctx.settings.backtest_cutoffs:
        sales = ctx.sales
        actual = sales[:, cutoff : cutoff + horizon]
        forecasts = {
            "seasonal_naive": seasonal_naive(sales[:, :cutoff], horizon),
            "tsb": tsb(sales[:, cutoff - 730 : cutoff], horizon),
            **candidate_forecasts(ctx, cutoff, include_tft),
        }
        result.forecasts[cutoff] = forecasts
        result.folds[cutoff] = {}
        for name, fc in forecasts.items():
            scores = ctx.evaluator.score(fc, cutoff + 1)
            result.folds[cutoff][name] = {**scores, "wape": wape(actual, fc), "bias_pct": bias_pct(actual, fc)}
        print(
            f"backtest {cutoff}: " + ", ".join(f"{k}={v['wrmsse']:.4f}" for k, v in result.folds[cutoff].items()),
            flush=True,
        )
    return result


def select_candidate(result: BacktestResult) -> str:
    available = [c for c in CANDIDATES if c in next(iter(result.folds.values()))]
    return min(available, key=result.mean)


def conformal_report(ctx: ForecastContext, result: BacktestResult, model: str) -> dict:
    """Calibrate on the first fold, check coverage on the later folds."""
    cutoffs = ctx.settings.backtest_cutoffs
    sales, horizon = ctx.sales, ctx.settings.horizon
    first = cutoffs[0]
    conf = CumulativeConformal.fit(sales[:, :first], result.forecasts[first][model], sales[:, first : first + horizon])
    report: dict[str, dict[str, float]] = {}
    for cutoff in cutoffs[1:]:
        for level in SERVICE_LEVELS:
            for window in COVERAGE_WINDOWS:
                cov = conf.coverage(
                    sales[:, :cutoff],
                    result.forecasts[cutoff][model],
                    sales[:, cutoff : cutoff + horizon],
                    level,
                    window,
                )
                report.setdefault(f"level_{level}", {})[f"cutoff_{cutoff}_window_{window}"] = round(cov, 4)
    return report


def gate(
    candidate_wrmsse: float,
    baseline_wrmsse: float,
    coverage: dict,
    champion: dict | None,
    min_gain: float = 0.10,
    tolerance: float = 0.03,
) -> tuple[bool, str]:
    nominal_90 = list(coverage.get("level_0.9", {}).values())
    if nominal_90 and not all(abs(c - 0.9) <= tolerance for c in nominal_90):
        return False, f"P90 coverage outside 0.9 ± {tolerance}: {nominal_90}"
    if champion is None:
        gain = 1 - candidate_wrmsse / baseline_wrmsse
        if gain < min_gain:
            return False, f"First model must beat seasonal naive by {min_gain:.0%}; gain was {gain:.1%}"
        return True, f"No champion yet; beats seasonal naive by {gain:.1%}"
    if candidate_wrmsse >= champion["wrmsse_mean"]:
        return False, f"WRMSSE {candidate_wrmsse:.4f} does not beat champion {champion['wrmsse_mean']:.4f}"
    return True, f"Beats champion WRMSSE {champion['wrmsse_mean']:.4f} with {candidate_wrmsse:.4f}"


def model_forecast_fn(ctx: ForecastContext, cutoff: int, first_day: int, days: int, model: str):
    """Daily forecasts usable at any review day in [first_day, first_day + days) from models trained at cutoff.

    A target that is k days after the review uses the 14-day-lag model when k < 14, otherwise the 28-day-lag
    model, so no forecast ever relies on sales after the review day.
    """
    n = ctx.ids.height
    span = days + 28
    preds = {}
    for lag in (14, 28):
        feats = ctx.features if lag == 28 else build_features(ctx.long, lag)
        preds[lag] = lgbm.predict_wide(load_model(ctx.settings, cutoff, lag), feats, first_day, span, n)
    use_two_stage = model != "lgbm_lag28"

    def forecast_at(day: int, length: int) -> np.ndarray:
        offset = day - first_day
        out = preds[28][:, offset : offset + length].copy()
        if use_two_stage:
            short = min(14, length)
            out[:, :short] = preds[14][:, offset : offset + short]
        return out

    return forecast_at


def run_simulation(ctx: ForecastContext, result: BacktestResult, model: str) -> dict:
    s = ctx.settings
    days = s.simulation_weeks * 7
    start_day = s.last_day - days + 1
    cutoff = start_day - 1
    if cutoff not in s.backtest_cutoffs:
        raise ValueError("Simulation must start right after a backtest cut-off")
    first = s.backtest_cutoffs[0]
    conf = CumulativeConformal.fit(
        ctx.sales[:, :first], result.forecasts[first][model], ctx.sales[:, first : first + s.horizon]
    )
    supply_frame = build_supply(ctx.long, before_d=start_day)
    supply = SupplyArrays.from_frame(supply_frame)
    budget_fn = lambda hist: weekly_budgets(hist, supply)  # noqa: E731
    forecast_fn = model_forecast_fn(ctx, cutoff, start_day, days, model)

    runs = {
        "rule_of_thumb": simulate(
            "rule_of_thumb", rule_of_thumb(supply, s.review_period), ctx.sales, start_day, days, supply, s.review_period
        )
    }
    policies = {}
    for level in SERVICE_LEVELS:
        policy = PlatformPolicy(supply, forecast_fn, conf, level, s.review_period, budget_fn)
        started = time.time()
        res = simulate(f"platform_sl{level}", policy, ctx.sales, start_day, days, supply, s.review_period, budget_fn)
        res.solve_seconds = policy.solve_seconds
        runs[res.policy] = res
        policies[level] = res
        print(f"simulation service level {level}: {time.time() - started:.0f}s", flush=True)

    base = runs["rule_of_thumb"]
    chosen = runs[f"platform_sl{s.service_level}"]
    comparison = {
        "stockout_item_days_change_pct": pct(chosen.stockout_item_days, base.stockout_item_days),
        "lost_sales_value_change_pct": pct(chosen.lost_sales_value, base.lost_sales_value),
        "avg_inventory_value_change_pct": pct(chosen.avg_inventory_value, base.avg_inventory_value),
        "holding_cost_change_pct": pct(chosen.holding_cost, base.holding_cost),
        "fill_rate_points": round((chosen.fill_rate - base.fill_rate) * 100, 2),
    }
    return {
        "window": {"start_day": start_day, "days": days, "series": int(ctx.ids.height)},
        "policies": {k: v.as_dict() for k, v in runs.items()},
        "comparison_at_service_level": {"service_level": s.service_level, **comparison},
        "_supply": supply_frame,
        "_ending": chosen,
    }


def pct(new: float, old: float) -> float:
    return round((new - old) / old * 100, 2) if old else 0.0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    path.write_text(json.dumps(clean, indent=2, default=float))


def production_forecast(ctx: ForecastContext, model: str) -> tuple[np.ndarray, CumulativeConformal]:
    """Forecast the 28 days after the last observed day; conformal calibrated on the most recent fold."""
    s = ctx.settings
    latest = s.backtest_cutoffs[-1]
    result = candidate_forecasts(ctx, latest, include_tft=model == "tft_middle_out")
    conf = CumulativeConformal.fit(ctx.sales[:, :latest], result[model], ctx.sales[:, latest : latest + s.horizon])
    prod = candidate_forecasts(ctx, s.last_day, include_tft=model == "tft_middle_out")[model]
    return prod, conf


def calendar_dates(ctx: ForecastContext, first_day: int, days: int) -> list:
    cal = load_calendar(ctx.settings.raw_dir)
    return cal.filter((pl.col("d") >= first_day) & (pl.col("d") < first_day + days)).sort("d")["date"].to_list()
