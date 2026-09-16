import numpy as np
import polars as pl

from retail_platform.forecast.baselines import seasonal_naive, tsb
from retail_platform.forecast.conformal import CumulativeConformal
from retail_platform.forecast.reconcile import allocate_to_items, bottom_series, build_hierarchy, mint
from retail_platform.forecast.wrmsse import LEVELS, WRMSSEEvaluator, naive_scale


def ids_frame() -> pl.DataFrame:
    rows = []
    for store in ["CA_1", "CA_2"]:
        for dept in ["FOODS_1", "FOODS_2", "HOBBIES_1"]:
            for k in range(2):
                rows.append(
                    {
                        "item_id": f"{dept}_{k:03d}",
                        "dept_id": dept,
                        "cat_id": dept.split("_")[0],
                        "store_id": store,
                        "state_id": "CA",
                    }
                )
    return pl.DataFrame(rows)


def test_perfect_forecast_scores_zero_and_naive_scale_skips_leading_zeros():
    rng = np.random.default_rng(3)
    ids = ids_frame()
    sales = rng.poisson(4, size=(ids.height, 200)).astype(np.float32)
    ev = WRMSSEEvaluator(ids, sales, sales * 2.0)
    assert ev.score(sales[:, 172:], 173)["wrmsse"] == 0.0
    assert set(LEVELS) <= set(ev.score(sales[:, 172:] + 1, 173))
    hist = np.array([[0, 0, 0, 2, 4, 2]], dtype=float)
    assert naive_scale(hist)[0] == np.mean([4, 4])


def test_worse_forecast_scores_higher():
    rng = np.random.default_rng(4)
    ids = ids_frame()
    sales = rng.poisson(5, size=(ids.height, 300)).astype(np.float32)
    ev = WRMSSEEvaluator(ids, sales, sales)
    actual = sales[:, 272:]
    assert ev.score(actual * 1.1, 273)["wrmsse"] < ev.score(actual * 1.5, 273)["wrmsse"]


def test_baselines_shapes_and_values():
    hist = np.tile(np.arange(7, dtype=float), (2, 4))
    assert np.array_equal(seasonal_naive(hist, 10)[0], [0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    intermittent = np.array([[0, 0, 5, 0, 0, 5, 0, 0, 5] * 20], dtype=float)
    level = tsb(intermittent, 3)[0, 0]
    assert 1.0 < level < 3.0


def test_conformal_coverage_close_to_nominal():
    rng = np.random.default_rng(5)
    n = 4000
    rate = rng.uniform(0.2, 10, n)
    hist = rng.poisson(rate[:, None], (n, 60)).astype(float)
    cal_actual = rng.poisson(rate[:, None], (n, 28)).astype(float)
    test_actual = rng.poisson(rate[:, None], (n, 28)).astype(float)
    forecast = np.repeat(rate[:, None], 28, axis=1)
    conf = CumulativeConformal.fit(hist, forecast, cal_actual)
    coverage = conf.coverage(hist, forecast, test_actual, level=0.9, window=14)
    assert 0.87 <= coverage <= 0.93


def test_mint_output_is_coherent_and_allocation_matches_targets():
    ids = ids_frame()
    h = build_hierarchy(ids)
    rng = np.random.default_rng(6)
    items = rng.uniform(1, 5, size=(ids.height, 7))
    base = h.summing @ bottom_series(items, h) * rng.uniform(0.9, 1.1, size=(len(h.labels), 1))
    residuals = rng.normal(size=(len(h.labels), 60))
    reconciled = mint(base, residuals, h)
    assert reconciled.shape == (h.n_bottom, 7)
    allocated = allocate_to_items(items, reconciled, h)
    np.testing.assert_allclose(bottom_series(allocated, h), reconciled, rtol=1e-5)


def test_psi_detects_shift_and_ignores_same_distribution():
    from retail_platform.pipeline.drift import psi, status

    rng = np.random.default_rng(9)
    base = rng.normal(0, 1, 20_000)
    assert status(psi(base, rng.normal(0, 1, 20_000))) == "stable"
    assert status(psi(base, rng.normal(1.0, 1, 20_000))) == "significant"
