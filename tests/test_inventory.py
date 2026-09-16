import numpy as np
import polars as pl

from retail_platform.inventory.optimizer import OrderProblem, solve
from retail_platform.inventory.simulator import SupplyArrays, rule_of_thumb, simulate


def problem(budget: float, n: int = 40, seed: int = 0) -> OrderProblem:
    rng = np.random.default_rng(seed)
    return OrderProblem(
        need=rng.uniform(0, 60, n),
        case_pack=rng.choice([1, 6, 12], n),
        moq=rng.choice([6, 12, 24], n),
        unit_cost=rng.uniform(1, 20, n),
        shortage_cost=rng.uniform(1, 10, n),
        excess_cost=rng.uniform(0.01, 0.5, n),
        capacity_left=rng.integers(0, 200, n),
        budget=budget,
    )


def test_solution_respects_budget_case_packs_moq_and_capacity():
    for seed in range(5):
        p = problem(budget=800, seed=seed)
        sol = solve(p, time_limit_s=5)
        assert sol.status in {"OPTIMAL", "FEASIBLE"}
        assert (sol.units * p.unit_cost).sum() <= p.budget + 0.01
        assert np.all(sol.units % p.case_pack == 0)
        ordered = sol.units > 0
        assert np.all(sol.units[ordered] >= p.moq[ordered])
        assert np.all(sol.units <= p.capacity_left)


def test_more_budget_never_increases_cost_of_the_objective():
    tight = solve(problem(budget=300))
    loose = solve(problem(budget=5000))
    assert loose.objective <= tight.objective + 1e-6


def test_simulator_conserves_units():
    n, days = 5, 28
    rng = np.random.default_rng(1)
    sales = rng.poisson(3, size=(n, 100)).astype(float)
    supply = SupplyArrays.from_frame(
        pl.DataFrame(
            {
                "series_idx": list(range(n)),
                "store_id": ["CA_1"] * n,
                "lead_time": [2] * n,
                "case_pack": [6] * n,
                "moq": [6] * n,
                "shelf_capacity": [200] * n,
                "unit_cost": [1.0] * n,
                "holding_cost_day": [0.01] * n,
                "shortage_cost": [0.5] * n,
                "price": [2.0] * n,
            }
        )
    )
    result = simulate("rule", rule_of_thumb(supply, 7), sales, start_day=61, days=days, supply=supply, review_period=7)
    demand = sales[:, 60:88].sum()
    assert 0 <= result.fill_rate <= 1
    assert result.lost_sales_value / 2.0 <= demand
    start = np.ceil(sales[:, 32:60].mean(axis=1) * 9)
    sold = demand - result.lost_sales_value / 2.0
    assert start.sum() + result.ordered_units == sold + result.ending_on_hand.sum() + result.ending_on_order.sum()


def test_same_inputs_give_identical_plans():
    p = problem(budget=600, n=200, seed=11)
    first, second = solve(p), solve(p)
    assert np.array_equal(first.units, second.units)
    assert first.status == "OPTIMAL" and first.gap <= 0.001
