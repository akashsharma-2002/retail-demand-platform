"""Day-by-day inventory replay on actual sales for a rule-of-thumb policy and the platform policy."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from retail_platform.forecast.conformal import CumulativeConformal
from retail_platform.inventory.optimizer import OrderProblem, solve


@dataclass
class SupplyArrays:
    store: np.ndarray
    lead_time: np.ndarray
    case_pack: np.ndarray
    moq: np.ndarray
    capacity: np.ndarray
    unit_cost: np.ndarray
    holding_cost_day: np.ndarray
    shortage_cost: np.ndarray
    price: np.ndarray

    @classmethod
    def from_frame(cls, supply: pl.DataFrame) -> "SupplyArrays":
        s = supply.sort("series_idx")
        return cls(
            store=s["store_id"].to_numpy(),
            lead_time=s["lead_time"].to_numpy().astype(np.int64),
            case_pack=s["case_pack"].to_numpy().astype(np.int64),
            moq=s["moq"].to_numpy().astype(np.int64),
            capacity=s["shelf_capacity"].to_numpy().astype(np.int64),
            unit_cost=s["unit_cost"].to_numpy(),
            holding_cost_day=s["holding_cost_day"].to_numpy(),
            shortage_cost=s["shortage_cost"].to_numpy(),
            price=s["price"].to_numpy(),
        )


@dataclass
class SimulationResult:
    policy: str
    ordered_units: int
    ending_on_hand: np.ndarray
    ending_on_order: np.ndarray
    fill_rate: float
    stockout_item_days: int
    lost_sales_value: float
    avg_inventory_value: float
    holding_cost: float
    ordered_value: float
    budget_violations: int
    solve_seconds: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int | str]:
        d = {k: v for k, v in self.__dict__.items() if k not in {"solve_seconds", "ending_on_hand", "ending_on_order"}}
        if self.solve_seconds:
            d["solver_p95_seconds"] = float(np.percentile(self.solve_seconds, 95))
        return d


OrderFn = Callable[[int, np.ndarray, np.ndarray], np.ndarray]  # (day, history, position) -> units
BudgetFn = Callable[[np.ndarray], dict[str, float]]  # history -> weekly budget per store
ForecastFn = Callable[[int, int], np.ndarray]  # (review day, days) -> (n, days) daily forecasts from that day


def round_to_case(units: np.ndarray, supply: SupplyArrays) -> np.ndarray:
    packs = np.ceil(np.maximum(units, 0) / supply.case_pack)
    rounded = packs * supply.case_pack
    return np.where(rounded > 0, np.maximum(rounded, supply.moq), 0).astype(np.int64)


def simulate(
    policy: str,
    order_fn: OrderFn,
    sales: np.ndarray,
    start_day: int,
    days: int,
    supply: SupplyArrays,
    review_period: int,
    budget_fn: BudgetFn | None = None,
) -> SimulationResult:
    """sales columns are day numbers minus one. Replays days start_day..start_day+days-1."""
    n = sales.shape[0]
    hist0 = sales[:, : start_day - 1]
    on_hand = np.ceil(hist0[:, -28:].mean(axis=1) * (supply.lead_time + review_period)).astype(np.int64)
    arrivals = np.zeros((n, days + 30), dtype=np.int64)
    ordered_units = 0
    demand_total = sold_total = stockouts = 0
    lost_value = holding = ordered_value = inv_value_sum = 0.0
    violations = 0
    for k in range(days):
        day = start_day + k
        on_hand += arrivals[:, k]
        if k % review_period == 0:
            on_order = arrivals[:, k + 1 :].sum(axis=1)
            units = order_fn(day, sales[:, : day - 1], on_hand + on_order)
            ordered_value += float((units * supply.unit_cost).sum())
            ordered_units += int(units.sum())
            if budget_fn is not None:
                for store, budget in budget_fn(sales[:, : day - 1]).items():
                    spend = float((units[supply.store == store] * supply.unit_cost[supply.store == store]).sum())
                    violations += int(spend > budget + 0.01)
            np.add.at(arrivals, (np.arange(n), k + supply.lead_time), units)
        demand = sales[:, day - 1].astype(np.int64)
        sold = np.minimum(on_hand, demand)
        lost = demand - sold
        on_hand -= sold
        demand_total += int(demand.sum())
        sold_total += int(sold.sum())
        stockouts += int((lost > 0).sum())
        lost_value += float((lost * supply.price).sum())
        holding += float((on_hand * supply.holding_cost_day).sum())
        inv_value_sum += float((on_hand * supply.unit_cost).sum())
    return SimulationResult(
        policy=policy,
        ordered_units=ordered_units,
        ending_on_hand=on_hand.copy(),
        ending_on_order=arrivals[:, days:].sum(axis=1),
        fill_rate=sold_total / max(demand_total, 1),
        stockout_item_days=stockouts,
        lost_sales_value=lost_value,
        avg_inventory_value=inv_value_sum / days,
        holding_cost=holding,
        ordered_value=ordered_value,
        budget_violations=violations,
    )


def rule_of_thumb(supply: SupplyArrays, review_period: int, safety_days: int = 7) -> OrderFn:
    def order(day: int, history: np.ndarray, position: np.ndarray) -> np.ndarray:
        mean28 = history[:, -28:].mean(axis=1)
        reorder_point = mean28 * (supply.lead_time + safety_days)
        order_up_to = mean28 * (supply.lead_time + review_period + safety_days)
        units = np.where(position < reorder_point, order_up_to - position, 0)
        return round_to_case(units, supply)

    return order


@dataclass
class PlatformPolicy:
    supply: SupplyArrays
    forecast_fn: ForecastFn
    conformal: CumulativeConformal
    service_level: float
    review_period: int
    budget_fn: BudgetFn
    solve_seconds: list[float] = field(default_factory=list)

    def targets(self, day: int, history: np.ndarray) -> np.ndarray:
        cover = self.supply.lead_time + self.review_period
        cum = np.cumsum(self.forecast_fn(day, int(cover.max())), axis=1)
        point = cum[np.arange(len(cover)), cover - 1]
        buckets = self.conformal.buckets(history)
        return point + self.conformal.quantile_offset(buckets, cover, self.service_level)

    def __call__(self, day: int, history: np.ndarray, position: np.ndarray) -> np.ndarray:
        need = np.maximum(self.targets(day, history) - position, 0)
        units = np.zeros(len(need), dtype=np.int64)
        cover = self.supply.lead_time + self.review_period
        for store, budget in self.budget_fn(history).items():
            idx = np.flatnonzero(self.supply.store == store)
            sol = solve(
                OrderProblem(
                    need=need[idx],
                    case_pack=self.supply.case_pack[idx],
                    moq=self.supply.moq[idx],
                    unit_cost=self.supply.unit_cost[idx],
                    shortage_cost=self.supply.shortage_cost[idx],
                    excess_cost=self.supply.holding_cost_day[idx] * cover[idx],
                    capacity_left=np.maximum(self.supply.capacity[idx] - position[idx], 0),
                    budget=budget,
                )
            )
            units[idx] = sol.units
            self.solve_seconds.append(sol.solve_seconds)
        return units


def weekly_budgets(sales_hist: np.ndarray, supply: SupplyArrays, headroom: float = 1.15) -> dict[str, float]:
    weekly_cost = sales_hist[:, -28:].mean(axis=1) * 7 * supply.unit_cost
    return {s: float(weekly_cost[supply.store == s].sum() * headroom) for s in np.unique(supply.store)}
