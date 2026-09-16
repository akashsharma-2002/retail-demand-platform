"""Order quantities for one store at one review: a mixed-integer program solved with HiGHS through OR-Tools.

HiGHS was chosen over CP-SAT after benchmarking real store plans (3,049 items): it proves optimality in under a
second, including weeks where the budget binds, and returns identical plans on every run. See ADR-005.
"""

import time
from dataclasses import dataclass

import numpy as np
from ortools.linear_solver import pywraplp

from retail_platform.observability.metrics import SOLVER_SECONDS


@dataclass
class OrderProblem:
    need: np.ndarray  # units wanted per item (>= 0)
    case_pack: np.ndarray
    moq: np.ndarray
    unit_cost: np.ndarray
    shortage_cost: np.ndarray  # per unit short
    excess_cost: np.ndarray  # per unit ordered above need (holding over the cover period)
    capacity_left: np.ndarray  # shelf capacity minus current position
    budget: float


@dataclass
class OrderSolution:
    units: np.ndarray
    status: str
    objective: float
    spend: float
    solve_seconds: float
    gap: float = 0.0  # relative distance between the plan's cost and the best proven lower bound


STATUS = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}


def solve(problem: OrderProblem, time_limit_s: float = 20.0, gap_limit: float = 0.001) -> OrderSolution:
    """Minimise shortage + excess cost subject to budget, MOQ, whole case packs and shelf capacity."""
    n = len(problem.need)
    units = np.zeros(n, dtype=np.int64)
    need = np.ceil(problem.need).astype(np.int64)
    active = np.flatnonzero((need > 0) & (problem.capacity_left >= problem.moq) & (problem.capacity_left > 0))
    if len(active) == 0:
        return OrderSolution(units, "NOTHING_TO_ORDER", 0.0, 0.0, 0.0)

    solver = pywraplp.Solver.CreateSolver("HIGHS")
    if solver is None:  # pragma: no cover - HiGHS ships with OR-Tools
        raise RuntimeError("HiGHS solver is not available in this OR-Tools build")

    spend_terms, cost_terms, packs_vars = [], [], {}
    for i in active:
        case = int(problem.case_pack[i])
        max_packs = int(problem.capacity_left[i]) // case
        min_packs = -(-int(problem.moq[i]) // case)  # ceil(moq / case)
        if max_packs == 0 or max_packs < min_packs:
            continue
        packs = solver.IntVar(0, max_packs, f"packs_{i}")
        if min_packs > 1:  # otherwise any order of one case already satisfies the MOQ
            order = solver.BoolVar(f"order_{i}")
            solver.Add(packs >= min_packs * order)
            solver.Add(packs <= max_packs * order)
        short = solver.NumVar(0, float(need[i]), f"short_{i}")
        excess = solver.NumVar(0, float(max_packs * case), f"excess_{i}")
        solver.Add(case * packs - float(need[i]) == excess - short)
        cost_terms.append(float(problem.shortage_cost[i]) * short + float(problem.excess_cost[i]) * excess)
        spend_terms.append(float(case * problem.unit_cost[i]) * packs)
        packs_vars[i] = (packs, case)

    solver.Add(solver.Sum(spend_terms) <= float(problem.budget))
    solver.Minimize(solver.Sum(cost_terms))
    solver.SetTimeLimit(int(time_limit_s * 1000))
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, gap_limit)

    started = time.perf_counter()
    status = solver.Solve(params)
    elapsed = time.perf_counter() - started
    solved = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    if solved:
        for i, (packs, case) in packs_vars.items():
            units[i] = round(packs.solution_value()) * case
    objective = solver.Objective().Value() if solved else 0.0
    bound = solver.Objective().BestBound() if solved else 0.0
    gap = max((objective - bound) / max(abs(objective), 1e-9), 0.0) if solved else 1.0
    spend = float((units * problem.unit_cost).sum())
    SOLVER_SECONDS.labels(STATUS.get(status, str(status))).observe(elapsed)
    return OrderSolution(units, STATUS.get(status, str(status)), objective, spend, elapsed, round(gap, 5))
