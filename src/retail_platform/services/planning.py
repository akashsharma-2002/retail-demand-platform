"""Business operations shared by the API, the MCP tools and the assistant."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from retail_platform.inventory.optimizer import OrderProblem, solve
from retail_platform.services.errors import ConflictError, ForbiddenError, NotFoundError
from retail_platform.storage.models import AuditLog, ConformalOffset, Forecast, Item, Plan, PlanLine

DEFAULT_SERVICE_LEVEL = 0.9
REVIEW_PERIOD = 7


@dataclass(frozen=True)
class Actor:
    subject: str
    roles: frozenset[str]
    request_id: str | None = None


def audit(
    session: Session,
    actor: Actor,
    action: str,
    entity: str,
    entity_id: str,
    before: dict | None = None,
    after: dict | None = None,
    note: str | None = None,
) -> None:
    session.add(
        AuditLog(
            actor=actor.subject,
            action=action,
            entity=entity,
            entity_id=entity_id,
            before=before,
            after=after,
            request_id=actor.request_id,
            note=note,
        )
    )


def get_item(session: Session, item_id: str, store_id: str) -> Item:
    item = session.scalar(select(Item).where(Item.item_id == item_id, Item.store_id == store_id))
    if item is None:
        raise NotFoundError(f"No item {item_id} in store {store_id}")
    return item


def get_forecast(session: Session, item_id: str, store_id: str, days: int = 28) -> dict:
    item = get_item(session, item_id, store_id)
    rows = session.scalars(
        select(Forecast).where(Forecast.series_idx == item.series_idx).order_by(Forecast.day).limit(days)
    ).all()
    cover = item.lead_time + REVIEW_PERIOD
    offsets = _offsets(session, [item.velocity_bucket], DEFAULT_SERVICE_LEVEL)
    daily = [{"day": r.day.isoformat(), "p50": round(r.p50, 2)} for r in rows]
    p50_cover = sum(r.p50 for r in rows[:cover])
    return {
        "item_id": item.item_id,
        "store_id": item.store_id,
        "model_version": rows[0].model_version if rows else None,
        "daily": daily,
        "days": len(daily),
        "total_p50": round(sum(r.p50 for r in rows), 1),
        "cover_days": cover,
        "cover_p50": round(p50_cover, 1),
        "cover_p90": round(p50_cover + offsets.get((item.velocity_bucket, cover), 0.0), 1),
    }


def get_inventory_position(session: Session, item_id: str, store_id: str) -> dict:
    item = get_item(session, item_id, store_id)
    return {
        "item_id": item.item_id,
        "store_id": item.store_id,
        "on_hand": item.on_hand,
        "on_order": item.on_order,
        "position": item.on_hand + item.on_order,
        "lead_time_days": item.lead_time,
        "case_pack": item.case_pack,
        "moq": item.moq,
        "shelf_capacity": item.shelf_capacity,
        "unit_cost": round(item.unit_cost, 2),
    }


def _offsets(session: Session, buckets: list[int], level: float) -> dict[tuple[int, int], float]:
    rows = session.scalars(
        select(ConformalOffset).where(ConformalOffset.level == level, ConformalOffset.bucket.in_(set(buckets)))
    ).all()
    return {(r.bucket, r.window): r.offset for r in rows}


def default_budget(session: Session, store_id: str) -> float:
    items = session.scalars(select(Item).where(Item.store_id == store_id)).all()
    first_week = _forecast_matrix(session, [i.series_idx for i in items], 7)
    return round(float(sum(first_week[i.series_idx].sum() * i.unit_cost for i in items)) * 1.15, 2)


def _forecast_matrix(session: Session, series: list[int], days: int) -> dict[int, np.ndarray]:
    rows = session.execute(
        select(Forecast.series_idx, Forecast.p50)
        .where(Forecast.series_idx.in_(series))
        .order_by(Forecast.series_idx, Forecast.day)
    ).all()
    out: dict[int, list[float]] = {s: [] for s in series}
    for s, p50 in rows:
        if len(out[s]) < days:
            out[s].append(p50)
    return {s: np.array(v + [0.0] * (days - len(v))) for s, v in out.items()}


def _targets(
    session: Session,
    items: Sequence[Item],
    service_level: float = DEFAULT_SERVICE_LEVEL,
    demand_change_pct: float = 0.0,
    lead_time_change_days: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cover days (lead time + review) and the upper-quantile demand target over that cover, per item."""
    lead = np.array([max(1, i.lead_time + lead_time_change_days) for i in items])
    cover = lead + REVIEW_PERIOD
    fc = _forecast_matrix(session, [i.series_idx for i in items], 28)
    offsets = _offsets(session, [i.velocity_bucket for i in items], service_level)
    factor = 1 + demand_change_pct / 100
    target = np.array(
        [
            fc[i.series_idx][:c].sum() * factor + offsets.get((i.velocity_bucket, int(min(c, 28))), 0.0)
            for i, c in zip(items, cover, strict=True)
        ]
    )
    return cover, target


def store_stock_summary(session: Session, store_id: str, top: int = 5) -> dict:
    """Store-level stock: totals, value, and the items most at risk of running out before the next delivery."""
    items = session.scalars(select(Item).where(Item.store_id == store_id).order_by(Item.series_idx)).all()
    if not items:
        raise NotFoundError(f"Unknown store {store_id}")
    _, target = _targets(session, items)
    on_hand = np.array([i.on_hand for i in items])
    on_order = np.array([i.on_order for i in items])
    unit_cost = np.array([i.unit_cost for i in items])
    shortfall = np.maximum(target - (on_hand + on_order), 0)
    at_risk = np.flatnonzero(shortfall > 0)
    worst = at_risk[np.argsort(-shortfall[at_risk] * unit_cost[at_risk])][:top]
    return {
        "store_id": store_id,
        "items": len(items),
        "units_on_hand": int(on_hand.sum()),
        "units_on_order": int(on_order.sum()),
        "stock_value_on_hand": round(float((on_hand * unit_cost).sum()), 2),
        "items_out_of_stock": int((on_hand == 0).sum()),
        "items_below_p90_target": len(at_risk),
        "shortfall_units": round(float(shortfall.sum()), 1),
        "most_at_risk": [
            {
                "item_id": items[i].item_id,
                "on_hand": int(on_hand[i]),
                "on_order": int(on_order[i]),
                "p90_target": round(float(target[i]), 1),
                "shortfall_units": round(float(shortfall[i]), 1),
            }
            for i in worst
        ],
    }


def compute_plan(
    session: Session,
    store_id: str,
    budget: float | None = None,
    demand_change_pct: float = 0.0,
    lead_time_change_days: int = 0,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> dict:
    """Solve the order plan without saving it."""
    items = session.scalars(select(Item).where(Item.store_id == store_id).order_by(Item.series_idx)).all()
    if not items:
        raise NotFoundError(f"Unknown store {store_id}")
    budget = default_budget(session, store_id) if budget is None else budget
    cover, target = _targets(session, items, service_level, demand_change_pct, lead_time_change_days)
    position = np.array([i.on_hand + i.on_order for i in items])
    need = np.maximum(target - position, 0)
    arr = lambda attr: np.array([getattr(i, attr) for i in items])  # noqa: E731
    sol = solve(
        OrderProblem(
            need=need,
            case_pack=arr("case_pack"),
            moq=arr("moq"),
            unit_cost=arr("unit_cost"),
            shortage_cost=arr("shortage_cost"),
            excess_cost=arr("holding_cost_day") * cover,
            capacity_left=np.maximum(arr("shelf_capacity") - position, 0),
            budget=budget,
        )
    )
    lines: list[dict[str, Any]] = [
        {
            "series_idx": i.series_idx,
            "item_id": i.item_id,
            "target": round(float(t), 1),
            "position": int(p),
            "need": round(float(n), 1),
            "units": int(u),
            "cost": round(float(u * i.unit_cost), 2),
        }
        for i, t, p, n, u in zip(items, target, position, need, sol.units, strict=True)
        if u > 0 or n > 0
    ]
    ordered = [ln for ln in lines if ln["units"] > 0]
    return {
        "store_id": store_id,
        "budget": round(budget, 2),
        "scenario": {
            "demand_change_pct": demand_change_pct,
            "lead_time_change_days": lead_time_change_days,
            "service_level": service_level,
        },
        "total_units": int(sol.units.sum()),
        "total_cost": round(sol.spend, 2),
        "lines_ordered": len(ordered),
        "items_needing_stock": sum(1 for ln in lines if ln["need"] > 0),
        "unfilled_need_units": round(float(np.maximum(need - sol.units, 0).sum()), 1),
        "solver_status": sol.status,
        "solve_seconds": round(sol.solve_seconds, 3),
        "top_lines": sorted(ordered, key=lambda ln: -ln["cost"])[:10],
        "lines": lines,
    }


def _supersede_pending(session: Session, actor: Actor, store_id: str, keep_plan_id: str, reason: str) -> int:
    """Older waiting plans were computed on stock positions that no longer hold; approving them would double-order."""
    stale = session.scalars(
        select(Plan).where(Plan.store_id == store_id, Plan.status == "pending_approval", Plan.plan_id != keep_plan_id)
    ).all()
    for old in stale:
        old.status = "superseded"
        old.decided_by = actor.subject
        old.decided_at = datetime.now(UTC)
        audit(
            session,
            actor,
            "plan.supersede",
            "plan",
            old.plan_id,
            before={"status": "pending_approval"},
            after={"status": "superseded"},
            note=f"{reason} {keep_plan_id}",
        )
    return len(stale)


def create_plan(session: Session, actor: Actor, store_id: str, budget: float | None = None, **scenario) -> dict:
    result = compute_plan(session, store_id, budget, **scenario)
    plan = Plan(
        store_id=store_id,
        budget=result["budget"],
        scenario=result["scenario"],
        created_by=actor.subject,
        total_units=result["total_units"],
        total_cost=result["total_cost"],
        lines_ordered=result["lines_ordered"],
        solver_status=result["solver_status"],
        solve_seconds=result["solve_seconds"],
    )
    plan.lines = [PlanLine(**ln) for ln in result["lines"]]
    session.add(plan)
    session.flush()
    audit(
        session,
        actor,
        "plan.create",
        "plan",
        plan.plan_id,
        after={"store_id": store_id, "total_cost": plan.total_cost, "status": plan.status},
    )
    superseded = _supersede_pending(session, actor, store_id, plan.plan_id, "replaced by newer plan")
    session.commit()
    return {**_plan_summary(plan), "top_lines": result["top_lines"], "superseded_plans": superseded}


def _plan_summary(plan: Plan) -> dict:
    return {
        "plan_id": plan.plan_id,
        "store_id": plan.store_id,
        "status": plan.status,
        "budget": plan.budget,
        "total_units": plan.total_units,
        "total_cost": plan.total_cost,
        "lines_ordered": plan.lines_ordered,
        "solver_status": plan.solver_status,
        "created_by": plan.created_by,
        "created_at": plan.created_at.isoformat() if plan.created_at else None,
        "decided_by": plan.decided_by,
        "scenario": plan.scenario,
    }


def get_plan(session: Session, plan_id: str, include_lines: bool = False) -> dict:
    plan = session.get(Plan, plan_id)
    if plan is None:
        raise NotFoundError(f"No plan {plan_id}")
    out = _plan_summary(plan)
    if include_lines:
        out["lines"] = [
            {"item_id": ln.item_id, "units": ln.units, "cost": ln.cost, "need": ln.need, "position": ln.position}
            for ln in sorted(plan.lines, key=lambda ln: -ln.cost)
            if ln.units > 0
        ]
    return out


def decide_plan(session: Session, actor: Actor, plan_id: str, approve: bool, note: str | None = None) -> dict:
    """Approve or reject. Requires the approver role and a different person from the plan creator."""
    if "approver" not in actor.roles:
        raise ForbiddenError("Only approvers can approve or reject plans")
    plan = session.get(Plan, plan_id)
    if plan is None:
        raise NotFoundError(f"No plan {plan_id}")
    if plan.created_by == actor.subject:
        raise ForbiddenError("You cannot approve or reject a plan you created")
    if plan.status != "pending_approval":
        raise ConflictError(f"Plan is already {plan.status}")
    before = {"status": plan.status}
    plan.status = "approved" if approve else "rejected"
    plan.decided_by = actor.subject
    plan.decided_at = datetime.now(UTC)
    if approve:
        for line in plan.lines:
            if line.units:
                item = session.get(Item, line.series_idx)
                if item is not None:
                    item.on_order += line.units
    audit(
        session,
        actor,
        "plan.approve" if approve else "plan.reject",
        "plan",
        plan_id,
        before=before,
        after={"status": plan.status},
        note=note,
    )
    superseded = (
        _supersede_pending(session, actor, plan.store_id, plan_id, "stock changed by approved plan") if approve else 0
    )
    session.commit()
    return {**_plan_summary(plan), "superseded_plans": superseded}


def list_stores(session: Session) -> list[str]:
    return list(session.scalars(select(Item.store_id).distinct().order_by(Item.store_id)).all())


def forecast_start(session: Session) -> date | None:
    return session.scalar(select(Forecast.day).order_by(Forecast.day).limit(1))


def explain_item_order(
    session: Session, item_id: str, store_id: str, service_level: float = DEFAULT_SERVICE_LEVEL
) -> dict:
    """All facts needed to explain one item's order, before budget and MOQ trade-offs across the store."""
    item = get_item(session, item_id, store_id)
    fc = get_forecast(session, item_id, store_id, 28)
    position = item.on_hand + item.on_order
    need = max(0.0, fc["cover_p90"] - position)
    packs = int(np.ceil(need / item.case_pack)) if need > 0 else 0
    units = max(packs * item.case_pack, item.moq) if packs else 0
    return {
        "item_id": item.item_id,
        "store_id": item.store_id,
        "service_level": service_level,
        "lead_time_days": item.lead_time,
        "review_period_days": REVIEW_PERIOD,
        "cover_days": fc["cover_days"],
        "cover_p50": fc["cover_p50"],
        "cover_p90": fc["cover_p90"],
        "on_hand": item.on_hand,
        "on_order": item.on_order,
        "position": position,
        "need_units": round(need, 1),
        "case_pack": item.case_pack,
        "moq": item.moq,
        "suggested_units": units,
        "suggested_cost": round(units * item.unit_cost, 2),
        "unit_cost": round(item.unit_cost, 2),
        "model_version": fc["model_version"],
    }


def list_plans(session: Session, store_id: str | None = None, status: str | None = None, limit: int = 50) -> list[dict]:
    query = select(Plan).order_by(Plan.created_at.desc()).limit(limit)
    if store_id:
        query = query.where(Plan.store_id == store_id)
    if status:
        query = query.where(Plan.status == status)
    return [_plan_summary(p) for p in session.scalars(query).all()]


def search_items(session: Session, store_id: str, query: str = "", limit: int = 20) -> list[dict]:
    stmt = select(Item).where(Item.store_id == store_id)
    if query:
        stmt = stmt.where(Item.item_id.startswith(query.upper()))
    rows = session.scalars(stmt.order_by(Item.item_id).limit(limit)).all()
    return [{"item_id": i.item_id, "dept_id": i.dept_id, "on_hand": i.on_hand} for i in rows]
