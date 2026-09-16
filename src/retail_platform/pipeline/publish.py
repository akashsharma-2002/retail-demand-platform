"""Write the champion's outputs to PostgreSQL for the API and assistant."""

import uuid

import numpy as np
import polars as pl
from sqlalchemy import delete, insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from retail_platform.forecast.conformal import CumulativeConformal
from retail_platform.pipeline.steps import SERVICE_LEVELS
from retail_platform.storage.migrate import upgrade_head
from retail_platform.storage.models import ConformalOffset, Forecast, Item, ModelRun

BATCH = 20_000


def publish(
    engine: Engine,
    supply: pl.DataFrame,
    velocity_bucket: np.ndarray,
    on_hand: np.ndarray,
    on_order: np.ndarray,
    forecast: np.ndarray,
    dates: list,
    conformal: CumulativeConformal,
    model_version: str,
    summary: dict,
) -> str:
    upgrade_head()
    s = supply.sort("series_idx")
    items = [
        {
            "series_idx": int(r["series_idx"]),
            "item_id": r["item_id"],
            "store_id": r["store_id"],
            "dept_id": r["dept_id"],
            "cat_id": r["cat_id"],
            "price": float(r["price"]),
            "unit_cost": float(r["unit_cost"]),
            "lead_time": int(r["lead_time"]),
            "case_pack": int(r["case_pack"]),
            "moq": int(r["moq"]),
            "shelf_capacity": int(r["shelf_capacity"]),
            "holding_cost_day": float(r["holding_cost_day"]),
            "shortage_cost": float(r["shortage_cost"]),
            "velocity_bucket": int(velocity_bucket[r["series_idx"]]),
            "on_hand": int(on_hand[r["series_idx"]]),
            "on_order": int(on_order[r["series_idx"]]),
        }
        for r in s.iter_rows(named=True)
    ]
    n_buckets = len(conformal.edges) + 1
    offsets = [
        {
            "bucket": b,
            "window": w,
            "level": level,
            "offset": float(conformal.quantile_offset(np.array([b]), np.array([w]), level)[0]),
        }
        for b in range(n_buckets)
        for w in range(1, forecast.shape[1] + 1)
        for level in SERVICE_LEVELS
    ]
    run_id = f"{model_version}-{uuid.uuid4().hex[:8]}"
    with Session(engine) as session, session.begin():
        session.execute(delete(Forecast))
        session.execute(delete(ConformalOffset))
        session.execute(delete(Item))
        session.execute(insert(Item), items)
        rows = [
            {"series_idx": int(i), "day": dates[d], "p50": float(forecast[i, d]), "model_version": model_version}
            for i in s["series_idx"].to_list()
            for d in range(forecast.shape[1])
        ]
        for start in range(0, len(rows), BATCH):
            session.execute(insert(Forecast), rows[start : start + BATCH])
        session.execute(insert(ConformalOffset), offsets)
        session.query(ModelRun).update({ModelRun.is_champion: False})
        session.add(ModelRun(run_id=run_id, is_champion=True, summary=summary))
    return run_id
