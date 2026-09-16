"""Relational schema. Vectors and BM25 for policy search live in rag/pg_store.py (PostgreSQL only)."""

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Item(Base):
    __tablename__ = "items"
    series_idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[str] = mapped_column(String(40), index=True)
    store_id: Mapped[str] = mapped_column(String(10), index=True)
    dept_id: Mapped[str] = mapped_column(String(20))
    cat_id: Mapped[str] = mapped_column(String(20))
    price: Mapped[float] = mapped_column(Float)
    unit_cost: Mapped[float] = mapped_column(Float)
    lead_time: Mapped[int] = mapped_column(Integer)
    case_pack: Mapped[int] = mapped_column(Integer)
    moq: Mapped[int] = mapped_column(Integer)
    shelf_capacity: Mapped[int] = mapped_column(Integer)
    holding_cost_day: Mapped[float] = mapped_column(Float)
    shortage_cost: Mapped[float] = mapped_column(Float)
    velocity_bucket: Mapped[int] = mapped_column(Integer)
    on_hand: Mapped[int] = mapped_column(Integer)
    on_order: Mapped[int] = mapped_column(Integer, default=0)


class Forecast(Base):
    __tablename__ = "forecasts"
    series_idx: Mapped[int] = mapped_column(ForeignKey("items.series_idx"), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    p50: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64))


class ConformalOffset(Base):
    __tablename__ = "conformal_offsets"
    bucket: Mapped[int] = mapped_column(Integer, primary_key=True)
    window: Mapped[int] = mapped_column(Integer, primary_key=True)
    level: Mapped[float] = mapped_column(Float, primary_key=True)
    offset: Mapped[float] = mapped_column(Float)


class ModelRun(Base):
    __tablename__ = "model_runs"
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_champion: Mapped[bool] = mapped_column(default=False)
    summary: Mapped[dict] = mapped_column(JSON)


class Plan(Base):
    __tablename__ = "plans"
    plan_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    store_id: Mapped[str] = mapped_column(String(10), index=True)
    budget: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="pending_approval")
    scenario: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_units: Mapped[int] = mapped_column(Integer)
    total_cost: Mapped[float] = mapped_column(Float)
    lines_ordered: Mapped[int] = mapped_column(Integer)
    solver_status: Mapped[str] = mapped_column(String(32))
    solve_seconds: Mapped[float] = mapped_column(Float)
    lines: Mapped[list["PlanLine"]] = relationship(back_populates="plan", cascade="all, delete-orphan")


class PlanLine(Base):
    __tablename__ = "plan_lines"
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.plan_id"), primary_key=True)
    series_idx: Mapped[int] = mapped_column(ForeignKey("items.series_idx"), primary_key=True)
    item_id: Mapped[str] = mapped_column(String(40))
    target: Mapped[float] = mapped_column(Float)
    position: Mapped[int] = mapped_column(Integer)
    need: Mapped[float] = mapped_column(Float)
    units: Mapped[int] = mapped_column(Integer)
    cost: Mapped[float] = mapped_column(Float)
    plan: Mapped[Plan] = relationship(back_populates="lines")


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(64))
    entity: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str] = mapped_column(String(64))
    before: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    after: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
