"""FastAPI application: versioned routes, auth, rate limits, request ids, metrics."""

import json
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.orm import Session

from retail_platform.api.auth import current_actor, require
from retail_platform.api.schemas import ChatRequest, DecisionRequest, PlanRequest, ResumeRequest
from retail_platform.config import get_settings
from retail_platform.services import planning
from retail_platform.services.errors import ConflictError, ForbiddenError, NotFoundError
from retail_platform.services.planning import Actor
from retail_platform.storage.db import get_session
from retail_platform.storage.models import ModelRun

log = structlog.get_logger()
REQUESTS = Histogram("http_request_seconds", "Request latency", ["method", "route", "status"])
ERRORS = Counter("http_errors_total", "Error responses", ["route", "status"])
MAX_BODY_BYTES = 16_000


def rate_key(request: Request) -> str:
    return getattr(request.state, "subject", None) or get_remote_address(request)


as_viewer = require("viewer")
as_planner = require("planner")
as_approver = require("approver")


@asynccontextmanager
async def lifespan(app: FastAPI):
    structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()])
    yield
    from retail_platform.observability import tracing

    tracing.flush()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Retail Demand & Inventory Decision Platform",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.env != "production" else None,
    )
    # One limiter per app instance: a shared module-level limiter registers every route limit again each time an
    # app is created, which silently counts each request twice.
    limiter = Limiter(key_func=rate_key, default_limits=["120/minute"], enabled=settings.rate_limit_enabled)
    app.state.limiter = limiter
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.exception_handler(RateLimitExceeded)
    async def rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse({"detail": "Too many requests. Try again in a minute."}, status_code=429)

    @app.exception_handler(NotFoundError)
    async def not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ForbiddenError)
    async def forbidden(request: Request, exc: ForbiddenError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(ConflictError)
    async def conflict(request: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        REQUESTS.labels(request.method, path, response.status_code).observe(elapsed)
        if response.status_code >= 400:
            ERRORS.labels(path, response.status_code).inc()
        response.headers["x-request-id"] = request.state.request_id
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["referrer-policy"] = "no-referrer"
        log.info(
            "request",
            method=request.method,
            path=path,
            status=response.status_code,
            ms=round(elapsed * 1000, 1),
            request_id=request.state.request_id,
            user=getattr(request.state, "subject", None),
        )
        return response

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def readyz(session: Session = Depends(get_session)) -> dict:
        session.execute(text("select 1"))
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    v1 = APIRouter(prefix="/v1")

    @v1.get("/me")
    def me(actor: Actor = Depends(current_actor)) -> dict:
        return {"subject": actor.subject, "roles": sorted(actor.roles)}

    @v1.get("/stores")
    @limiter.limit("300/minute")
    def stores(
        request: Request, actor: Actor = Depends(as_viewer), session: Session = Depends(get_session)
    ) -> list[str]:
        return planning.list_stores(session)

    @v1.get("/stores/{store_id}/items")
    def items(
        store_id: str, q: str = "", actor: Actor = Depends(as_viewer), session: Session = Depends(get_session)
    ) -> list[dict]:
        if len(q) > 40:
            raise HTTPException(422, "q is too long")
        return planning.search_items(session, store_id, q)

    @v1.get("/forecasts/{store_id}/{item_id}")
    @limiter.limit("300/minute")
    def forecast(
        request: Request,
        store_id: str,
        item_id: str,
        days: int = 28,
        actor: Actor = Depends(as_viewer),
        session: Session = Depends(get_session),
    ) -> dict:
        if not 1 <= days <= 28:
            raise HTTPException(422, "days must be between 1 and 28")
        return {
            **planning.get_forecast(session, item_id, store_id, days),
            "position": planning.get_inventory_position(session, item_id, store_id),
        }

    @v1.get("/backtest/summary")
    def backtest(actor: Actor = Depends(as_viewer), session: Session = Depends(get_session)) -> dict:
        run = (
            session.query(ModelRun).filter(ModelRun.is_champion.is_(True)).order_by(ModelRun.created_at.desc()).first()
        )
        if run is None:
            raise NotFoundError("No champion model has been published yet")
        return {"run_id": run.run_id, **run.summary}

    @v1.post("/plans", status_code=201)
    @limiter.limit("20/minute")
    def create_plan(
        request: Request,
        body: PlanRequest,
        actor: Actor = Depends(as_planner),
        session: Session = Depends(get_session),
    ) -> dict:
        return planning.create_plan(
            session,
            actor,
            body.store_id,
            body.budget,
            demand_change_pct=body.demand_change_pct,
            lead_time_change_days=body.lead_time_change_days,
        )

    @v1.get("/plans")
    def plans(
        store_id: str | None = None,
        status: str | None = None,
        actor: Actor = Depends(as_viewer),
        session: Session = Depends(get_session),
    ) -> list[dict]:
        if status and status not in {"pending_approval", "approved", "rejected", "superseded"}:
            raise HTTPException(422, "status must be pending_approval, approved, rejected or superseded")
        return planning.list_plans(session, store_id, status)

    @v1.get("/plans/{plan_id}")
    def get_plan(plan_id: str, actor: Actor = Depends(as_viewer), session: Session = Depends(get_session)) -> dict:
        return planning.get_plan(session, plan_id, include_lines=True)

    @v1.post("/plans/{plan_id}/approve")
    def approve(
        plan_id: str,
        body: DecisionRequest,
        actor: Actor = Depends(as_approver),
        session: Session = Depends(get_session),
    ) -> dict:
        return planning.decide_plan(session, actor, plan_id, approve=True, note=body.note)

    @v1.post("/plans/{plan_id}/reject")
    def reject(
        plan_id: str,
        body: DecisionRequest,
        actor: Actor = Depends(as_approver),
        session: Session = Depends(get_session),
    ) -> dict:
        return planning.decide_plan(session, actor, plan_id, approve=False, note=body.note)

    @v1.post("/assistant/chat")
    @limiter.limit("30/minute")
    async def chat(request: Request, body: ChatRequest, actor: Actor = Depends(as_planner)) -> dict:
        from retail_platform.agent.graph import get_assistant

        return await get_assistant().ask(body.message, actor, body.thread_id, body.store_id)

    @v1.post("/assistant/resume")
    async def resume(body: ResumeRequest, actor: Actor = Depends(as_approver)) -> dict:
        from retail_platform.agent.graph import get_assistant

        return await get_assistant().resume(body.thread_id, actor, body.approve, body.note)

    app.include_router(v1)
    return app


app = create_app()

if __name__ == "__main__":
    print(json.dumps(app.openapi())[:200])
