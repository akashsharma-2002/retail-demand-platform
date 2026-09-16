"""Planner assistant: route -> MCP tools -> explain -> numeric guard -> (approval interrupt) -> respond."""

import contextvars
import time
import uuid
from functools import lru_cache
from typing import Any, TypedDict

from fastmcp import Client, FastMCP
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy.orm import Session, sessionmaker

from retail_platform.agent.explain import SYSTEM_PROMPT, llm_prompt, template_answer
from retail_platform.agent.guard import unsupported_numbers
from retail_platform.agent.llm import OllamaChat, OpenAIChat, Usage
from retail_platform.agent.router import route
from retail_platform.mcp.server import build_server
from retail_platform.observability import tracing
from retail_platform.observability.metrics import ASSISTANT_SECONDS, GUARD_RESULTS
from retail_platform.services import planning
from retail_platform.services.errors import ConflictError, ForbiddenError, NotFoundError
from retail_platform.services.planning import Actor

MAX_TOOL_CALLS = 6
_current_actor: contextvars.ContextVar[Actor] = contextvars.ContextVar("current_actor")


class State(TypedDict, total=False):
    question: str
    default_store: str | None
    intent: str
    missing: str | None
    tool_calls: list[dict]
    draft: str
    answer: str
    mode: str
    guard: dict
    plan_id: str | None
    approval: dict | None


class Assistant:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        search,
        llm: OpenAIChat | OllamaChat | None = None,
        checkpoint_url: str | None = None,
    ):
        """checkpoint_url: PostgreSQL URL for durable conversation state (pending approvals survive restarts).
        Without it, state is kept in memory (tests and offline evals)."""
        self.session_factory = session_factory
        self.llm = llm
        self.server: FastMCP = build_server(session_factory, search, _current_actor.get)
        self._pool: Any = None
        if checkpoint_url:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            self._pool = AsyncConnectionPool(
                checkpoint_url,
                open=False,
                max_size=10,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            )
            self._checkpointer: Any = AsyncPostgresSaver(self._pool)
        else:
            self._checkpointer = InMemorySaver()
        self._ready = checkpoint_url is None
        self.graph = self._build().compile(checkpointer=self._checkpointer)
        self._usage: dict[str, Usage] = {}

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        await self._pool.open()
        await self._checkpointer.setup()
        self._ready = True

    # graph nodes -------------------------------------------------------------------------------------------
    def _route(self, state: State) -> State:
        r = route(state["question"], state.get("default_store"))
        return {
            "intent": r.intent,
            "missing": r.missing,
            "tool_calls": [{"tool": t, "args": a} for t, a in r.tools[:MAX_TOOL_CALLS]],
        }

    async def _call_tools(self, state: State) -> State:
        calls = []
        async with Client(self.server) as client:
            for call in state.get("tool_calls", []):
                args = {k: v for k, v in call["args"].items() if v is not None}
                with tracing.observe(f"tool.{call['tool']}", as_type="tool", input=args) as obs:
                    res = await client.call_tool(call["tool"], args, raise_on_error=False)
                    obs.update(output=res.structured_content, level="ERROR" if res.is_error else "DEFAULT")
                if res.is_error:
                    text = res.content[0].text if res.content else "tool error"
                    calls.append({**call, "result": None, "error": text})
                else:
                    data = res.structured_content
                    if isinstance(data, dict) and set(data) == {"result"}:
                        data = data["result"]
                    calls.append({**call, "result": data})
        return {"tool_calls": calls}

    def _explain(self, state: State, config) -> State:
        calls = state.get("tool_calls", [])
        template = template_answer(state["intent"], calls, state.get("missing"))
        usage = self._usage.setdefault(config["configurable"]["thread_id"], Usage())
        if (
            self.llm is None
            or state.get("missing")
            or state["intent"] == "unknown"
            or any(c.get("error") for c in calls)
            or state["intent"] in {"forecast", "position", "store_stock", "order"}
        ):
            return {"draft": template, "mode": "template"}
        try:
            draft = self.llm.complete(SYSTEM_PROMPT, llm_prompt(state["question"], state["intent"], calls), usage)
            return {"draft": draft, "mode": "llm"}
        except Exception as exc:  # fall back rather than fail the request
            return {"draft": template, "mode": f"template_fallback:{type(exc).__name__}"}

    def _guard(self, state: State) -> State:
        calls = [c.get("result") for c in state.get("tool_calls", [])]
        bad = unsupported_numbers(state["draft"], calls, state["question"])
        GUARD_RESULTS.labels("rewritten" if bad else "passed").inc()
        if bad:
            fallback = template_answer(state["intent"], state.get("tool_calls", []), state.get("missing"))
            return {"answer": fallback, "guard": {"passed": False, "unsupported": bad}, "mode": "template_guarded"}
        return {"answer": state["draft"], "guard": {"passed": True, "unsupported": []}}

    def _await_approval(self, state: State) -> State:
        plan = next(c["result"] for c in state["tool_calls"] if c["tool"] == "plan_replenishment")
        decision = interrupt(
            {
                "type": "approval_required",
                "plan_id": plan["plan_id"],
                "store_id": plan["store_id"],
                "total_cost": plan["total_cost"],
                "created_by": plan["created_by"],
            }
        )
        approver = Actor(decision["subject"], frozenset(decision["roles"]), decision.get("request_id"))
        try:
            with self.session_factory() as session:
                result = planning.decide_plan(
                    session, approver, plan["plan_id"], decision["approve"], decision.get("note")
                )
            verb = "approved" if decision["approve"] else "rejected"
            answer = f"Plan {plan['plan_id']} was {verb} by {approver.subject}."
            return {"plan_id": plan["plan_id"], "approval": result, "answer": answer}
        except (ForbiddenError, ConflictError, NotFoundError) as exc:
            return {
                "plan_id": plan["plan_id"],
                "approval": {"error": str(exc)},
                "answer": f"Plan {plan['plan_id']} is still waiting for approval: {exc}.",
            }

    def _after_guard(self, state: State) -> str:
        ordered = state["intent"] == "order" and any(
            c["tool"] == "plan_replenishment" and c.get("result") for c in state.get("tool_calls", [])
        )
        return "await_approval" if ordered else END

    def _build(self) -> StateGraph:
        g = StateGraph(State)
        g.add_node("route", self._route)
        g.add_node("call_tools", self._call_tools)
        g.add_node("explain", self._explain)
        g.add_node("guard", self._guard)
        g.add_node("await_approval", self._await_approval)
        g.add_edge(START, "route")
        g.add_conditional_edges("route", lambda s: "call_tools" if s.get("tool_calls") else "explain")
        g.add_edge("call_tools", "explain")
        g.add_edge("explain", "guard")
        g.add_conditional_edges("guard", self._after_guard, ["await_approval", END])
        g.add_edge("await_approval", END)
        return g

    # public API ----------------------------------------------------------------------------------------------
    async def ask(
        self, message: str, actor: Actor, thread_id: str | None = None, store_id: str | None = None
    ) -> dict[str, Any]:
        await self._ensure_ready()
        thread_id = thread_id or uuid.uuid4().hex
        token = _current_actor.set(actor)
        started = time.perf_counter()
        with (
            tracing.trace_attributes(actor.subject, thread_id),
            tracing.observe(
                "assistant.ask",
                as_type="agent",
                input=message,
                metadata={"user": actor.subject, "thread_id": thread_id},
            ) as trace,
        ):
            try:
                config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}
                self._usage[thread_id] = Usage()
                state = await self.graph.ainvoke(State(question=message, default_store=store_id), config)
            finally:
                _current_actor.reset(token)
            response = self._response(thread_id, state, started)
            trace.update(
                output=response["answer"],
                metadata={
                    "intent": response["intent"],
                    "mode": response["mode"],
                    "guard": response["guard"],
                    "awaiting_approval": bool(response["awaiting_approval"]),
                    "telemetry": response["telemetry"],
                },
            )
        ASSISTANT_SECONDS.labels(response["intent"] or "unknown").observe(time.perf_counter() - started)
        return response

    async def resume(self, thread_id: str, actor: Actor, approve: bool, note: str | None = None) -> dict[str, Any]:
        await self._ensure_ready()
        started = time.perf_counter()
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}
        snapshot = await self.graph.aget_state(config)
        if not snapshot.next:
            raise ConflictError("Nothing is waiting for approval in this conversation")
        state = await self.graph.ainvoke(
            Command[Any](
                resume={
                    "approve": approve,
                    "subject": actor.subject,
                    "roles": sorted(actor.roles),
                    "note": note,
                    "request_id": actor.request_id,
                }
            ),
            config,
        )
        return self._response(thread_id, state, started)

    def _response(self, thread_id: str, state: dict, started: float) -> dict[str, Any]:
        usage = self._usage.pop(thread_id, Usage())
        pending = state.get("__interrupt__")
        return {
            "thread_id": thread_id,
            "answer": state.get("answer"),
            "intent": state.get("intent"),
            "mode": state.get("mode"),
            "tools": [
                {"tool": c["tool"], "args": c["args"], "ok": not c.get("error")} for c in state.get("tool_calls", [])
            ],
            "data": {c["tool"]: c.get("result") for c in state.get("tool_calls", []) if c["tool"] != "search_policy"},
            "sources": [
                f"{h['title']} - {h['section']}"
                for c in state.get("tool_calls", [])
                if c["tool"] == "search_policy"
                for h in (c.get("result") or [])
                if h["trusted"]
            ],
            "guard": state.get("guard"),
            "awaiting_approval": pending[0].value if pending else None,
            "approval": state.get("approval"),
            "telemetry": {
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "llm_calls": usage.calls,
                "input_tokens": usage.input_tokens,
                "cached_tokens": usage.cached_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": round(usage.cost_usd, 6),
            },
        }


@lru_cache
def get_assistant() -> Assistant:
    from retail_platform.config import get_settings
    from retail_platform.rag.factory import build_searcher
    from retail_platform.storage.db import get_engine, session_factory

    settings = get_settings()
    llm: OpenAIChat | OllamaChat | None = None
    if settings.llm_provider == "ollama":
        llm = OllamaChat(settings.llm_model, settings.ollama_url, settings.ollama_api_key)
    elif settings.llm_provider == "openai" and settings.openai_api_key:
        llm = OpenAIChat(settings.openai_api_key, settings.llm_model)
    searcher = build_searcher(settings, get_engine())
    url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    checkpoint_url = url if url.startswith("postgresql://") else None  # SQLite in tests keeps state in memory
    return Assistant(session_factory(get_engine()), lambda q, k: searcher.search(q, k), llm, checkpoint_url)
