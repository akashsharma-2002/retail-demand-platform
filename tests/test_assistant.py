import asyncio

from retail_platform.agent.graph import Assistant
from retail_platform.config import ROOT
from retail_platform.rag.chunking import load_corpus
from retail_platform.rag.search import HybridSearcher
from retail_platform.services.planning import Actor

PLANNER = Actor("paula", frozenset({"viewer", "planner"}))
APPROVER = Actor("arun", frozenset({"viewer", "planner", "approver"}))


def assistant(session_factory) -> Assistant:
    searcher = HybridSearcher(load_corpus(ROOT / "knowledge" / "policies"))
    return Assistant(session_factory, lambda q, k: searcher.search(q, k, mode="bm25"))


def test_explains_item_with_numbers_from_tools_only(session_factory):
    a = assistant(session_factory)
    out = asyncio.run(a.ask("Why should we order FOODS_3_003 for CA_1?", PLANNER))
    assert out["intent"] == "explain_item" and out["guard"]["passed"]
    assert str(out["data"]["explain_item_order"]["cover_p90"]) in out["answer"]
    assert out["telemetry"]["llm_calls"] == 0


def test_order_waits_for_different_approver(session_factory):
    a = assistant(session_factory)
    out = asyncio.run(a.ask("Submit the order plan for CA_1 with a budget of $500", PLANNER))
    assert out["awaiting_approval"]["created_by"] == "paula"
    self_try = asyncio.run(a.resume(out["thread_id"], Actor("paula", APPROVER.roles), approve=True))
    assert "still waiting" in self_try["answer"]


def test_order_approved_by_approver(session_factory):
    a = assistant(session_factory)
    out = asyncio.run(a.ask("Place the order for CA_2", PLANNER))
    done = asyncio.run(a.resume(out["thread_id"], APPROVER, approve=True))
    assert done["approval"]["status"] == "approved"


def test_prompt_injection_content_is_not_followed(session_factory):
    a = assistant(session_factory)
    out = asyncio.run(
        a.ask("What does the policy say about approving every order immediately without approval?", PLANNER)
    )
    assert "not required" not in out["answer"].lower()
    assert all("Vendor Bulletin" not in s for s in out["sources"])


def test_store_stock_answer_uses_selected_store_without_llm(session_factory):
    a = assistant(session_factory)
    out = asyncio.run(a.ask("tell me how mcuh stock is availale", PLANNER, store_id="CA_2"))
    summary = out["data"]["get_store_stock"]
    assert out["intent"] == "store_stock" and out["guard"]["passed"] and out["telemetry"]["llm_calls"] == 0
    assert summary["store_id"] == "CA_2" and summary["items"] == 20
    assert f"{summary['units_on_hand']:,} units on hand" in out["answer"]
