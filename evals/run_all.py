"""Offline evaluations: retrieval quality, agent routing, numeric guard. Writes artifacts/reports/evals.json."""

import json
import os
import time
from pathlib import Path

import numpy as np

from retail_platform.agent.guard import unsupported_numbers
from retail_platform.agent.router import route
from retail_platform.config import ROOT, get_settings
from retail_platform.rag.chunking import load_corpus
from retail_platform.rag.search import CrossEncoderReranker, HybridSearcher, SentenceTransformerEmbedder

HERE = Path(__file__).parent


def read_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / name).read_text().splitlines() if line.strip()]


def retrieval(modes: list[str], rerank_tops: tuple[int, ...] = (10, 20, 30)) -> dict:
    settings = get_settings()
    corpus = load_corpus(ROOT / "knowledge" / "policies")
    needs_models = any(m != "bm25" for m in modes)
    embedder = SentenceTransformerEmbedder(settings.embedding_model) if needs_models else None
    reranker = CrossEncoderReranker(settings.reranker_model) if "hybrid_rerank" in modes else None
    searcher = HybridSearcher(corpus, embedder, reranker)
    golden = read_jsonl("retrieval_golden.jsonl")
    out = {}
    runs = [(m, None) for m in modes if m != "hybrid_rerank"]
    runs += [("hybrid_rerank", k) for k in rerank_tops] if "hybrid_rerank" in modes else []
    for mode, top in runs:
        if top is not None:
            searcher.rerank_top = top
        label = f"{mode}_top{top}" if top is not None else mode
        ranks, latencies = [], []
        for case in golden:
            started = time.perf_counter()
            hits = searcher.search(case["q"], top_k=5, mode=mode)
            latencies.append((time.perf_counter() - started) * 1000)
            ids = [h.chunk_id for h in hits]
            ranks.append(ids.index(case["expected"]) + 1 if case["expected"] in ids else None)
        out[label] = {
            "hit_at_1": round(np.mean([r == 1 for r in ranks]), 4),
            "hit_at_3": round(np.mean([r is not None and r <= 3 for r in ranks]), 4),
            "hit_at_5": round(np.mean([r is not None for r in ranks]), 4),
            "mrr_at_5": round(np.mean([1 / r if r else 0 for r in ranks]), 4),
            "p50_ms": round(float(np.percentile(latencies, 50)), 1),
            "p95_ms": round(float(np.percentile(latencies, 95)), 1),
            "misses": [c["q"] for c, r in zip(golden, ranks, strict=True) if r is None],
        }
    return {"questions": len(golden), "chunks": len(corpus), "modes": out}


def routing() -> dict:
    cases = read_jsonl("agent_scenarios.jsonl")
    intent_ok = tool_ok = args_ok = 0
    failures = []
    for case in cases:
        r = route(case["q"])
        tool = r.tools[0][0] if r.tools else None
        args = r.tools[0][1] if r.tools else {}
        i_ok = r.intent == case["intent"] and (case.get("missing") == r.missing)
        t_ok = tool == case["tool"]
        a_ok = all(args.get(k) == v for k, v in case["args"].items())
        intent_ok += i_ok
        tool_ok += t_ok
        args_ok += t_ok and a_ok
        if not (i_ok and t_ok and a_ok):
            failures.append({"q": case["q"], "got": {"intent": r.intent, "tool": tool, "args": args}})
    n = len(cases)
    return {
        "scenarios": n,
        "intent_accuracy": round(intent_ok / n, 4),
        "tool_accuracy": round(tool_ok / n, 4),
        "tool_and_args_accuracy": round(args_ok / n, 4),
        "failures": failures,
    }


def guard() -> dict:
    cases = read_jsonl("guard_cases.jsonl")
    tp = fp = tn = fn = 0
    for case in cases:
        flagged = bool(unsupported_numbers(case["answer"], case["tools"], ""))
        if not case["faithful"]:
            tp += flagged
            fn += not flagged
        else:
            fp += flagged
            tn += not flagged
    return {
        "cases": len(cases),
        "caught_unfaithful": f"{tp}/{tp + fn}",
        "false_alarms": f"{fp}/{fp + tn}",
        "recall": round(tp / max(tp + fn, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
    }


if __name__ == "__main__":
    modes = os.getenv("RP_EVAL_MODES", "bm25,dense,hybrid,hybrid_rerank").split(",")
    report = {"routing": routing(), "guard": guard(), "retrieval": retrieval(modes)}
    out = ROOT / "artifacts" / "reports" / "evals.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk not in {"failures", "misses"}} for k, v in report.items()},
            indent=2,
        )
    )
