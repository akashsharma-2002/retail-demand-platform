"""Prometheus metrics shared by the API, optimiser, retrieval and assistant (exposed at /metrics)."""

from prometheus_client import Counter, Histogram

SOLVER_SECONDS = Histogram(
    "optimizer_solve_seconds",
    "Order optimisation solve time",
    ["status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
)
RETRIEVAL_CACHE = Counter("retrieval_cache_requests_total", "Policy retrieval cache lookups", ["result"])
RETRIEVAL_SECONDS = Histogram(
    "retrieval_seconds", "Policy retrieval time on cache miss", buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["model", "direction"])
LLM_SECONDS = Histogram("llm_request_seconds", "LLM request time", ["model"], buckets=(0.5, 1, 2, 4, 8, 16, 32))
ASSISTANT_SECONDS = Histogram(
    "assistant_request_seconds", "End-to-end assistant request time", ["intent"], buckets=(0.1, 0.5, 1, 2, 4, 8, 16, 32)
)
GUARD_RESULTS = Counter("assistant_guard_total", "Numeric faithfulness guard outcomes", ["result"])
