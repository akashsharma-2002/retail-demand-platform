"""Optional Langfuse tracing. Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set."""

import contextlib
import os
from collections.abc import Iterator
from functools import lru_cache
from typing import Any


class _NoopObservation:
    def update(self, **_: Any) -> None:
        return None


@lru_cache
def _client():
    public, secret = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
    if not (public and secret):
        return None
    from langfuse import Langfuse

    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
    return Langfuse(
        public_key=public,
        secret_key=secret,
        base_url=base_url,
        environment=os.getenv("RP_ENV", "local"),
        release="1.0.0",
        blocked_instrumentation_scopes=["fastmcp"],  # our own tool spans already cover MCP calls
    )


@contextlib.contextmanager
def trace_attributes(user_id: str, session_id: str) -> Iterator[None]:
    """Tag every observation in this request with the user and conversation (filterable in Langfuse)."""
    if _client() is None:
        yield
        return
    from langfuse import propagate_attributes

    with propagate_attributes(user_id=user_id, session_id=session_id):
        yield


def enabled() -> bool:
    return _client() is not None


@contextlib.contextmanager
def observe(name: str, as_type: str = "span", **fields: Any) -> Iterator[Any]:
    """Context manager yielding an observation with `.update(...)`; a no-op when tracing is disabled."""
    client = _client()
    if client is None:
        yield _NoopObservation()
        return
    try:
        cm = client.start_as_current_observation(name=name, as_type=as_type, **fields)
    except Exception:  # tracing must never break a request
        yield _NoopObservation()
        return
    with cm as observation:
        yield observation


def flush() -> None:
    client = _client()
    if client is not None:
        with contextlib.suppress(Exception):
            client.flush()
