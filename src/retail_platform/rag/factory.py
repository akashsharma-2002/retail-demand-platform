import os

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from retail_platform.config import ROOT, Settings
from retail_platform.rag.chunking import load_corpus
from retail_platform.rag.search import (
    CachedSearcher,
    CrossEncoderReranker,
    HybridSearcher,
    RedisCache,
    SentenceTransformerEmbedder,
)

POLICY_DIR = ROOT / "knowledge" / "policies"


def build_searcher(settings: Settings, engine: Engine | None = None):
    """RP_RAG_MODE: bm25 (no models), hybrid (BGE-M3), hybrid_rerank (BGE-M3 + reranker, default)."""
    mode = os.getenv("RP_RAG_MODE", "hybrid_rerank")
    embedder = SentenceTransformerEmbedder(settings.embedding_model) if mode != "bm25" else None
    reranker = CrossEncoderReranker(settings.reranker_model) if mode == "hybrid_rerank" else None
    cache = RedisCache(settings.redis_url)
    if (
        engine is not None
        and embedder is not None
        and engine.dialect.name == "postgresql"
        and inspect(engine).has_table("policy_chunks")
    ):
        from retail_platform.rag.pg_store import PgHybridSearcher

        return CachedSearcher(PgHybridSearcher(engine, embedder, reranker), cache)
    return CachedSearcher(HybridSearcher(load_corpus(POLICY_DIR), embedder, reranker), cache)
