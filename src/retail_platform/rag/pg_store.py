"""PostgreSQL backend for policy search: pgvector (HNSW, cosine) + ParadeDB pg_search BM25."""

from collections.abc import Sequence

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

from retail_platform.rag.chunking import Chunk
from retail_platform.rag.search import CANDIDATES, RERANK_TOP, Embedder, Reranker, SearchHit, rrf

DDL = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_search",
    """CREATE TABLE IF NOT EXISTS policy_chunks (
        id serial PRIMARY KEY, chunk_id text UNIQUE NOT NULL, title text NOT NULL, section text NOT NULL,
        body text NOT NULL, search_text text NOT NULL, trusted boolean NOT NULL, embedding vector(1024))""",
    "CREATE INDEX IF NOT EXISTS policy_chunks_hnsw ON policy_chunks USING hnsw (embedding vector_cosine_ops)",
    """CREATE INDEX IF NOT EXISTS policy_chunks_bm25 ON policy_chunks
        USING bm25 (id, search_text) WITH (key_field='id')""",
]


def index_chunks(engine: Engine, chunks: Sequence[Chunk], embedder: Embedder) -> None:
    vectors = embedder.encode([c.search_text for c in chunks])
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))
        conn.execute(text("TRUNCATE policy_chunks"))
        for chunk, vec in zip(chunks, vectors, strict=True):
            conn.execute(
                text("""INSERT INTO policy_chunks (chunk_id, title, section, body, search_text, trusted, embedding)
                        VALUES (:cid, :title, :section, :body, :st, :trusted, CAST(:emb AS vector))"""),
                {
                    "cid": chunk.chunk_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "body": chunk.text,
                    "st": chunk.search_text,
                    "trusted": chunk.trusted,
                    "emb": _vec(vec),
                },
            )


def _vec(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class PgHybridSearcher:
    def __init__(self, engine: Engine, embedder: Embedder, reranker: Reranker | None = None):
        self.engine, self.embedder, self.reranker = engine, embedder, reranker

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid_rerank") -> list[SearchHit]:
        q = _vec(self.embedder.encode([query])[0])
        with self.engine.connect() as conn:
            dense = (
                conn.execute(
                    text("SELECT id FROM policy_chunks ORDER BY embedding <=> CAST(:q AS vector) LIMIT :n"),
                    {"q": q, "n": CANDIDATES},
                )
                .scalars()
                .all()
            )
            sparse = (
                conn.execute(
                    text(
                        "SELECT id FROM policy_chunks WHERE search_text @@@ :query ORDER BY paradedb.score(id) DESC LIMIT :n"
                    ),
                    {"query": query, "n": CANDIDATES},
                )
                .scalars()
                .all()
            )
            order = rrf([sparse, dense])[:RERANK_TOP]
            if not order:
                return []
            rows = {
                r.id: r
                for r in conn.execute(
                    text(
                        "SELECT id, chunk_id, title, section, body, search_text, trusted FROM policy_chunks WHERE id = ANY(:ids)"
                    ),
                    {"ids": order},
                )
            }
        scores = [1.0 / (n + 1) for n in range(len(order))]
        if self.reranker is not None:
            s = self.reranker.score(query, [rows[i].search_text for i in order])
            ranked = np.argsort(-s)
            order, scores = [order[j] for j in ranked], [float(s[j]) for j in ranked]
        return [
            SearchHit(rows[i].chunk_id, rows[i].title, rows[i].section, rows[i].body, rows[i].trusted, sc)
            for i, sc in zip(order[:top_k], scores[:top_k], strict=True)
        ]
