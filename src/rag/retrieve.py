from __future__ import annotations
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from rag.ingest import Chunk

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _reranker


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    metadata: dict
    score: float


def bm25_retrieve(query: str, corpus: list[Chunk], k: int = 10) -> list[RetrievalResult]:
    tokenized = [c.text.lower().split() for c in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [
        RetrievalResult(corpus[i].chunk_id, corpus[i].text, corpus[i].metadata, float(s))
        for i, s in ranked if s > 0
    ]


def dense_retrieve(query: str, collection: chromadb.Collection, k: int = 10) -> list[RetrievalResult]:
    res = collection.query(query_texts=[query], n_results=k)
    return [
        RetrievalResult(
            res["ids"][0][i],
            res["documents"][0][i],
            res["metadatas"][0][i],
            1.0 - res["distances"][0][i],
        )
        for i in range(len(res["ids"][0]))
    ]


def hybrid_rrf(
    bm25_results: list[RetrievalResult],
    dense_results: list[RetrievalResult],
    k: int = 60,
) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    metas: dict[str, dict] = {}
    for rank, r in enumerate(bm25_results):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1 / (k + rank + 1)
        texts[r.chunk_id] = r.text; metas[r.chunk_id] = r.metadata
    for rank, r in enumerate(dense_results):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1 / (k + rank + 1)
        texts[r.chunk_id] = r.text; metas[r.chunk_id] = r.metadata
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [RetrievalResult(cid, texts[cid], metas[cid], sc) for cid, sc in ranked]


def rerank(query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
    if not results:
        return []
    pairs = [[query, r.text] for r in results]
    scores = _get_reranker().predict(pairs)
    if isinstance(scores, float):
        scores = [scores]
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)[:top_k]
    return [RetrievalResult(r.chunk_id, r.text, r.metadata, float(s)) for s, r in ranked]
