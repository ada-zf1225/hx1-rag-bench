"""Hybrid retrieval via Reciprocal Rank Fusion.

Given K sub-retrievers and per-retriever weights w_i, the fused score for
document d is

    score(d) = Σ_i  w_i / (k + rank_i(d))

where `rank_i(d)` is d's 1-indexed position in retriever i's output (or
omitted if d is absent from that retriever's top-N), and `k` is the RRF
smoothing constant (typical 60). Default weights are uniform.

Sub-retrievers handle their own indexing and persistence; HybridRRFRetriever
just dispatches and fuses. `release()` cascades to sub-retrievers — useful
for freeing GPU embedders before downstream stages load.

Save/load are deliberately not implemented here: callers should persist each
sub-retriever individually (their formats differ — pickle for BM25, pickle
for BGE-M3 — and reuniting them under a single blob would be brittle).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from hx1_rag_bench.data import RAGSample
from hx1_rag_bench.retrieval.base import RetrievalResult, Retriever

logger = logging.getLogger(__name__)


class HybridRRFRetriever(Retriever):
    """Reciprocal Rank Fusion over multiple sub-retrievers."""

    def __init__(
        self,
        sub_retrievers: Sequence[Retriever],
        weights: Sequence[float] | None = None,
        k: int = 60,
    ) -> None:
        if not sub_retrievers:
            raise ValueError("HybridRRFRetriever requires at least one sub-retriever")
        if weights is None:
            weights = [1.0] * len(sub_retrievers)
        if len(weights) != len(sub_retrievers):
            raise ValueError(
                f"weights length {len(weights)} != sub_retrievers length "
                f"{len(sub_retrievers)}"
            )
        self.sub_retrievers: list[Retriever] = list(sub_retrievers)
        self.weights: list[float] = list(weights)
        self.k = k

    def index(self, samples: list[RAGSample]) -> None:
        for r in self.sub_retrievers:
            r.index(samples)

    def retrieve(
        self, sample_id: str, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        # Each sub contributes a generous top-N so fusion has signal beyond top_k.
        fusion_topn = max(top_k * 4, 50)
        rrf_scores: dict[tuple[str, int, str], float] = {}
        canonical: dict[tuple[str, int, str], RetrievalResult] = {}

        for sub, weight in zip(self.sub_retrievers, self.weights, strict=True):
            results = sub.retrieve(sample_id, query, top_k=fusion_topn)
            for rank0, r in enumerate(results):
                key = (r.title, r.sent_idx, r.sentence)
                contribution = weight / (self.k + rank0 + 1)  # rank is 1-indexed
                rrf_scores[key] = rrf_scores.get(key, 0.0) + contribution
                canonical.setdefault(key, r)

        sorted_keys = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            RetrievalResult(
                title=key[0],
                sent_idx=key[1],
                sentence=canonical[key].sentence,
                score=score,
            )
            for key, score in sorted_keys[:top_k]
        ]

    def release(self) -> None:
        for r in self.sub_retrievers:
            r.release()


__all__ = ["HybridRRFRetriever"]
