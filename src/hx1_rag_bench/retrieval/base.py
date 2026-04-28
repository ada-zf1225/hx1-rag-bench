"""Retrieval interface and result schema.

Multi-hop QA benchmarks (HotpotQA distractor / MuSiQue / 2Wiki) ship a
candidate corpus *per sample* (a mix of supporting and distractor paragraphs).
Retrievers in this project therefore build one logical index per dataset and
expose `retrieve(sample_id, query, top_k)`, restricting each query to that
sample's own corpus. This matches the standard distractor-setting evaluation
in the literature.

For future open-domain (fullwiki) settings, a separate retriever subclass can
ignore `sample_id` and search a global index — the interface still holds.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from hx1_rag_bench.data import RAGSample


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved sentence with provenance and score."""

    title: str
    sent_idx: int
    sentence: str
    score: float


class Retriever(ABC):
    """Abstract retriever: build per-dataset, query per sample."""

    @abstractmethod
    def index(self, samples: list[RAGSample]) -> None:
        """Build / overwrite the index from a list of samples."""

    @abstractmethod
    def retrieve(
        self, sample_id: str, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Return top-k sentences for `query`, restricted to `sample_id`'s corpus."""

    def save(self, path: Path) -> None:
        """Persist the index to disk. Override if applicable.

        Default raises — useful for composite retrievers (e.g. Hybrid RRF)
        whose persistence is per sub-retriever and decided by the caller.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.save() not implemented; persist sub-retrievers individually"
        )

    @classmethod
    def load(cls, path: Path) -> Retriever:
        """Reconstruct a retriever from a previously-saved index. Override if applicable."""
        raise NotImplementedError(f"{cls.__name__}.load() not implemented")

    def release(self) -> None:
        """Release any GPU resources (model weights, embeddings buffers).

        No-op by default. GPU-backed retrievers (e.g. dense embedders) should
        override to free the encoder before downstream stages (vLLM) load.
        """
        return None


__all__ = ["RetrievalResult", "Retriever"]
