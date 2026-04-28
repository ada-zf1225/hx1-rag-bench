"""BM25 retriever using `rank_bm25.BM25Okapi`, sentence-grained.

Each sample's context is flattened to (title, sent_idx, sentence) entries and
wrapped in a per-sample BM25Okapi instance. At query time, the per-sample
index is consulted directly — distractor-setting datasets like HotpotQA /
MuSiQue / 2Wiki already ship a small (~50 sentences) candidate pool per
sample, so per-sample BM25 is fast and avoids cross-sample contamination.

Tokenization is the standard "lowercase + non-alphanumeric split" baseline
used by most BM25 multi-hop QA evaluations.
"""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from hx1_rag_bench.data import RAGSample
from hx1_rag_bench.retrieval.base import RetrievalResult, Retriever

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _SampleIndex:
    """Per-sample flat sentence index + fitted BM25Okapi."""

    entries: list[tuple[str, int, str]]  # (title, sent_idx, sentence)
    bm25: BM25Okapi


class BM25Retriever(Retriever):
    """Per-sample sentence-level BM25 retriever.

    The index is a `dict[sample_id, _SampleIndex]` and is fully picklable.
    Persistence format (pickle) carries the BM25Okapi state plus our entry
    list — a fresh `BM25Retriever` reconstructs identically via `.load(path)`.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._per_sample: dict[str, _SampleIndex] = {}

    def __len__(self) -> int:
        return len(self._per_sample)

    def index(self, samples: list[RAGSample]) -> None:
        self._per_sample.clear()
        for s in samples:
            entries: list[tuple[str, int, str]] = [
                (doc.title, i, sent)
                for doc in s.context
                for i, sent in enumerate(doc.sentences)
            ]
            if not entries:
                logger.warning("Skipping sample %s: empty context", s.id)
                continue
            tokenized = [_tokenize(sent) for _, _, sent in entries]
            bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
            self._per_sample[s.id] = _SampleIndex(entries=entries, bm25=bm25)
        logger.info("Indexed %d samples", len(self._per_sample))

    def retrieve(
        self, sample_id: str, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        if sample_id not in self._per_sample:
            raise KeyError(
                f"sample_id {sample_id!r} not in index "
                f"(indexed={len(self._per_sample)} samples)"
            )
        idx = self._per_sample[sample_id]
        scores = idx.bm25.get_scores(_tokenize(query))
        k = min(top_k, len(scores))
        # argpartition is O(n) for top-k; sort the small slice for stable order
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [
            RetrievalResult(
                title=idx.entries[i][0],
                sent_idx=idx.entries[i][1],
                sentence=idx.entries[i][2],
                score=float(scores[i]),
            )
            for i in top
        ]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "k1": self.k1,
            "b": self.b,
            "per_sample": self._per_sample,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved BM25 index (%d samples) to %s", len(self._per_sample), path)

    @classmethod
    def load(cls, path: Path) -> BM25Retriever:
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        retriever = cls(k1=payload["k1"], b=payload["b"])
        retriever._per_sample = payload["per_sample"]
        logger.info(
            "Loaded BM25 index (%d samples) from %s", len(retriever._per_sample), path
        )
        return retriever


__all__ = ["BM25Retriever"]
