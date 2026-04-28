"""Tests for HybridRRFRetriever — pure CPU.

Sub-retrievers are stubbed with canned outputs so we can verify the RRF
formula, dedup logic, weight handling, and lifecycle (index / release / save)
without touching FlagEmbedding or rank_bm25.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hx1_rag_bench.data import RAGSample
from hx1_rag_bench.retrieval.base import RetrievalResult, Retriever
from hx1_rag_bench.retrieval.rrf import HybridRRFRetriever


class _CannedRetriever(Retriever):
    """Returns a fixed list of RetrievalResults; tracks lifecycle calls."""

    def __init__(self, canned: list[RetrievalResult]) -> None:
        self._canned = canned
        self.indexed_with: list[RAGSample] | None = None
        self.released = False

    def index(self, samples: list[RAGSample]) -> None:
        self.indexed_with = list(samples)

    def retrieve(
        self, sample_id: str, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        return self._canned[:top_k]

    def release(self) -> None:
        self.released = True


def _r(title: str, sent_idx: int, sentence: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        title=title,
        sent_idx=sent_idx,
        sentence=sentence if sentence is not None else f"{title}-{sent_idx}",
        score=1.0,
    )


# ---------------------------------------------------------------------------
# RRF formula
# ---------------------------------------------------------------------------


def test_single_retriever_preserves_order_and_scores() -> None:
    sub = _CannedRetriever([_r("A", 0), _r("B", 0), _r("C", 0)])
    h = HybridRRFRetriever([sub], k=60)
    out = h.retrieve("any", "any", top_k=3)

    assert [(o.title, o.sent_idx) for o in out] == [("A", 0), ("B", 0), ("C", 0)]
    assert out[0].score == pytest.approx(1 / 61)
    assert out[1].score == pytest.approx(1 / 62)
    assert out[2].score == pytest.approx(1 / 63)


def test_two_retrievers_agreement_sums_scores() -> None:
    a = _CannedRetriever([_r("A", 0), _r("B", 0)])
    b = _CannedRetriever([_r("A", 0), _r("C", 0)])
    h = HybridRRFRetriever([a, b], k=60)
    out = h.retrieve("any", "any", top_k=3)

    # A appears at rank 1 in both → 1/61 + 1/61 = 2/61
    assert out[0].title == "A"
    assert out[0].score == pytest.approx(2 / 61)
    # B and C each appear once at rank 2 → 1/62
    rest = sorted((o.title, round(o.score, 8)) for o in out[1:])
    assert rest == sorted([("B", round(1 / 62, 8)), ("C", round(1 / 62, 8))])


def test_weights_scale_contributions() -> None:
    a = _CannedRetriever([_r("A", 0), _r("B", 0)])  # weight 0.7
    b = _CannedRetriever([_r("B", 0), _r("A", 0)])  # weight 0.3
    h = HybridRRFRetriever([a, b], weights=[0.7, 0.3], k=60)
    out = h.retrieve("any", "any", top_k=2)

    # A: 0.7/61 + 0.3/62; B: 0.7/62 + 0.3/61. Heavier retriever ranks A first → A wins.
    assert out[0].title == "A"
    assert out[1].title == "B"
    assert out[0].score == pytest.approx(0.7 / 61 + 0.3 / 62)
    assert out[1].score == pytest.approx(0.7 / 62 + 0.3 / 61)


def test_top_k_truncation_after_fusion() -> None:
    a = _CannedRetriever([_r("A", 0), _r("B", 0), _r("C", 0)])
    b = _CannedRetriever([_r("D", 0), _r("E", 0), _r("F", 0)])
    h = HybridRRFRetriever([a, b], k=60)
    out = h.retrieve("any", "any", top_k=2)
    assert len(out) == 2
    # All items have unique titles; top-2 are the rank-1 docs from each sub
    assert {o.title for o in out} == {"A", "D"}


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_dedup_by_title_sent_idx_sentence() -> None:
    """Identical (title, sent_idx, sentence) triples fuse to one entry."""
    a = _CannedRetriever([_r("A", 0, "alpha"), _r("B", 0, "beta")])
    b = _CannedRetriever([_r("A", 0, "alpha"), _r("C", 0, "gamma")])
    h = HybridRRFRetriever([a, b], k=60)
    out = h.retrieve("any", "any", top_k=10)

    assert len(out) == 3  # A merged, B + C unique
    assert {(o.title, o.sentence) for o in out} == {
        ("A", "alpha"),
        ("B", "beta"),
        ("C", "gamma"),
    }


def test_same_title_different_sentences_are_kept_separate() -> None:
    """Same title + sent_idx but distinct sentence text → two entries (MuSiQue case)."""
    a = _CannedRetriever([_r("X", 0, "version_a")])
    b = _CannedRetriever([_r("X", 0, "version_b")])
    h = HybridRRFRetriever([a, b], k=60)
    out = h.retrieve("any", "any", top_k=10)

    assert len(out) == 2
    assert {o.sentence for o in out} == {"version_a", "version_b"}


# ---------------------------------------------------------------------------
# Lifecycle: index / release / save
# ---------------------------------------------------------------------------


def test_index_dispatches_to_subs() -> None:
    a = _CannedRetriever([])
    b = _CannedRetriever([])
    h = HybridRRFRetriever([a, b])
    h.index([])
    assert a.indexed_with == [] and b.indexed_with == []


def test_release_cascades() -> None:
    a = _CannedRetriever([])
    b = _CannedRetriever([])
    h = HybridRRFRetriever([a, b])
    h.release()
    assert a.released and b.released


def test_save_raises_with_helpful_message(tmp_path: Path) -> None:
    h = HybridRRFRetriever([_CannedRetriever([])])
    with pytest.raises(NotImplementedError, match="sub-retrievers"):
        h.save(tmp_path / "foo.pkl")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_init_rejects_empty_sub_retrievers() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRRFRetriever(sub_retrievers=[])


def test_init_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="weights length"):
        HybridRRFRetriever(
            sub_retrievers=[_CannedRetriever([])],
            weights=[0.5, 0.5],
        )


def test_init_default_uniform_weights() -> None:
    h = HybridRRFRetriever([_CannedRetriever([]), _CannedRetriever([])])
    assert h.weights == [1.0, 1.0]
