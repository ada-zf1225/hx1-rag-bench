"""BM25 retriever tests on real MuSiQue samples.

Builds an index from 10 MuSiQue validation samples, then exercises retrieval,
title-level recall@5 sanity, and pickle round-trip. Skips gracefully when HF
Hub is unreachable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hx1_rag_bench.data import RAGSample, load_dataset
from hx1_rag_bench.retrieval import BM25Retriever, RetrievalResult


@pytest.fixture(scope="module")
def musique_samples() -> list[RAGSample]:
    try:
        return load_dataset("musique", split="validation", max_samples=10, seed=42)
    except (ConnectionError, OSError) as e:
        pytest.skip(f"HF Hub unreachable: {e}")


@pytest.fixture(scope="module")
def indexed_retriever(musique_samples: list[RAGSample]) -> BM25Retriever:
    r = BM25Retriever()
    r.index(musique_samples)
    return r


def test_index_size(
    indexed_retriever: BM25Retriever, musique_samples: list[RAGSample]
) -> None:
    assert len(indexed_retriever) == len(musique_samples)


def test_retrieve_returns_top_k(
    indexed_retriever: BM25Retriever, musique_samples: list[RAGSample]
) -> None:
    s = musique_samples[0]
    results = indexed_retriever.retrieve(s.id, s.question, top_k=5)
    assert len(results) == 5
    assert all(isinstance(r, RetrievalResult) for r in results)
    # scores should be monotonically non-increasing (top-k sorted desc)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # Each result must be a real (title, sent_idx, sentence) triple from the
    # sample's flattened corpus. Use the triple — a sample may carry several
    # paragraphs with the same title (different Wikipedia excerpts), so
    # (title, sent_idx) alone is not unique.
    flat_corpus = {
        (doc.title, i, sent)
        for doc in s.context
        for i, sent in enumerate(doc.sentences)
    }
    for r in results:
        assert (r.title, r.sent_idx, r.sentence) in flat_corpus


def test_title_level_recall_at_5(
    indexed_retriever: BM25Retriever, musique_samples: list[RAGSample]
) -> None:
    """At least 60% of samples should have a supporting-paragraph title in top-5.

    Loose threshold — BM25 on MuSiQue distractor with simple whitespace
    tokenization typically hits 70-90% title-level recall@5. 60% gives margin
    for the 10-sample noise floor.
    """
    hits = 0
    for s in musique_samples:
        results = indexed_retriever.retrieve(s.id, s.question, top_k=5)
        retrieved_titles = {r.title for r in results}
        sf_titles = {sf.title for sf in s.supporting_facts}
        if retrieved_titles & sf_titles:
            hits += 1
    recall = hits / len(musique_samples)
    assert recall >= 0.6, f"title-level recall@5 = {recall:.2f} (expected >= 0.60)"


def test_unknown_sample_id_raises(indexed_retriever: BM25Retriever) -> None:
    with pytest.raises(KeyError):
        indexed_retriever.retrieve("does-not-exist", "any query", top_k=5)


def test_save_load_roundtrip(
    indexed_retriever: BM25Retriever,
    musique_samples: list[RAGSample],
    tmp_path: Path,
) -> None:
    path = tmp_path / "bm25_musique.pkl"
    indexed_retriever.save(path)
    assert path.exists() and path.stat().st_size > 0

    reloaded = BM25Retriever.load(path)
    assert len(reloaded) == len(indexed_retriever)
    assert reloaded.k1 == indexed_retriever.k1
    assert reloaded.b == indexed_retriever.b

    s = musique_samples[0]
    expected = indexed_retriever.retrieve(s.id, s.question, top_k=5)
    actual = reloaded.retrieve(s.id, s.question, top_k=5)
    assert [(r.title, r.sent_idx) for r in actual] == [
        (r.title, r.sent_idx) for r in expected
    ]
    assert all(
        a.score == pytest.approx(e.score) for a, e in zip(actual, expected, strict=True)
    )
