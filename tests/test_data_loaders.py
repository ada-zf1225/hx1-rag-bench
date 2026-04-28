"""Tests for the unified RAG dataset loader.

These tests perform real downloads from HF Hub on first run (~10-50 MB total
across the three datasets at validation split). When HF Hub is unreachable
the affected dataset's tests are skipped rather than failed, so offline CI
stays green.
"""
from __future__ import annotations

import dataclasses

import pytest

from hx1_rag_bench.data import (
    PARAGRAPH_LEVEL,
    ContextDoc,
    RAGSample,
    SupportingFact,
    load_dataset,
)

_DATASETS = ["musique", "two_wiki", "hotpotqa"]


@pytest.fixture(scope="module", params=_DATASETS)
def small_set(request: pytest.FixtureRequest) -> list[RAGSample]:
    name = request.param
    try:
        return load_dataset(name, split="validation", max_samples=10, seed=42)
    except (ConnectionError, OSError) as e:
        pytest.skip(f"HF Hub unreachable for {name}: {e}")


def test_count(small_set: list[RAGSample]) -> None:
    assert len(small_set) == 10


def test_field_types(small_set: list[RAGSample]) -> None:
    for s in small_set:
        assert isinstance(s, RAGSample)
        assert isinstance(s.id, str) and s.id
        assert isinstance(s.question, str) and s.question
        assert isinstance(s.answer, str)
        assert isinstance(s.answer_aliases, tuple)
        assert all(isinstance(a, str) for a in s.answer_aliases)
        assert isinstance(s.context, tuple) and len(s.context) > 0
        assert all(isinstance(d, ContextDoc) for d in s.context)
        assert all(isinstance(d.sentences, tuple) for d in s.context)
        assert isinstance(s.supporting_facts, tuple)
        assert all(isinstance(sf, SupportingFact) for sf in s.supporting_facts)


def test_supporting_facts_consistency(small_set: list[RAGSample]) -> None:
    for s in small_set:
        title_to_doc = {d.title: d for d in s.context}
        assert s.supporting_facts, f"sample {s.id!r} has no supporting facts"
        for sf in s.supporting_facts:
            assert sf.title in title_to_doc, (
                f"supporting fact title {sf.title!r} not in context of {s.id!r}"
            )
            if sf.sent_idx != PARAGRAPH_LEVEL:
                doc = title_to_doc[sf.title]
                assert 0 <= sf.sent_idx < len(doc.sentences), (
                    f"sent_idx {sf.sent_idx} out of range for title "
                    f"{sf.title!r} (len={len(doc.sentences)})"
                )


def test_frozen(small_set: list[RAGSample]) -> None:
    s = small_set[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.id = "modified"  # type: ignore[misc]
