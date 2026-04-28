"""Tests for QA metrics: SQuAD-style EM/F1 and Recall@k.

Pure-Python tests (no GPU, no network) — exhaustive parametrized coverage of
normalization edge cases, alias/multi-gold scoring, and the paragraph-level
vs sentence-level Recall@k branches.
"""
from __future__ import annotations

import pytest

from hx1_rag_bench.data.loaders import PARAGRAPH_LEVEL, SupportingFact
from hx1_rag_bench.eval.metrics import (
    compute_em,
    compute_f1,
    compute_recall_at_k,
    normalize_answer,
)
from hx1_rag_bench.retrieval.base import RetrievalResult

# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Eiffel Tower", "eiffel tower"),
        ("the, eiffel tower!", "eiffel tower"),
        ("  An apple  ", "apple"),
        ("Don't", "dont"),
        ("U.S.A.", "usa"),
        ("", ""),
        ("THE", ""),
        ("a an the", ""),
        ("Albert Einstein (physicist)", "albert einstein physicist"),
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


# ---------------------------------------------------------------------------
# compute_em
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pred", "golds", "expected"),
    [
        ("Paris", ["Paris"], 1.0),
        ("The Paris", ["Paris"], 1.0),  # article stripped
        ("Paris.", ["paris"], 1.0),  # punctuation stripped
        ("USA", ["United States", "U.S.A."], 1.0),  # alias hits
        ("London", ["Paris"], 0.0),
        ("anything", [], 0.0),  # empty golds
        ("", [""], 1.0),  # both empty after norm
        ("  PARIS  ", ["paris"], 1.0),  # whitespace + case
    ],
)
def test_compute_em(pred: str, golds: list[str], expected: float) -> None:
    assert compute_em(pred, golds) == expected


# ---------------------------------------------------------------------------
# compute_f1
# ---------------------------------------------------------------------------


def test_f1_perfect_match() -> None:
    assert compute_f1("Paris", ["paris"]) == pytest.approx(1.0)


def test_f1_partial_overlap() -> None:
    # pred=[eiffel,tower] (2 toks), gold=[eiffel,tower,in,paris] (4 toks),
    # common=2 → P=1.0, R=0.5, F1 = 2*1*0.5 / (1+0.5) = 0.6667
    assert compute_f1("Eiffel Tower", ["The Eiffel Tower in Paris"]) == pytest.approx(
        2 * 1.0 * 0.5 / 1.5
    )


def test_f1_no_overlap() -> None:
    assert compute_f1("Paris", ["London"]) == 0.0


def test_f1_max_over_aliases() -> None:
    # "USA" normalizes to "usa"; alias "U.S.A." also normalizes to "usa" → F1=1.0
    assert compute_f1("USA", ["United States of America", "U.S.A."]) == pytest.approx(1.0)


def test_f1_empty_prediction() -> None:
    assert compute_f1("", ["Paris"]) == 0.0


def test_f1_empty_golds() -> None:
    assert compute_f1("anything", []) == 0.0


def test_f1_repeated_tokens_uses_multiset() -> None:
    # pred=[paris,paris,paris], gold=[paris]; multiset intersection = {paris:1}
    # P=1/3, R=1/1, F1 = 2*(1/3)*1 / (1/3+1) = 0.5
    assert compute_f1("paris paris paris", ["paris"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_recall_at_k
# ---------------------------------------------------------------------------


def _result(title: str, sent_idx: int, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        title=title, sent_idx=sent_idx, sentence=f"{title}-{sent_idx}", score=score
    )


def test_recall_paragraph_level_full_hit() -> None:
    facts = [
        SupportingFact(title="A", sent_idx=PARAGRAPH_LEVEL),
        SupportingFact(title="B", sent_idx=PARAGRAPH_LEVEL),
    ]
    retrieved = [_result("A", 0), _result("B", 3)]
    assert compute_recall_at_k(retrieved, facts) == 1.0


def test_recall_paragraph_level_partial() -> None:
    facts = [
        SupportingFact(title="A", sent_idx=PARAGRAPH_LEVEL),
        SupportingFact(title="B", sent_idx=PARAGRAPH_LEVEL),
    ]
    retrieved = [_result("A", 0), _result("C", 0)]
    assert compute_recall_at_k(retrieved, facts) == 0.5


def test_recall_sentence_level_pair_match() -> None:
    facts = [
        SupportingFact(title="A", sent_idx=2),
        SupportingFact(title="B", sent_idx=0),
    ]
    retrieved = [_result("A", 2), _result("A", 1)]  # A:2 hits, B:0 misses
    assert compute_recall_at_k(retrieved, facts) == 0.5


def test_recall_sentence_level_title_only_does_not_count() -> None:
    """Sentence-level fact requires (title, sent_idx) match, not just title."""
    facts = [SupportingFact(title="A", sent_idx=5)]
    retrieved = [_result("A", 1), _result("A", 2)]  # right title, wrong sent_idx
    assert compute_recall_at_k(retrieved, facts) == 0.0


def test_recall_top_k_truncation() -> None:
    facts = [SupportingFact(title="A", sent_idx=PARAGRAPH_LEVEL)]
    retrieved = [
        _result("X", 0, score=10),
        _result("Y", 0, score=5),
        _result("A", 0, score=1),  # A is at index 2
    ]
    assert compute_recall_at_k(retrieved, facts, k=2) == 0.0
    assert compute_recall_at_k(retrieved, facts, k=3) == 1.0


def test_recall_no_facts_returns_zero() -> None:
    assert compute_recall_at_k([_result("A", 0)], []) == 0.0


def test_recall_mixed_levels() -> None:
    """Sample with both paragraph- and sentence-level facts."""
    facts = [
        SupportingFact(title="P", sent_idx=PARAGRAPH_LEVEL),
        SupportingFact(title="Q", sent_idx=2),
    ]
    retrieved = [_result("P", 0), _result("Q", 1)]  # P hit (paragraph), Q miss (idx)
    assert compute_recall_at_k(retrieved, facts) == 0.5


def test_recall_k_none_uses_all_retrieved() -> None:
    facts = [SupportingFact(title="A", sent_idx=PARAGRAPH_LEVEL)]
    retrieved = [_result("X", 0), _result("Y", 0), _result("A", 0)]
    assert compute_recall_at_k(retrieved, facts, k=None) == 1.0
