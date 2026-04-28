"""Standard QA metrics: SQuAD-style EM / F1, retrieval Recall@k.

Answer normalization follows the SQuAD v1.1 / HotpotQA convention, applied in
this exact order to match official scripts:
    1. lowercase
    2. strip ASCII punctuation
    3. strip articles ("a" / "an" / "the")
    4. collapse whitespace

EM and F1 take an iterable of gold answers (e.g. `(answer, *answer_aliases)`)
and return the maximum over them.

Recall@k handles the paragraph-level sentinel `sent_idx == PARAGRAPH_LEVEL`
introduced by the data loader: those facts match purely on `title` (any
retrieved sentence under that title hits). Sentence-level facts require an
exact `(title, sent_idx)` match.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence

from hx1_rag_bench.data.loaders import PARAGRAPH_LEVEL, SupportingFact
from hx1_rag_bench.retrieval.base import RetrievalResult

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_WS_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lower → punct → articles → whitespace."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLE_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _f1_pair(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        # Both empty after normalization → exact match (1.0); else 0.0.
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, golds: Iterable[str]) -> float:
    """Exact Match: 1.0 iff normalized prediction equals any normalized gold."""
    pn = normalize_answer(prediction)
    return float(any(pn == normalize_answer(g) for g in golds))


def compute_f1(prediction: str, golds: Iterable[str]) -> float:
    """Token-level F1, taking max over gold candidates."""
    scores = [_f1_pair(prediction, g) for g in golds]
    return max(scores) if scores else 0.0


def compute_recall_at_k(
    retrieved: Sequence[RetrievalResult],
    supporting_facts: Sequence[SupportingFact],
    k: int | None = None,
) -> float:
    """Fraction of `supporting_facts` covered by the top-k retrieved sentences.

    For each gold fact:
        - `sent_idx == PARAGRAPH_LEVEL` (-1, MuSiQue paragraph-level): hit iff
          `title` appears in retrieved titles.
        - `sent_idx >= 0` (HotpotQA / 2Wiki sentence-level): hit iff the exact
          `(title, sent_idx)` pair appears in retrieved.

    Returns 0.0 when `supporting_facts` is empty (undefined recall) — callers
    should filter such samples from aggregates if that's undesired.
    """
    if not supporting_facts:
        return 0.0

    top = retrieved if k is None else list(retrieved)[:k]
    hit_titles = {r.title for r in top}
    hit_pairs = {(r.title, r.sent_idx) for r in top}

    hits = 0
    for sf in supporting_facts:
        if sf.sent_idx == PARAGRAPH_LEVEL:
            if sf.title in hit_titles:
                hits += 1
        elif (sf.title, sf.sent_idx) in hit_pairs:
            hits += 1
    return hits / len(supporting_facts)


__all__ = [
    "compute_em",
    "compute_f1",
    "compute_recall_at_k",
    "normalize_answer",
]
