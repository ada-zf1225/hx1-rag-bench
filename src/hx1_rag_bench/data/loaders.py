"""Multi-hop QA dataset loaders.

Three datasets are supported, all normalized to a unified `RAGSample` schema:

- MuSiQue (`dgslibisey/MuSiQue`): 2-4 hop reasoning, paragraph-level supporting
  fact annotation. Sentences are obtained by regex-splitting `paragraph_text`.
- 2WikiMultihopQA (`framolfese/2WikiMultihopQA`): HotpotQA-aligned mirror of the
  Alab-NII 2Wiki dataset. Sentence-level supporting facts.
- HotpotQA distractor (`hotpotqa/hotpot_qa`): sentence-level supporting facts.

For paragraph-level annotation (MuSiQue) we emit one `SupportingFact` with
`sent_idx == PARAGRAPH_LEVEL` (-1) per supporting paragraph as a sentinel
meaning "any sentence under this title counts as a hit". Downstream Recall@k
must respect this convention.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

DatasetName = Literal["musique", "two_wiki", "hotpotqa"]

PARAGRAPH_LEVEL = -1


@dataclass(frozen=True)
class ContextDoc:
    """A candidate document: a titled paragraph split into sentences."""

    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class SupportingFact:
    """Ground-truth fact reference used for Recall@k.

    `sent_idx == PARAGRAPH_LEVEL` (-1) is a sentinel for paragraph-level
    annotation (e.g. MuSiQue). Sentence-level datasets use `sent_idx >= 0`.
    """

    title: str
    sent_idx: int


@dataclass(frozen=True)
class RAGSample:
    """Unified multi-hop QA sample shape used across retrievers / pipelines."""

    id: str
    question: str
    answer: str
    answer_aliases: tuple[str, ...]
    supporting_facts: tuple[SupportingFact, ...]
    context: tuple[ContextDoc, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


_DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "musique": {"hf_id": "dgslibisey/MuSiQue", "config": None},
    "two_wiki": {"hf_id": "framolfese/2WikiMultihopQA", "config": None},
    "hotpotqa": {"hf_id": "hotpotqa/hotpot_qa", "config": "distractor"},
}

# Permissive sentence boundary: split on .!? followed by whitespace.
# Over-splits abbreviations (Dr., U.S.) but acceptable for BM25 indexing and
# Recall@k since MuSiQue uses paragraph-level supporting-fact sentinel.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> tuple[str, ...]:
    text = (text or "").strip()
    if not text:
        return ()
    return tuple(p.strip() for p in _SENT_BOUNDARY.split(text) if p.strip())


def _normalize_musique(row: dict[str, Any]) -> RAGSample:
    docs: list[ContextDoc] = []
    sup: list[SupportingFact] = []
    for para in row["paragraphs"]:
        title = para["title"]
        sents = _split_sentences(para["paragraph_text"])
        docs.append(ContextDoc(title=title, sentences=sents))
        if para.get("is_supporting"):
            sup.append(SupportingFact(title=title, sent_idx=PARAGRAPH_LEVEL))
    aliases = row.get("answer_aliases") or []
    return RAGSample(
        id=str(row["id"]),
        question=row["question"],
        answer=row["answer"],
        answer_aliases=tuple(aliases),
        supporting_facts=tuple(sup),
        context=tuple(docs),
        metadata={"source": "musique"},
    )


def _normalize_hotpotqa_like(row: dict[str, Any], source: str) -> RAGSample:
    """Shared normalizer for HotpotQA-format rows (HotpotQA + 2Wiki mirror)."""
    ctx = row["context"]  # {"title": [...], "sentences": [[...], ...]}
    docs = tuple(
        ContextDoc(title=t, sentences=tuple(s))
        for t, s in zip(ctx["title"], ctx["sentences"], strict=True)
    )

    sf = row["supporting_facts"]  # {"title": [...], "sent_id": [...]}
    sup = tuple(
        SupportingFact(title=t, sent_idx=int(i))
        for t, i in zip(sf["title"], sf["sent_id"], strict=True)
    )

    sample_id = row.get("_id") or row.get("id")
    return RAGSample(
        id=str(sample_id),
        question=row["question"],
        answer=row["answer"],
        answer_aliases=(),
        supporting_facts=sup,
        context=docs,
        metadata={
            "source": source,
            "type": row.get("type", ""),
            "level": row.get("level", ""),
        },
    )


def _normalize_two_wiki(row: dict[str, Any]) -> RAGSample:
    return _normalize_hotpotqa_like(row, source="two_wiki")


def _normalize_hotpotqa(row: dict[str, Any]) -> RAGSample:
    return _normalize_hotpotqa_like(row, source="hotpotqa")


_NORMALIZERS = {
    "musique": _normalize_musique,
    "two_wiki": _normalize_two_wiki,
    "hotpotqa": _normalize_hotpotqa,
}


def load_dataset(
    name: DatasetName,
    split: str = "validation",
    max_samples: int | None = None,
    seed: int = 42,
    cache_dir: Path | str | None = None,
) -> list[RAGSample]:
    """Load and normalize a multi-hop QA dataset to a list of `RAGSample`.

    Args:
        name: dataset key (`musique` / `two_wiki` / `hotpotqa`).
        split: HF split. Defaults to "validation".
        max_samples: If set, seeded-shuffle then truncate to this many samples.
        seed: RNG seed used by `Dataset.shuffle`.
        cache_dir: HF datasets cache location. Defaults to `data/raw/`.

    Returns:
        List of `RAGSample` (length <= `max_samples` if specified).
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}; expected one of {list(_DATASET_REGISTRY)}"
        )

    from datasets import load_dataset as hf_load_dataset

    spec = _DATASET_REGISTRY[name]
    cache_path = Path(cache_dir) if cache_dir is not None else Path("data/raw")
    cache_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading dataset name=%s hf_id=%s split=%s cache_dir=%s",
        name,
        spec["hf_id"],
        split,
        cache_path,
    )

    kwargs: dict[str, Any] = {"cache_dir": str(cache_path)}
    if spec["config"] is not None:
        ds = hf_load_dataset(spec["hf_id"], spec["config"], split=split, **kwargs)
    else:
        ds = hf_load_dataset(spec["hf_id"], split=split, **kwargs)

    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    normalize = _NORMALIZERS[name]
    samples = [normalize(row) for row in ds]
    logger.info("Loaded %d samples from %s", len(samples), name)
    return samples


__all__ = [
    "PARAGRAPH_LEVEL",
    "ContextDoc",
    "DatasetName",
    "RAGSample",
    "SupportingFact",
    "load_dataset",
]
