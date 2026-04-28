"""BGE-M3 dense embedder retriever.

Uses `FlagEmbedding.BGEM3FlagModel` for fp16 sentence encoding. Indexes are
per-sample (matching the distractor setting); each sample's index is the
`(N, dim)` embedding matrix plus the parallel `(title, sent_idx, sentence)`
entries list. BGE-M3 returns unit-normalized vectors, so cosine similarity
is just a dot product.

The encoder model is loaded lazily on first encode() call. After indexing
and querying are done, callers should invoke `release()` to free GPU memory
before downstream stages (e.g. vLLM) load — see `pipeline.runner`.

Persistence: pickle (numpy arrays pickle natively).
"""
from __future__ import annotations

import gc
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hx1_rag_bench.data import RAGSample
from hx1_rag_bench.retrieval.base import RetrievalResult, Retriever

logger = logging.getLogger(__name__)


@dataclass
class _SampleIndex:
    entries: list[tuple[str, int, str]]  # (title, sent_idx, sentence)
    embeddings: np.ndarray  # (N, dim) fp32, unit-normalized


class BGEM3Retriever(Retriever):
    """Per-sample dense retriever powered by BGE-M3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 64,
        max_length: int = 512,
        use_fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self._model: Any = None
        self._per_sample: dict[str, _SampleIndex] = {}

    def __len__(self) -> int:
        return len(self._per_sample)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from FlagEmbedding import BGEM3FlagModel

        logger.info(
            "Loading BGE-M3 model: %s (use_fp16=%s)", self.model_name, self.use_fp16
        )
        self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)

    def _encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        out = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        return np.asarray(out["dense_vecs"], dtype=np.float32)

    def index(self, samples: list[RAGSample]) -> None:
        self._per_sample.clear()

        # Flatten all sentences across all samples for one big batch encode —
        # BGE-M3 on A100 is much more efficient with large batches.
        flat_entries: list[tuple[str, str, int, str]] = []
        for s in samples:
            for doc in s.context:
                for i, sent in enumerate(doc.sentences):
                    flat_entries.append((s.id, doc.title, i, sent))

        if not flat_entries:
            logger.warning("No sentences to index")
            return

        sentences = [e[3] for e in flat_entries]
        logger.info(
            "Encoding %d sentences across %d samples", len(sentences), len(samples)
        )
        embeddings = self._encode(sentences)

        grouped: dict[str, list[tuple[str, int, str, np.ndarray]]] = defaultdict(list)
        for entry, emb in zip(flat_entries, embeddings, strict=True):
            sample_id, title, sent_idx, sentence = entry
            grouped[sample_id].append((title, sent_idx, sentence, emb))

        for sample_id, items in grouped.items():
            entries = [(t, si, s) for t, si, s, _ in items]
            embs = np.stack([e for _, _, _, e in items])
            self._per_sample[sample_id] = _SampleIndex(entries=entries, embeddings=embs)

        logger.info(
            "Indexed %d samples (%d total sentences, dim=%d)",
            len(self._per_sample),
            len(flat_entries),
            embeddings.shape[1],
        )

    def retrieve(
        self, sample_id: str, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        if sample_id not in self._per_sample:
            raise KeyError(
                f"sample_id {sample_id!r} not in index "
                f"(indexed={len(self._per_sample)} samples)"
            )
        idx = self._per_sample[sample_id]
        q_emb = self._encode([query])[0]
        scores = idx.embeddings @ q_emb
        k = min(top_k, len(scores))
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
        payload = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "use_fp16": self.use_fp16,
            "per_sample": self._per_sample,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "Saved BGE-M3 index (%d samples) to %s", len(self._per_sample), path
        )

    @classmethod
    def load(cls, path: Path) -> BGEM3Retriever:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        retriever = cls(
            model_name=payload["model_name"],
            batch_size=payload["batch_size"],
            max_length=payload["max_length"],
            use_fp16=payload["use_fp16"],
        )
        retriever._per_sample = payload["per_sample"]
        logger.info(
            "Loaded BGE-M3 index (%d samples) from %s", len(retriever._per_sample), path
        )
        return retriever

    def release(self) -> None:
        if self._model is None:
            return
        del self._model
        self._model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Released BGE-M3 encoder")


__all__ = ["BGEM3Retriever"]
