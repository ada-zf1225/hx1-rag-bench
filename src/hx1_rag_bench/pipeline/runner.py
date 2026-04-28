"""End-to-end RAG evaluation pipeline.

`run_eval(cfg)` orchestrates:

    load_dataset → build retriever → per-sample retrieve → vLLM batch generate
    → score (EM / F1 / Recall@k) → persist run dir.

Each run lands in `<output_dir>/<run_id>/` with:

    config.json        full cfg snapshot (so the run is reproducible)
    metrics.json       aggregate scores + timing
    predictions.jsonl  one row per sample (question, retrieval, prediction, scores)
    meta.json          host / GPU / package versions

The runner does not write unit tests for itself — `rag-bench hello` already
serves as a vLLM smoke test, and `tests/test_metrics.py` covers the scoring
math. End-to-end correctness is verified by running this on a real GPU.
"""
from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hx1_rag_bench.config import AppConfig, RetrieverConfig
from hx1_rag_bench.data import load_dataset
from hx1_rag_bench.eval.metrics import compute_em, compute_f1, compute_recall_at_k
from hx1_rag_bench.inference.engine import VLLMEngine
from hx1_rag_bench.inference.prompts import build_rag_messages
from hx1_rag_bench.retrieval import (
    BGEM3Retriever,
    BM25Retriever,
    HybridRRFRetriever,
    RetrievalResult,
    Retriever,
)

logger = logging.getLogger(__name__)

_RECALL_K_LADDER: tuple[int, ...] = (1, 5, 10)


@dataclass
class EvalReport:
    """Aggregate result of one `run_eval()` call."""

    run_id: str
    run_dir: Path
    n_samples: int
    em: float
    f1: float
    recall_at_k: dict[int, float]
    timing: dict[str, float]


def _make_retriever(cfg: RetrieverConfig) -> Retriever:
    if cfg.backend == "bm25":
        return BM25Retriever()
    if cfg.backend == "bge_m3":
        return BGEM3Retriever(
            model_name=cfg.bge_model,
            batch_size=cfg.bge_batch_size,
            max_length=cfg.bge_max_length,
            use_fp16=cfg.bge_use_fp16,
        )
    if cfg.backend == "hybrid_rrf":
        return HybridRRFRetriever(
            sub_retrievers=[
                BM25Retriever(),
                BGEM3Retriever(
                    model_name=cfg.bge_model,
                    batch_size=cfg.bge_batch_size,
                    max_length=cfg.bge_max_length,
                    use_fp16=cfg.bge_use_fp16,
                ),
            ],
            weights=cfg.rrf_weights,
            k=cfg.rrf_k,
        )
    raise NotImplementedError(f"Retriever backend {cfg.backend!r} not implemented")


def _format_run_id(cfg: AppConfig) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_short = cfg.model.name.split("/")[-1]
    return f"{ts}_{cfg.dataset.name}_{cfg.retriever.backend}_{model_short}"


def _retrieved_to_strings(hits: list[RetrievalResult]) -> list[str]:
    """Format retrieved sentences as title-prefixed strings for the prompt."""
    return [f"{h.title}: {h.sentence}" for h in hits]


def _config_snapshot(cfg: AppConfig) -> dict[str, Any]:
    return json.loads(cfg.model_dump_json())


def _gpu_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "device_count": torch.cuda.device_count(),
        }
    except ImportError:
        return {"available": False}


def _meta_snapshot() -> dict[str, Any]:
    import importlib.metadata as md

    pkgs: dict[str, str | None] = {}
    for p in ("vllm", "transformers", "torch", "rank-bm25", "datasets"):
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "gpu": _gpu_info(),
        "packages": pkgs,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_eval(
    cfg: AppConfig,
    output_dir: Path | None = None,
) -> EvalReport:
    """Run a full RAG evaluation per `cfg` and persist results.

    Args:
        cfg: validated `AppConfig` (typically `AppConfig.from_yaml(...)`).
        output_dir: parent dir for the per-run subdir; defaults to
            `cfg.results_dir / "evaluations"`.
    """
    t_start = time.time()
    run_id = _format_run_id(cfg)
    out_root = (
        Path(output_dir) if output_dir is not None else cfg.results_dir / "evaluations"
    )
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    # --- Phase A: data + retrieval (CPU; BGE-M3 / Hybrid touch GPU here) --
    samples = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.split,
        max_samples=cfg.dataset.max_samples,
        seed=cfg.dataset.seed,
        cache_dir=cfg.dataset.cache_dir,
    )
    logger.info("Loaded %d samples from %s", len(samples), cfg.dataset.name)

    retriever = _make_retriever(cfg.retriever)
    t0 = time.time()
    retriever.index(samples)
    t_index = time.time() - t0

    top_k = cfg.retriever.top_k
    t0 = time.time()
    all_retrieved: list[list[RetrievalResult]] = [
        retriever.retrieve(s.id, s.question, top_k=top_k) for s in samples
    ]
    t_retrieve = time.time() - t0
    logger.info(
        "Indexed in %.2fs, retrieved top-%d for %d queries in %.2fs",
        t_index, top_k, len(samples), t_retrieve,
    )

    # Free any retriever-side GPU memory (e.g. BGE-M3 encoder) before the LLM
    # claims its share of the device. No-op for CPU-only retrievers.
    retriever.release()

    # --- Phase B: load engine + format prompts (needs tokenizer) -----------
    engine = VLLMEngine(cfg.model)
    try:
        engine.load()

        prompts = [
            engine.format_chat(build_rag_messages(s.question, _retrieved_to_strings(hits)))
            for s, hits in zip(samples, all_retrieved, strict=True)
        ]

        # --- Phase C: batch generate ---------------------------------------
        t0 = time.time()
        batch = engine.generate(prompts)
        t_generate = time.time() - t0
        logger.info(
            "Generated %d completions in %.2fs (%.1f tok/s aggregate)",
            len(batch.outputs), t_generate, batch.aggregate_throughput_tps,
        )
    finally:
        engine.shutdown()

    # --- Phase D: score ----------------------------------------------------
    rows: list[dict[str, Any]] = []
    em_sum = 0.0
    f1_sum = 0.0
    recall_sums: dict[int, float] = dict.fromkeys(_RECALL_K_LADDER, 0.0)

    for s, hits, gen in zip(samples, all_retrieved, batch.outputs, strict=True):
        golds = [s.answer, *s.answer_aliases]
        em = compute_em(gen.text, golds)
        f1 = compute_f1(gen.text, golds)
        recalls = {
            k: compute_recall_at_k(hits, s.supporting_facts, k=k)
            for k in _RECALL_K_LADDER
        }

        em_sum += em
        f1_sum += f1
        for k, v in recalls.items():
            recall_sums[k] += v

        rows.append({
            "id": s.id,
            "question": s.question,
            "answer": s.answer,
            "answer_aliases": list(s.answer_aliases),
            "supporting_facts": [
                {"title": sf.title, "sent_idx": sf.sent_idx} for sf in s.supporting_facts
            ],
            "retrieved": [
                {
                    "title": h.title,
                    "sent_idx": h.sent_idx,
                    "sentence": h.sentence,
                    "score": h.score,
                }
                for h in hits
            ],
            "prediction": gen.text,
            "em": em,
            "f1": f1,
            **{f"recall_at_{k}": recalls[k] for k in _RECALL_K_LADDER},
            "prompt_tokens": gen.prompt_token_count,
            "output_tokens": gen.output_token_count,
            "finish_reason": gen.finish_reason,
        })

    n = len(samples)
    em_mean = em_sum / n
    f1_mean = f1_sum / n
    recall_means = {k: v / n for k, v in recall_sums.items()}

    # --- Phase E: persist --------------------------------------------------
    timing = {
        "index_s": t_index,
        "retrieve_s": t_retrieve,
        "generate_s": t_generate,
        "total_s": time.time() - t_start,
        "throughput_tok_per_s": batch.aggregate_throughput_tps,
    }
    metrics = {
        "run_id": run_id,
        "n_samples": n,
        "em": em_mean,
        "f1": f1_mean,
        **{f"recall_at_{k}": recall_means[k] for k in _RECALL_K_LADDER},
        "timing": timing,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(
        json.dumps(_config_snapshot(cfg), indent=2, default=str)
    )
    (run_dir / "meta.json").write_text(json.dumps(_meta_snapshot(), indent=2))
    _write_jsonl(run_dir / "predictions.jsonl", rows)

    logger.info(
        "Done: EM=%.3f F1=%.3f Recall@5=%.3f → %s",
        em_mean, f1_mean, recall_means[5], run_dir,
    )

    return EvalReport(
        run_id=run_id,
        run_dir=run_dir,
        n_samples=n,
        em=em_mean,
        f1=f1_mean,
        recall_at_k=recall_means,
        timing=timing,
    )


__all__ = ["EvalReport", "run_eval"]
