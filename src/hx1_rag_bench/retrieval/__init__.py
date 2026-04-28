"""Retrieval backends: BM25 (M2), BGE-M3 dense (M4), Hybrid RRF (M4)."""
from hx1_rag_bench.retrieval.base import RetrievalResult, Retriever
from hx1_rag_bench.retrieval.bm25 import BM25Retriever

__all__ = ["BM25Retriever", "RetrievalResult", "Retriever"]
