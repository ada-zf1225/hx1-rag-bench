"""Retrieval backends: BM25 (M2), BGE-M3 dense (M4), Hybrid RRF (M4)."""
from hx1_rag_bench.retrieval.base import RetrievalResult, Retriever
from hx1_rag_bench.retrieval.bge_m3 import BGEM3Retriever
from hx1_rag_bench.retrieval.bm25 import BM25Retriever
from hx1_rag_bench.retrieval.rrf import HybridRRFRetriever

__all__ = [
    "BGEM3Retriever",
    "BM25Retriever",
    "HybridRRFRetriever",
    "RetrievalResult",
    "Retriever",
]
