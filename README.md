# hx1-rag-bench

Industrial-grade RAG inference benchmark and diagnosis framework, optimized for Imperial HX1 A100 80GB cluster.

## Features

- **vLLM-powered inference**: Qwen2.5-{7B,14B,32B,72B-AWQ}, PagedAttention, continuous batching
- **Multi-retriever**: BM25, BGE-M3 dense, Hybrid RRF
- **Multi-dataset**: MuSiQue, 2WikiMultihopQA, HotpotQA
- **Failure taxonomy**: 5-class diagnosis (Retrieval Miss / Distraction / Generation Ignore / Hallucinate / Misattribute)
- **Benchmark suite**: latency / throughput / TTFT / ITL / VRAM
- **OpenAI-compatible API**: FastAPI server for downstream consumption

## Quick Start

```bash
mamba env create -f environment.yml
conda activate rag-bench
rag-bench run --config configs/qwen25_7b_bge.yaml
rag-bench bench latency --model qwen25-72b-awq --batch-sizes 1,4,16,64
rag-bench serve --config configs/qwen25_7b.yaml --port 8000
```

## Architecture

See `docs/architecture.md`.

## Author

Ziheng Fan (Ethan), Imperial College London MSc ACSE 2025-2026
