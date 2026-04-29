# RAG eval baseline matrix

Aggregated from `results/evaluations/`. Each cell shows the most recent run for `(model, retriever, dataset)`.

## Qwen/Qwen2.5-7B-Instruct

`n_samples=100`, `top_k=5`, `temperature=0`.

### Exact Match

| retriever | hotpotqa | musique | two_wiki |
|---|---|---|---|
| **bge_m3** | 0.510 | 0.220 | 0.310 |
| **bm25** | 0.500 | 0.150 | 0.240 |
| **hybrid_rrf** | 0.470 | 0.200 | 0.270 |

### Token-level F1

| retriever | hotpotqa | musique | two_wiki |
|---|---|---|---|
| **bge_m3** | 0.604 | 0.348 | 0.358 |
| **bm25** | 0.596 | 0.236 | 0.280 |
| **hybrid_rrf** | 0.583 | 0.339 | 0.321 |

### Recall@5

| retriever | hotpotqa | musique | two_wiki |
|---|---|---|---|
| **bge_m3** | 0.631 | 0.633 | 0.534 |
| **bm25** | 0.590 | 0.479 | 0.398 |
| **hybrid_rrf** | 0.686 | 0.612 | 0.471 |

### Recall@1

| retriever | hotpotqa | musique | two_wiki |
|---|---|---|---|
| **bge_m3** | 0.265 | 0.321 | 0.249 |
| **bm25** | 0.293 | 0.224 | 0.164 |
| **hybrid_rrf** | 0.267 | 0.250 | 0.201 |

### Timing

| run | n | gen (s) | throughput (tok/s) | total (s) |
|---|---:|---:|---:|---:|
| bge_m3 / hotpotqa | 100 | 2.2 | 214 | 183.4 |
| bge_m3 / musique | 100 | 2.7 | 193 | 82.1 |
| bge_m3 / two_wiki | 100 | 2.7 | 198 | 82.0 |
| bm25 / hotpotqa | 100 | 2.3 | 214 | 49.4 |
| bm25 / musique | 100 | 2.4 | 181 | 69.9 |
| bm25 / two_wiki | 100 | 2.5 | 206 | 47.7 |
| hybrid_rrf / hotpotqa | 100 | 2.3 | 208 | 85.4 |
| hybrid_rrf / musique | 100 | 2.2 | 204 | 78.7 |
| hybrid_rrf / two_wiki | 100 | 2.2 | 223 | 63.8 |

