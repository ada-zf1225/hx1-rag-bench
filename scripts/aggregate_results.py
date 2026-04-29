#!/usr/bin/env python3
"""Aggregate eval run metrics into a markdown report.

Walks `results/evaluations/`, picks the most recent run per
(model, retriever, dataset) cell, and renders an EM / F1 / Recall@k matrix
plus timing summary.

Run from project root:
    python scripts/aggregate_results.py                 # stdout
    python scripts/aggregate_results.py -o docs/baseline_qwen25_7b.md
    python scripts/aggregate_results.py --model-filter Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_run(run_dir: Path) -> dict[str, Any] | None:
    m_path = run_dir / "metrics.json"
    c_path = run_dir / "config.json"
    if not (m_path.exists() and c_path.exists()):
        return None
    metrics = json.loads(m_path.read_text())
    cfg = json.loads(c_path.read_text())
    return {
        "run_id": metrics["run_id"],
        "run_dir": run_dir.name,
        "dataset": cfg["dataset"]["name"],
        "retriever": cfg["retriever"]["backend"],
        "top_k": cfg["retriever"]["top_k"],
        "model": cfg["model"]["name"],
        "quantization": cfg["model"].get("quantization"),
        "n_samples": metrics["n_samples"],
        "em": metrics["em"],
        "f1": metrics["f1"],
        "recall_at_1": metrics.get("recall_at_1"),
        "recall_at_5": metrics.get("recall_at_5"),
        "recall_at_10": metrics.get("recall_at_10"),
        "timing": metrics["timing"],
    }


def _collect_latest(root: Path, model_filter: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        row = _read_run(run_dir)
        if row is None:
            continue
        if model_filter and row["model"] != model_filter:
            continue
        rows.append(row)

    # Pick latest per (model, retriever, dataset) by run_id (timestamp prefix)
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["model"], row["retriever"], row["dataset"])
        if key not in latest or row["run_id"] > latest[key]["run_id"]:
            latest[key] = row
    return list(latest.values())


def _render_metric_table(rows: list[dict[str, Any]], metric: str, datasets: list[str], retrievers: list[str]) -> list[str]:
    cells: dict[tuple[str, str], dict[str, Any]] = {
        (r["retriever"], r["dataset"]): r for r in rows
    }
    out = [
        "| retriever | " + " | ".join(datasets) + " |",
        "|---|" + "---|" * len(datasets),
    ]
    for retriever in retrievers:
        line = f"| **{retriever}** |"
        for dataset in datasets:
            cell = cells.get((retriever, dataset))
            if cell is None or cell.get(metric) is None:
                line += " — |"
            else:
                line += f" {cell[metric]:.3f} |"
        out.append(line)
    return out


def _render_timing_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| run | n | gen (s) | throughput (tok/s) | total (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["model"], x["retriever"], x["dataset"])):
        label = f"{r['retriever']} / {r['dataset']}"
        out.append(
            f"| {label} | {r['n_samples']} | "
            f"{r['timing']['generate_s']:.1f} | "
            f"{r['timing']['throughput_tok_per_s']:.0f} | "
            f"{r['timing']['total_s']:.1f} |"
        )
    return out


def render_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No runs found._\n"

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    out: list[str] = ["# RAG eval baseline matrix", ""]
    out.append(
        "Aggregated from `results/evaluations/`. Each cell shows the most "
        "recent run for `(model, retriever, dataset)`."
    )
    out.append("")

    for model in sorted(by_model):
        model_rows = by_model[model]
        retrievers = sorted({r["retriever"] for r in model_rows})
        datasets = sorted({r["dataset"] for r in model_rows})
        first = model_rows[0]
        n = first["n_samples"]
        top_k = first["top_k"]
        quant = f" ({first['quantization']})" if first.get("quantization") else ""

        out.append(f"## {model}{quant}")
        out.append("")
        out.append(f"`n_samples={n}`, `top_k={top_k}`, `temperature=0`.")
        out.append("")

        for metric, label in (
            ("em", "Exact Match"),
            ("f1", "Token-level F1"),
            ("recall_at_5", "Recall@5"),
            ("recall_at_1", "Recall@1"),
        ):
            out.append(f"### {label}")
            out.append("")
            out.extend(_render_metric_table(model_rows, metric, datasets, retrievers))
            out.append("")

        out.append("### Timing")
        out.append("")
        out.extend(_render_timing_table(model_rows))
        out.append("")

    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("results/evaluations"))
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Markdown output path; default stdout")
    p.add_argument("--model-filter", default=None,
                   help="If set, only include runs whose model.name matches exactly")
    args = p.parse_args()

    rows = _collect_latest(args.root, args.model_filter)
    md = render_markdown(rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"Wrote {args.output}")
    else:
        print(md, end="")


if __name__ == "__main__":
    main()
