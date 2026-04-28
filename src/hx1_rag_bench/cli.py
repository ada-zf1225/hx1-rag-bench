"""hx1-rag-bench command-line interface.

Subcommands:
    rag-bench info                Show env / GPU / installed deps
    rag-bench hello               Smoke-test vLLM with a tiny model
    rag-bench run --config CFG    End-to-end RAG eval on a dataset
    rag-bench bench latency ...   Inference latency benchmark
    rag-bench bench throughput ...
    rag-bench serve --config CFG  Start FastAPI OpenAI-compatible server
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(
    name="rag-bench",
    help="HX1 RAG inference benchmark and diagnosis framework.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
bench_app = typer.Typer(name="bench", help="Inference benchmark suite.", no_args_is_help=True)
app.add_typer(bench_app)

console = Console()


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@app.command()
def info() -> None:
    """Show environment / GPU / installed dependency versions."""
    _setup_logging()
    table = Table(title="hx1-rag-bench env", show_header=True, header_style="bold cyan")
    table.add_column("Component")
    table.add_column("Version / Status")

    import platform

    table.add_row("Python", platform.python_version())
    table.add_row("Platform", platform.platform())

    try:
        import torch

        table.add_row("PyTorch", torch.__version__)
        table.add_row("CUDA available", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                table.add_row(
                    f"GPU {i}",
                    f"{p.name} ({p.total_memory / 1024**3:.1f} GB, "
                    f"sm_{p.major}{p.minor})",
                )
    except ImportError:
        table.add_row("PyTorch", "[red]not installed[/red]")

    for pkg in ("vllm", "transformers", "FlagEmbedding", "fastapi"):
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            table.add_row(pkg, ver)
        except ImportError:
            table.add_row(pkg, "[red]not installed[/red]")

    console.print(table)


@app.command()
def hello(
    model: str = typer.Option(
        "Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model id (use a small one for smoke tests)",
    ),
    prompt: str = typer.Option(
        "What is the capital of France?",
        help="Test prompt",
    ),
    max_tokens: int = typer.Option(64, help="Max output tokens"),
) -> None:
    """Quick smoke test: load vLLM, generate one completion, exit."""
    _setup_logging()
    from hx1_rag_bench.config import ModelConfig
    from hx1_rag_bench.inference.engine import VLLMEngine

    console.rule(f"[bold cyan]vLLM Hello World ({model})")
    cfg = ModelConfig(name=model, max_tokens=max_tokens, gpu_memory_utilization=0.85)
    engine = VLLMEngine(cfg)
    engine.load()

    messages = [{"role": "user", "content": prompt}]
    formatted = engine.format_chat(messages)
    console.print(f"[dim]Prompt:[/dim] {prompt}")

    result = engine.generate(formatted)
    out = result.outputs[0]
    console.print(f"[bold green]Output:[/bold green] {out.text}")
    console.print(
        f"[dim]"
        f"prompt_tokens={out.prompt_token_count}, "
        f"output_tokens={out.output_token_count}, "
        f"latency={out.latency_s:.2f}s, "
        f"throughput={out.tokens_per_second:.1f} tok/s"
        f"[/dim]"
    )

    engine.shutdown()


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    output_dir: Path = typer.Option(Path("results/evaluations"), help="Output dir"),
    max_samples: int | None = typer.Option(None, help="Override dataset.max_samples"),
) -> None:
    """End-to-end RAG eval on a dataset."""
    _setup_logging()
    from hx1_rag_bench.config import AppConfig
    from hx1_rag_bench.pipeline.runner import run_eval

    cfg = AppConfig.from_yaml(config)
    if max_samples is not None:
        cfg.dataset.max_samples = max_samples

    console.rule(
        f"[bold cyan]RAG eval: "
        f"{cfg.dataset.name} | {cfg.retriever.backend} | {cfg.model.name}"
        f"[/bold cyan]"
    )
    report = run_eval(cfg, output_dir=output_dir)

    table = Table(title="Aggregate metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("N samples", str(report.n_samples))
    table.add_row("EM", f"{report.em:.4f}")
    table.add_row("F1", f"{report.f1:.4f}")
    for k in sorted(report.recall_at_k):
        table.add_row(f"Recall@{k}", f"{report.recall_at_k[k]:.4f}")
    table.add_row("Generate (s)", f"{report.timing['generate_s']:.1f}")
    table.add_row("Throughput (tok/s)", f"{report.timing['throughput_tok_per_s']:.0f}")
    table.add_row("Total (s)", f"{report.timing['total_s']:.1f}")
    console.print(table)
    console.print(f"\n[dim]Run dir:[/dim] [cyan]{report.run_dir}[/cyan]")


@bench_app.command("latency")
def bench_latency(
    model: str = typer.Option(..., help="HF model id"),
    batch_sizes: str = typer.Option("1,4,16", help="comma-separated"),
    num_requests: int = typer.Option(32, help="requests per batch_size"),
    output_len: int = typer.Option(128, help="output tokens"),
    input_len: int = typer.Option(512, help="input tokens"),
) -> None:
    """Latency benchmark across batch sizes (stub)."""
    _setup_logging()
    console.print(f"[yellow]TODO[/yellow]: latency bench for {model}")
    console.print(f"batch_sizes={batch_sizes}, n={num_requests}")


@bench_app.command("throughput")
def bench_throughput(
    model: str = typer.Option(..., help="HF model id"),
    concurrency: str = typer.Option("1,4,16,64", help="comma-separated"),
    num_requests: int = typer.Option(256),
) -> None:
    """Throughput benchmark across concurrency levels (stub)."""
    _setup_logging()
    console.print(f"[yellow]TODO[/yellow]: throughput bench for {model}")


@app.command()
def serve(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
) -> None:
    """Start FastAPI OpenAI-compatible API server (stub)."""
    _setup_logging()
    console.print(f"[yellow]TODO[/yellow]: serve {config} on {host}:{port}")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main() or 0)
