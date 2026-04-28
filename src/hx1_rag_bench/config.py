"""Global configuration via pydantic-settings.

Loaded from YAML config files + environment variable overrides.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """LLM inference engine configuration."""

    name: str = Field(..., description="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct")
    quantization: Literal["none", "awq", "gptq", "bitsandbytes"] | None = None
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"

    # vLLM engine knobs
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = Field(0.90, ge=0.1, le=0.98)
    max_model_len: int | None = None
    max_num_seqs: int = 256
    max_num_batched_tokens: int | None = None
    swap_space_gb: int = 4
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    trust_remote_code: bool = True
    download_dir: str | None = None  # default: HF_HOME

    # Sampling defaults (overridable per-request)
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    repetition_penalty: float = 1.0
    stop: list[str] = Field(default_factory=list)


class RetrieverConfig(BaseModel):
    """Retrieval configuration."""

    backend: Literal["bm25", "bge_m3", "hybrid_rrf"] = "bm25"
    top_k: int = 5

    # BGE-M3 specific
    bge_model: str = "BAAI/bge-m3"
    bge_batch_size: int = 64
    bge_max_length: int = 512
    bge_use_fp16: bool = True

    # Index storage
    index_dir: Path = Path("data/indexes")

    # Hybrid RRF specific
    rrf_k: int = 60
    rrf_weights: list[float] = Field(default_factory=lambda: [0.5, 0.5])


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: Literal["musique", "two_wiki", "hotpotqa"] = "musique"
    split: Literal["train", "validation", "test"] = "validation"
    max_samples: int | None = None
    seed: int = 42
    cache_dir: Path = Path("data/raw")


class EvalConfig(BaseModel):
    """Evaluation metric configuration."""

    metrics: list[Literal["em", "f1", "rouge", "recall_at_k", "nli_attribution"]] = Field(
        default_factory=lambda: ["em", "f1", "recall_at_k"]
    )
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    nli_batch_size: int = 32


class BenchConfig(BaseModel):
    """Benchmark suite configuration."""

    batch_sizes: list[int] = Field(default_factory=lambda: [1, 4, 16, 64])
    concurrency: list[int] = Field(default_factory=lambda: [1, 4, 16])
    num_requests: int = 100
    warmup_requests: int = 10
    output_lengths: list[int] = Field(default_factory=lambda: [128, 512])
    input_lengths: list[int] = Field(default_factory=lambda: [512, 2048])


class ServerConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/v1"
    enable_streaming: bool = True
    request_timeout_s: int = 300


class AppConfig(BaseSettings):
    """Top-level config loaded from YAML + env vars (HX1_RAG_*)."""

    model_config = SettingsConfigDict(
        env_prefix="HX1_RAG_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Sub-configs
    model: ModelConfig
    retriever: RetrieverConfig = RetrieverConfig()
    dataset: DatasetConfig = DatasetConfig()
    eval: EvalConfig = EvalConfig()
    bench: BenchConfig = BenchConfig()
    server: ServerConfig = ServerConfig()

    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    results_dir: Path = Path("results")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # HF / vLLM env
    hf_home: Path | None = None
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load config from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


__all__ = [
    "AppConfig",
    "ModelConfig",
    "RetrieverConfig",
    "DatasetConfig",
    "EvalConfig",
    "BenchConfig",
    "ServerConfig",
]
