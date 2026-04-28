"""vLLM inference engine wrapper.

Centralizes vLLM lifecycle (LLM init, AsyncLLMEngine for serving),
prompt formatting, and sampling param construction.
"""
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

from hx1_rag_bench.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Single generation result."""

    text: str
    prompt: str
    prompt_token_count: int
    output_token_count: int
    finish_reason: str
    latency_s: float
    request_id: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_token_count + self.output_token_count

    @property
    def tokens_per_second(self) -> float:
        if self.latency_s <= 0:
            return 0.0
        return self.output_token_count / self.latency_s


@dataclass
class BatchGenerationOutput:
    """Batched generation results with aggregate stats."""

    outputs: list[GenerationOutput] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def total_output_tokens(self) -> int:
        return sum(o.output_token_count for o in self.outputs)

    @property
    def aggregate_throughput_tps(self) -> float:
        if self.wall_time_s <= 0:
            return 0.0
        return self.total_output_tokens / self.wall_time_s

    @property
    def mean_latency_s(self) -> float:
        if not self.outputs:
            return 0.0
        return sum(o.latency_s for o in self.outputs) / len(self.outputs)


class VLLMEngine:
    """Synchronous vLLM engine for offline batch inference (eval / bench).

    For online serving use VLLMServingEngine (AsyncLLMEngine-backed).
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._llm: Any = None  # vllm.LLM, lazily imported
        self._tokenizer: Any = None

    def load(self) -> None:
        """Heavy initialization: spawn vLLM workers, allocate KV cache."""
        from vllm import LLM

        kwargs: dict[str, Any] = dict(
            model=self.cfg.name,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            pipeline_parallel_size=self.cfg.pipeline_parallel_size,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            max_num_seqs=self.cfg.max_num_seqs,
            swap_space=self.cfg.swap_space_gb,
            enforce_eager=self.cfg.enforce_eager,
            enable_prefix_caching=self.cfg.enable_prefix_caching,
            trust_remote_code=self.cfg.trust_remote_code,
            dtype=self.cfg.dtype,
        )
        if self.cfg.quantization and self.cfg.quantization != "none":
            kwargs["quantization"] = self.cfg.quantization
        if self.cfg.max_model_len is not None:
            kwargs["max_model_len"] = self.cfg.max_model_len
        if self.cfg.max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = self.cfg.max_num_batched_tokens
        if self.cfg.download_dir:
            kwargs["download_dir"] = self.cfg.download_dir

        logger.info("Loading vLLM model: %s with kwargs: %s", self.cfg.name, kwargs)
        t0 = time.time()
        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        logger.info("vLLM ready in %.1fs", time.time() - t0)

    def _make_sampling_params(self, **overrides: Any) -> Any:
        from vllm import SamplingParams

        params = dict(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_tokens,
            repetition_penalty=self.cfg.repetition_penalty,
            stop=self.cfg.stop or None,
        )
        params.update(overrides)
        return SamplingParams(**params)

    def format_chat(self, messages: list[dict[str, str]]) -> str:
        """Apply chat template via the model's tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Engine not loaded; call .load() first")
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompts: str | Iterable[str],
        **sampling_overrides: Any,
    ) -> BatchGenerationOutput:
        """Run synchronous batch generation.

        vLLM automatically handles continuous batching internally;
        callers should pass as many prompts as memory allows for
        peak throughput.
        """
        if self._llm is None:
            raise RuntimeError("Engine not loaded; call .load() first")

        if isinstance(prompts, str):
            prompts = [prompts]
        prompts_list = list(prompts)

        sp = self._make_sampling_params(**sampling_overrides)

        t0 = time.time()
        raw_outputs = self._llm.generate(prompts_list, sp, use_tqdm=False)
        wall = time.time() - t0

        results: list[GenerationOutput] = []
        for raw in raw_outputs:
            comp = raw.outputs[0]
            results.append(
                GenerationOutput(
                    text=comp.text,
                    prompt=raw.prompt,
                    prompt_token_count=len(raw.prompt_token_ids),
                    output_token_count=len(comp.token_ids),
                    finish_reason=comp.finish_reason or "unknown",
                    latency_s=wall,  # per-request latency unavailable in sync API
                    request_id=raw.request_id,
                )
            )

        return BatchGenerationOutput(outputs=results, wall_time_s=wall)

    def shutdown(self) -> None:
        """Release vLLM workers and KV cache."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._tokenizer = None
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass


__all__ = ["VLLMEngine", "GenerationOutput", "BatchGenerationOutput"]
