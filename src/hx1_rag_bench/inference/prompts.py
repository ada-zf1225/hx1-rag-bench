"""RAG prompt templates.

Templates are deliberately minimal and instruction-style, optimized for
short-form answer extraction (EM/F1 metrics on multi-hop QA datasets).
"""
from __future__ import annotations

from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


SYSTEM_RAG = (
    "You are a precise QA assistant. Use the provided context documents "
    "to answer the question. If the answer cannot be found in the context, "
    "respond with 'unanswerable'. Keep the answer as short as possible — "
    "ideally a single entity or short phrase."
)

SYSTEM_NO_CONTEXT = (
    "You are a precise QA assistant. Answer the question as concisely as "
    "possible — ideally a single entity or short phrase. If you don't know "
    "the answer, say 'unknown'."
)


def format_context(docs: list[str]) -> str:
    """Pack a list of retrieved passages into an enumerated block."""
    if not docs:
        return "(no documents retrieved)"
    return "\n\n".join(f"[{i + 1}] {d.strip()}" for i, d in enumerate(docs))


def build_rag_messages(question: str, docs: list[str]) -> list[ChatMessage]:
    """Build chat messages for retrieval-augmented QA."""
    user = (
        f"Context documents:\n\n{format_context(docs)}\n\n"
        f"Question: {question.strip()}\n"
        f"Answer:"
    )
    return [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": user},
    ]


def build_no_context_messages(question: str) -> list[ChatMessage]:
    """Build chat messages for closed-book QA (no retrieval)."""
    return [
        {"role": "system", "content": SYSTEM_NO_CONTEXT},
        {"role": "user", "content": f"Question: {question.strip()}\nAnswer:"},
    ]


def build_judge_messages(
    question: str,
    gold: str,
    predicted: str,
    docs: list[str] | None = None,
) -> list[ChatMessage]:
    """Build chat messages for LLM-as-judge failure classification."""
    ctx = f"\n\nContext documents:\n\n{format_context(docs)}" if docs else ""
    user = (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Predicted answer: {predicted}{ctx}\n\n"
        "Classify the failure mode (or 'correct' if predicted matches gold). "
        "Output one of: correct, retrieval_miss, retrieval_distraction, "
        "generation_ignore, generation_hallucinate, generation_misattribute. "
        "Output only the label, no explanation."
    )
    return [
        {
            "role": "system",
            "content": "You are an expert RAG failure analyst.",
        },
        {"role": "user", "content": user},
    ]


__all__ = [
    "build_rag_messages",
    "build_no_context_messages",
    "build_judge_messages",
    "format_context",
    "SYSTEM_RAG",
    "SYSTEM_NO_CONTEXT",
]
