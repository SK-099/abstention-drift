"""Grounded answer generation with an explicit 'I don't know' option (the prompt-only baseline)."""
from __future__ import annotations

from .data import Task, format_history
from .idk import IDK_RESPONSE

GEN_SYSTEM = (
    "You are a helpful assistant that answers ONLY from the provided passages. "
    "If the passages do not contain the information needed to answer the latest question, reply exactly: "
    f"\"{IDK_RESPONSE}\" Do not use outside knowledge. Be concise."
)


def format_passages(passages: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {p.get('title', '')}\n{p['text'].strip()}" for i, p in enumerate(passages))


def generate_answer(llm, task: Task, passages: list[dict]) -> str:
    hist = format_history(task.history)
    user = (f"Passages:\n{format_passages(passages)}\n\n"
            + (f"Conversation so far:\n{hist}\n\n" if hist else "")
            + f"Latest question: {task.question}\nAnswer:")
    return llm.chat([{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}])
