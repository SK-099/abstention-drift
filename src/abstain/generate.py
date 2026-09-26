"""Grounded answer generation with an explicit 'I don't know' option (the prompt-only baseline)."""
from __future__ import annotations

from .data import Task, format_history
from .idk import IDK_RESPONSE

GEN_SYSTEM = (
    "You are a helpful assistant that answers ONLY from the provided passages. "
    "If the passages do not contain the information needed to answer the latest question, reply exactly: "
    f"\"{IDK_RESPONSE}\" Do not use outside knowledge. Be concise."
)


def format_passages(passages: list[dict], max_chars: int | None = None) -> str:
    def cut(x: str) -> str:
        x = x.strip()
        return x if not max_chars or len(x) <= max_chars else x[:max_chars].rsplit(" ", 1)[0] + " ..."
    return "\n\n".join(f"[{i + 1}] {p.get('title', '')}\n{cut(p['text'])}" for i, p in enumerate(passages))


def generate_answer(llm, task: Task, passages: list[dict], passage_chars: int | None = None,
                    history_turns: int = 6, history_chars: int | None = None) -> str:
    """Prompt length drives latency on small GPUs: the three limits trim passages and history."""
    hist = format_history(task.history, history_turns, history_chars)
    user = (f"Passages:\n{format_passages(passages, passage_chars)}\n\n"
            + (f"Conversation so far:\n{hist}\n\n" if hist else "")
            + f"Latest question: {task.question}\nAnswer:")
    return llm.chat([{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}])
