"""Turn a context-dependent follow-up into a standalone search query."""
from __future__ import annotations

from .data import Task, format_history

REWRITE_SYSTEM = (
    "You rewrite the user's latest question in a conversation into a single standalone search query. "
    "Resolve pronouns and references using the conversation. Keep the user's intent; do not answer it. "
    "Output only the rewritten question."
)


class QueryRewriter:
    """mode: last_turn | full_history | oracle | llm"""

    def __init__(self, mode: str = "llm", llm=None, oracle: dict[str, str] | None = None,
                 history_turns: int = 6, history_chars: int | None = None):
        if mode == "llm" and llm is None:
            raise ValueError("mode='llm' needs an llm")
        self.mode, self.llm, self.oracle = mode, llm, oracle or {}
        self.history_turns, self.history_chars = history_turns, history_chars

    def __call__(self, task: Task) -> str:
        if self.mode == "last_turn" or not task.history:
            return task.question
        if self.mode == "full_history":
            users = [t["text"] for t in task.history if t["speaker"] == "user"]
            return " ".join(users[-3:] + [task.question])
        if self.mode == "oracle":
            return self.oracle.get(task.task_id, task.question)
        msgs = [{"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": f"Conversation:\n{format_history(task.history, self.history_turns, self.history_chars)}\n\n"
                                            f"Latest question: {task.question}\n\nStandalone question:"}]
        out = self.llm.chat(msgs, max_new_tokens=64).split("\n")[0].strip().strip('"')
        return out or task.question
