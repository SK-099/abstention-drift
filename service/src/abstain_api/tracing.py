"""Langfuse tracing of every LangGraph step and LLM call (LangChain callback handler).
Enabled when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set."""

from __future__ import annotations

import logging
import os

from .settings import Settings

log = logging.getLogger(__name__)
_handler = None


def langfuse_handler(s: Settings):
    global _handler
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    if _handler is None:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key)
        if s.langfuse_host:
            os.environ.setdefault("LANGFUSE_HOST", s.langfuse_host)
            os.environ.setdefault("LANGFUSE_BASE_URL", s.langfuse_host)
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError:  # Langfuse SDK v2
            from langfuse.callback import CallbackHandler  # type: ignore
        _handler = CallbackHandler()
        log.info("Langfuse tracing enabled (%s)", s.langfuse_host or "cloud")
    return _handler


def trace_metadata(user_id: str, conversation_id: str, versions: dict, collection: str) -> dict:
    """Langfuse reads these keys from the LangChain run metadata (session, user, tags)."""
    return {
        "langfuse_session_id": conversation_id,
        "langfuse_user_id": user_id,
        "langfuse_tags": [f"collection:{collection}", f"prompt:{versions['prompt_version']}"]
        + ([f"model:{versions['llm_model']}"] if versions.get("llm_model") else []),
        "collection": collection,
        **versions,
    }


def flush() -> None:
    if _handler is None:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass


def turn_label(collection: str, turn: int, decision: str | None = None) -> str:
    """Readable trace name, e.g. 'chat · cloud · turn 3 · answer'."""
    return " · ".join(["chat", collection, f"turn {turn}"] + ([decision] if decision else []))


class TurnTrace:
    """One Langfuse trace per chat turn.

    Opens a root span named after the turn; the LangGraph callback handler nests every node and LLM
    call under it. When the turn ends, the trace is renamed with the decision
    ('chat · cloud · turn 3 · answer'), its input/output become the user's question and the final
    answer, and the decision is added as a tag. Must be entered in the thread that runs the graph.
    """

    def __init__(self, *, question: str, collection: str, turn: int, user_id: str, session_id: str, versions: dict):
        self.question, self.collection, self.turn = question, collection, turn
        self.user_id, self.session_id, self.versions = user_id, session_id, versions
        self.tags = [f"collection:{collection}", f"prompt:{versions['prompt_version']}"]
        if versions.get("llm_model"):
            self.tags.append(f"model:{versions['llm_model']}")
        self._stack = None
        self._root = None

    def __enter__(self):
        from contextlib import ExitStack

        from langfuse import get_client, propagate_attributes

        self._stack = ExitStack()
        try:
            self._stack.enter_context(
                propagate_attributes(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    tags=self.tags,
                    metadata={k: str(v)[:200] for k, v in self.versions.items()},
                )
            )
            self._root = self._stack.enter_context(
                get_client().start_as_current_observation(
                    as_type="chain",
                    name=turn_label(self.collection, self.turn),
                    input={"question": self.question},
                )
            )
        except Exception:  # tracing must never break a chat turn
            log.exception("could not start the Langfuse trace")
            self._stack.close()
            self._stack = self._root = None
        return self

    def finish(self, final: dict) -> None:
        """Name the trace after the decision and set the question / answer as trace input / output."""
        if self._root is None:
            return
        try:
            from langfuse import LangfuseOtelSpanAttributes as A

            label = turn_label(self.collection, self.turn, final.get("decision"))
            output = {
                "answer": final.get("answer"),
                "decision": final.get("decision"),
                "abstained_by": final.get("abstained_by"),
                "p_unanswerable": final.get("p_unanswerable"),
                "attempts": final.get("attempts"),
                "citations": [c.get("document_id") for c in final.get("citations", [])],
            }
            self._root.update(name=label, output=output)
            self._root.set_trace_io(input=self.question, output=final.get("answer"))
            span = self._root._otel_span
            span.update_name(label)
            span.set_attribute(A.TRACE_NAME, label)
            guard_tags = sorted({f"guardrail:{e['rule']}" for e in final.get("guardrails", [])})
            span.set_attribute(A.TRACE_TAGS, [*self.tags, f"decision:{final.get('decision')}", *guard_tags])
        except Exception:
            log.exception("could not finalise the Langfuse trace")

    def __exit__(self, *exc):
        if self._stack is not None:
            self._stack.close()
        return False
