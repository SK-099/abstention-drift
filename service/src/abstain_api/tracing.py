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
        "langfuse_tags": [f"collection:{collection}", f"prompt:{versions['prompt_version']}"],
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
