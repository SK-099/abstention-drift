"""Builds the S5 LangGraph agent with production components and runs it per conversation thread.

LLM: OpenAI-compatible client; retrieval: Qdrant hybrid + re-ranker; gate: fine-tuned checker;
memory: LangGraph Postgres checkpointer (thread_id = conversation id), so each conversation's agent
state is persisted and inspectable/resumable.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from abstain.agent import AbstentionAgent
from abstain.data import Task

from .settings import Settings

log = logging.getLogger(__name__)
STAGE_LABELS = {
    "rewrite": "Rewriting the question",
    "retrieve": "Searching the documents",
    "grade": "Checking the evidence",
    "generate": "Writing the answer",
    "self_check": "Verifying the answer",
    "retry": "Evidence weak: searching again",
    "clarify_or_abstain": "Deciding whether to ask for clarification",
}


@dataclass
class Components:
    llm: Any
    retriever: Any  # HybridQdrantRetriever-like: .search(query, collection, k)
    gate: Any  # callable with .threshold, or None
    checkpointer: Any


def make_checkpointer(s: Settings):
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serde = JsonPlusSerializer(allowed_msgpack_modules=[("abstain.data", "Task")])
    if s.checkpointer == "postgres" and s.database_url.startswith("postgresql"):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            s.checkpoint_conninfo,
            max_size=10,
            open=True,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        saver = PostgresSaver(pool, serde=serde)
        saver.setup()
        return saver
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver(serde=serde)


def default_components(s: Settings) -> Components:
    from .checker import make_gate
    from .llm import RoutingChat
    from .retrieval import HybridQdrantRetriever, Reranker, get_embedder, get_store

    return Components(
        RoutingChat(s),
        HybridQdrantRetriever(s, get_store(s), get_embedder(s), Reranker(s)),
        make_gate(s),
        make_checkpointer(s),
    )


class AgentRuntime:
    def __init__(self, s: Settings, components: Components | None = None):
        from .retrieval import make_lc_retriever

        self.s = s
        self.c = components or default_components(s)
        self.agent = AbstentionAgent(
            self.c.llm,
            make_lc_retriever(self.c.retriever),
            self.c.gate,
            threshold=s.resolved_gate_threshold() if self.c.gate is not None else 1.0,
            max_retries=s.max_retries,
            self_check=s.self_check,
            clarify=s.clarify,
            clarify_min_turn=s.clarify_min_turn,
            self_check_min_p=s.resolved_min_p(),
            checkpointer=self.c.checkpointer,
            rewrite_kwargs={"history_turns": s.history_turns, "history_chars": s.history_chars},
            gen_kwargs={
                "history_turns": s.history_turns,
                "history_chars": s.history_chars,
                "passage_chars": s.passage_chars,
            },
        )

    # ------------------------------------------------------------------ sync API
    def stream(
        self, task: Task, thread_id: str, callbacks=None, metadata=None, trace=None, model=None, documents=None
    ) -> Iterator[dict]:
        """`trace`: optional tracing.TurnTrace, entered here because it must live in the graph's thread.
        `model`: model id from the picker (None = default); `documents`: restrict search to these documents."""
        from abstain.agent import THRESHOLDS

        from .calibration import get_calibrations
        from .model_registry import REQUEST_DOCS, REQUEST_MODEL, default_model_id

        REQUEST_MODEL.set(model)
        REQUEST_DOCS.set(tuple(documents) if documents else None)
        th = get_calibrations(self.s).get(model or default_model_id(self.s))
        THRESHOLDS.set((th.tau, th.min_p))  # this model's own calibrated tau / min_p
        if trace is None:
            yield from self._stream(task, thread_id, callbacks, metadata)
            return
        with trace:
            for ev in self._stream(task, thread_id, callbacks, metadata):
                if ev["event"] == "final":
                    trace.finish(ev["data"])
                yield ev

    def _stream(self, task: Task, thread_id: str, callbacks=None, metadata=None) -> Iterator[dict]:
        t0 = time.perf_counter()
        for ev in self.agent.stream(task, thread_id=thread_id, callbacks=callbacks, metadata=metadata):
            if ev[0] == "token":
                yield {"event": "token", "data": {"text": ev[1]}}
            elif ev[0] == "step":
                node = ev[1]
                yield {"event": "step", "data": {"node": node, "label": STAGE_LABELS.get(node, node), "detail": ev[2]}}
            else:
                from .safety import guard_final

                final, events = guard_final(self.final_payload(ev[1], time.perf_counter() - t0))
                for e in events:  # output guardrails ran on the answer (redacted / blocked)
                    yield {"event": "guardrail", "data": e}
                final["guardrails"] = list(final.get("guardrails", [])) + events
                yield {"event": "final", "data": final}

    def run(self, task: Task, thread_id: str, callbacks=None, metadata=None) -> dict:
        final = None
        for ev in self.stream(task, thread_id, callbacks, metadata):
            if ev["event"] == "final":
                final = ev["data"]
        return final

    @staticmethod
    def final_payload(out, seconds: float) -> dict:
        cites = (
            [
                {
                    "index": i + 1,
                    "document_id": p["document_id"],
                    "title": p.get("title", ""),
                    "snippet": p["text"][:400],
                    "score": round(float(p.get("score", 0.0)), 4),
                    "source_document": p.get("source_document"),
                    "page": p.get("page"),
                    "section": p.get("section"),
                }
                for i, p in enumerate(out.passages)
            ]
            if out.decision == "answer"
            else []
        )
        return {
            "answer": out.answer,
            "decision": out.decision,
            "abstained_by": out.abstained_by,
            "p_unanswerable": None if out.p_unanswerable is None else round(float(out.p_unanswerable), 4),
            "query": out.query,
            "queries": out.queries,
            "attempts": out.attempts,
            "citations": cites,
            "trace": out.trace,
            "timings": {k: round(v, 3) for k, v in out.timings.items()},
            "latency_ms": int(seconds * 1000),
        }

    def state(self, thread_id: str) -> dict | None:
        snap = self.agent.graph.get_state(self.agent.make_config(thread_id))
        if not snap or not snap.values:
            return None
        v = dict(snap.values)
        v.pop("cache", None)
        t = v.pop("task", None)
        if t is not None:
            v["question"] = t.question
            v["turn"] = t.turn
        v["passages"] = [{"document_id": p["document_id"], "score": p.get("score")} for p in v.get("passages", [])]
        return v

    # ------------------------------------------------------------------ async bridge for SSE
    async def astream(
        self, task: Task, thread_id: str, callbacks=None, metadata=None, trace=None, model=None, documents=None
    ) -> AsyncIterator[dict]:
        """Run the (sync) graph in a worker thread and yield its events on the event loop."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        done = object()

        def work():
            try:
                for ev in self.stream(task, thread_id, callbacks, metadata, trace, model, documents):
                    loop.call_soon_threadsafe(q.put_nowait, ev)
            except Exception as e:
                log.exception("agent run failed")
                msg = str(e)[:500]
                if "timed out" in msg.lower() or "timeout" in type(e).__name__.lower():
                    msg = (
                        "The model took too long to answer. On a small GPU this happens when the model has to be "
                        "loaded first (after switching models) or while a source guide or Studio output is being "
                        "written. Please wait a moment and ask again."
                    )
                loop.call_soon_threadsafe(q.put_nowait, {"event": "error", "data": {"message": msg}})
            finally:
                loop.call_soon_threadsafe(q.put_nowait, done)

        ctx = contextvars.copy_context()
        threading.Thread(target=ctx.run, args=(work,), daemon=True, name=f"agent-{thread_id[:8]}").start()
        while True:
            ev = await q.get()
            if ev is done:
                break
            yield ev


_runtime: AgentRuntime | None = None
_lock = threading.Lock()


def get_runtime(s: Settings) -> AgentRuntime:
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = AgentRuntime(s)
    return _runtime


def set_runtime(rt: AgentRuntime | None) -> None:
    """Dependency injection for tests."""
    global _runtime
    _runtime = rt
