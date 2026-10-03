"""POST /chat/stream: one user turn through the S5 agent, streamed as Server-Sent Events.

Events: `meta` (conversation and message ids), `step` (agent node finished: rewrite, retrieve, grade,
generate, self_check, retry, clarify_or_abstain), `token` (answer tokens as the LLM writes them; a
later self-check may still replace the answer, so clients render the `final` event's answer),
`final` (answer, decision answer|abstain|clarify, p_unanswerable, citations, trace, timings) and
`error`. The conversation id doubles as the LangGraph thread id (Postgres checkpointer).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from abstain.data import Task

from ..agent_runtime import get_runtime
from ..auth import current_user
from ..calibration import get_calibrations
from ..db import async_sessionmaker_, get_session
from ..guardrails import limiter
from ..models import Conversation, Document, Message, User, now
from ..prompts import versions
from ..safety import BLOCKED_INPUT, check_input
from ..schemas import ChatRequest
from ..settings import get_settings
from ..telemetry import get_metrics
from ..tracing import TurnTrace, flush, langfuse_handler, trace_metadata
from ._access import own_conversation, readable_collection

log = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _history(msgs: list[Message]) -> list[dict]:
    return [{"speaker": "user" if m.role == "user" else "agent", "text": m.content} for m in msgs]


@router.post("/chat/stream", summary="Ask a question; streams agent steps, tokens and the final answer (SSE)")
@limiter.limit(lambda: get_settings().rate_limit_chat)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    s = get_settings()
    from ..model_registry import get_registry

    try:  # the model chosen in the picker (validated: provider configured, allow/block lists)
        model_id = get_registry(s).resolve(body.model)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    gin = await asyncio.to_thread(check_input, body.message)  # input guardrails (injection, toxicity, PII)
    question = gin.text  # personal data / secrets redacted before the LLM, the DB or the traces see them
    if body.conversation_id:
        conv = await own_conversation(session, body.conversation_id, user)
        collection = conv.collection_id
    else:
        collection = (await readable_collection(session, body.collection, user)).id
        conv = Conversation(user_id=user.id, collection_id=collection, title=question[:80])
        session.add(conv)
        await session.flush()
    prev = (
        (await session.execute(select(Message).where(Message.conversation_id == conv.id).order_by(Message.created_at)))
        .scalars()
        .all()
    )
    turn = 1 + sum(m.role == "user" for m in prev)
    documents = None
    if body.document_ids is not None:  # ticked sources: must belong to this notebook
        ok = set(
            (
                await session.execute(
                    select(Document.id).where(Document.collection_id == collection, Document.id.in_(body.document_ids))
                )
            )
            .scalars()
            .all()
        )
        if not ok:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "no selected source belongs to this notebook")
        documents = sorted(ok)
    umsg = Message(conversation_id=conv.id, role="user", content=question, turn=turn, meta={"guardrails": gin.events})
    session.add(umsg)
    conv.updated_at = now()
    await session.commit()

    task = Task(
        task_id=f"{conv.id}:{turn}",
        conversation_id=conv.id,
        turn=turn,
        collection=collection,
        history=_history(list(prev)),
        question=question,
        answerability="UNKNOWN",
    )
    ver = versions(s)
    handler = langfuse_handler(s)
    th = get_calibrations(s).get(model_id)
    ver = {
        **ver,
        "llm_model": model_id,
        "tau": round(th.tau, 4),
        "self_check_min_p": round(th.min_p, 4),
        "thresholds": th.source,  # research | calibrated | default (model not calibrated)
    }
    meta = trace_metadata(user.id, conv.id, ver, collection)
    trace = (
        TurnTrace(
            question=question, collection=collection, turn=turn, user_id=user.id, session_id=conv.id, versions=ver
        )
        if handler
        else None
    )
    conv_id, user_msg_id = conv.id, umsg.id
    get_metrics().guardrail_events(gin.events)
    if gin.blocked:
        return EventSourceResponse(_blocked(gin, conv_id, user_msg_id, turn, collection, ver, trace, handler), ping=15)
    rt = get_runtime(s)

    async def events():
        yield {
            "event": "meta",
            "data": json.dumps(
                {
                    "conversation_id": conv_id,
                    "message_id": user_msg_id,
                    "turn": turn,
                    "collection": collection,
                    "model": model_id,
                }
            ),
        }
        for e in gin.events:  # input redactions (personal data / secrets removed from the question)
            yield {"event": "guardrail", "data": json.dumps(e)}
        final, t0 = None, time.perf_counter()
        async for ev in rt.astream(
            task,
            conv_id,
            callbacks=[handler] if handler else None,
            metadata=meta,
            trace=trace,
            model=model_id,
            documents=documents,
        ):
            if ev["event"] == "final":
                final = ev["data"]
                continue  # sent after it is stored (with the message id)
            if ev["event"] == "token" and s.guardrails_buffer_output:
                continue  # only the checked final answer reaches the client
            yield {"event": ev["event"], "data": json.dumps(ev["data"])}
        if final is None:
            return
        async with async_sessionmaker_()() as db:  # request session is closed once streaming starts
            amsg = Message(
                conversation_id=conv_id,
                role="assistant",
                content=final["answer"],
                turn=turn,
                decision=final["decision"],
                p_unanswerable=final["p_unanswerable"],
                latency_ms=final["latency_ms"],
                meta={
                    k: final[k]
                    for k in ("abstained_by", "query", "queries", "attempts", "citations", "trace", "timings")
                }
                | {"guardrails": final.get("guardrails", [])}
                | {"versions": ver, "model": model_id, "documents": documents},
            )
            db.add(amsg)
            await db.commit()
            final["message_id"] = amsg.id
        get_metrics().turn(
            final["decision"], collection, time.perf_counter() - t0, final["timings"], final["attempts"] > 0
        )
        get_metrics().guardrail_events(e for e in final.get("guardrails", []) if e["stage"] == "output")
        final["conversation_id"] = conv_id
        final["model"] = model_id
        yield {"event": "final", "data": json.dumps(final)}
        if handler:
            flush()

    return EventSourceResponse(events(), ping=15)


async def _blocked(gin, conv_id, user_msg_id, turn, collection, ver, trace, handler):
    """The input guardrails blocked the question: answer with a fixed message, never call the agent."""
    final = {
        "answer": BLOCKED_INPUT,
        "decision": "blocked",
        "abstained_by": "guardrail",
        "p_unanswerable": None,
        "query": None,
        "queries": [],
        "attempts": 0,
        "citations": [],
        "trace": [f"guardrail[input]: {e['rule']} -> {e['action']}" for e in gin.events],
        "timings": {},
        "latency_ms": 0,
        "guardrails": gin.events,
    }
    yield {
        "event": "meta",
        "data": json.dumps(
            {"conversation_id": conv_id, "message_id": user_msg_id, "turn": turn, "collection": collection}
        ),
    }
    for e in gin.events:
        yield {"event": "guardrail", "data": json.dumps(e)}
    async with async_sessionmaker_()() as db:
        amsg = Message(
            conversation_id=conv_id,
            role="assistant",
            content=BLOCKED_INPUT,
            turn=turn,
            decision="blocked",
            latency_ms=0,
            meta={"abstained_by": "guardrail", "guardrails": gin.events, "versions": ver},
        )
        db.add(amsg)
        await db.commit()
        final["message_id"] = amsg.id
    get_metrics().turn("blocked", collection, 0.0, {}, False)
    if trace is not None:
        with trace:
            trace.finish(final)
        flush()
    final["conversation_id"] = conv_id
    yield {"event": "final", "data": json.dumps(final)}
