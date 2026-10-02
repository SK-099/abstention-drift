"""POST /chat/stream: one user turn through the S5 agent, streamed as Server-Sent Events.

Events: `meta` (conversation and message ids), `step` (agent node finished: rewrite, retrieve, grade,
generate, self_check, retry, clarify_or_abstain), `token` (answer tokens as the LLM writes them; a
later self-check may still replace the answer, so clients render the `final` event's answer),
`final` (answer, decision answer|abstain|clarify, p_unanswerable, citations, trace, timings) and
`error`. The conversation id doubles as the LangGraph thread id (Postgres checkpointer).
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from abstain.data import Task

from ..agent_runtime import get_runtime
from ..auth import current_user
from ..db import async_sessionmaker_, get_session
from ..guardrails import limiter
from ..models import Conversation, Message, User, now
from ..prompts import versions
from ..schemas import ChatRequest
from ..settings import get_settings
from ..telemetry import get_metrics
from ..tracing import flush, langfuse_handler, trace_metadata
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
    if body.conversation_id:
        conv = await own_conversation(session, body.conversation_id, user)
        collection = conv.collection_id
    else:
        collection = (await readable_collection(session, body.collection, user)).id
        conv = Conversation(user_id=user.id, collection_id=collection, title=body.message[:80])
        session.add(conv)
        await session.flush()
    prev = (
        (await session.execute(select(Message).where(Message.conversation_id == conv.id).order_by(Message.created_at)))
        .scalars()
        .all()
    )
    turn = 1 + sum(m.role == "user" for m in prev)
    umsg = Message(conversation_id=conv.id, role="user", content=body.message, turn=turn)
    session.add(umsg)
    conv.updated_at = now()
    await session.commit()

    task = Task(
        task_id=f"{conv.id}:{turn}",
        conversation_id=conv.id,
        turn=turn,
        collection=collection,
        history=_history(list(prev)),
        question=body.message,
        answerability="UNKNOWN",
    )
    ver = versions(s)
    handler = langfuse_handler(s)
    meta = trace_metadata(user.id, conv.id, ver, collection)
    rt = get_runtime(s)
    conv_id, user_msg_id = conv.id, umsg.id

    async def events():
        yield {
            "event": "meta",
            "data": json.dumps(
                {"conversation_id": conv_id, "message_id": user_msg_id, "turn": turn, "collection": collection}
            ),
        }
        final, t0 = None, time.perf_counter()
        async for ev in rt.astream(task, conv_id, callbacks=[handler] if handler else None, metadata=meta):
            if ev["event"] == "final":
                final = ev["data"]
                continue  # sent after it is stored (with the message id)
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
                | {"versions": ver},
            )
            db.add(amsg)
            await db.commit()
            final["message_id"] = amsg.id
        get_metrics().turn(
            final["decision"], collection, time.perf_counter() - t0, final["timings"], final["attempts"] > 0
        )
        final["conversation_id"] = conv_id
        yield {"event": "final", "data": json.dumps(final)}
        if handler:
            flush()

    return EventSourceResponse(events(), ping=15)
