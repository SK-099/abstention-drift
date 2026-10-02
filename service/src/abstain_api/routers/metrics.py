"""GET /metrics/abstention: product-level abstention dashboard data (admin), and GET /metrics (Prometheus)."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import String, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import admin_user
from ..db import get_session
from ..models import Conversation, EvalRun, Feedback, Message, User, now

router = APIRouter(tags=["metrics"])
TURN_BUCKETS = [("1", 1, 1), ("2-4", 2, 4), ("5-7", 5, 7), ("8+", 8, 10_000)]


def _rates(rows) -> dict:
    tot = sum(rows.values())
    return {
        "total": tot,
        **{k: rows.get(k, 0) for k in ("answer", "abstain", "clarify")},
        "abstention_rate": round((rows.get("abstain", 0) + rows.get("clarify", 0)) / tot, 4) if tot else None,
    }


@router.get(
    "/metrics/abstention",
    summary="Abstention monitoring: refusal rate by day / turn / collection, "
    "user-reported wrong refusals, latency, last evaluation run",
)
async def abstention_metrics(
    days: int = Query(30, ge=1, le=365), _: User = Depends(admin_user), session: AsyncSession = Depends(get_session)
):
    since = now() - timedelta(days=days)
    A = select(Message).where(Message.role == "assistant", Message.created_at >= since).subquery()

    overall = dict((await session.execute(select(A.c.decision, func.count()).group_by(A.c.decision))).all())

    day = func.date(A.c.created_at, type_=String)
    daily = {}
    for d, dec, n in (await session.execute(select(day, A.c.decision, func.count()).group_by(day, A.c.decision))).all():
        daily.setdefault(str(d), {})[dec] = n

    by_turn = {}
    for name, lo, hi in TURN_BUCKETS:
        q = select(A.c.decision, func.count()).where(A.c.turn.between(lo, hi)).group_by(A.c.decision)
        by_turn[name] = _rates(dict((await session.execute(q)).all()))

    by_col = {}
    q = (
        select(Conversation.collection_id, A.c.decision, func.count())
        .join(Conversation, Conversation.id == A.c.conversation_id)
        .group_by(Conversation.collection_id, A.c.decision)
    )
    for col, dec, n in (await session.execute(q)).all():
        by_col.setdefault(col, {})[dec] = n

    fb_q = (
        select(Feedback.reason, A.c.decision, func.count())
        .join(A, A.c.id == Feedback.message_id)
        .where(Feedback.rating == -1)
        .group_by(Feedback.reason, A.c.decision)
    )
    fb = (await session.execute(fb_q)).all()
    neg = sum(n for _, _, n in fb)
    pos = (
        await session.execute(
            select(func.count())
            .select_from(Feedback)
            .join(A, A.c.id == Feedback.message_id)
            .where(Feedback.rating == 1)
        )
    ).scalar_one()
    refusals = overall.get("abstain", 0) + overall.get("clarify", 0)
    wrong_ref = sum(n for r, d, n in fb if r == "wrong_refusal" or (r is None and d in ("abstain", "clarify")))
    wrong_ans = sum(n for r, d, n in fb if r in ("wrong_answer", "bad_citation") or (r is None and d == "answer"))

    lat = sorted(x for (x,) in (await session.execute(select(A.c.latency_ms).where(A.c.latency_ms.isnot(None)))).all())
    pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None
    retried = (
        await session.execute(
            select(func.count()).select_from(A).where(case((A.c.meta["attempts"].as_integer() > 0, 1), else_=0) == 1)
        )
    ).scalar_one()

    last = (await session.execute(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(1))).scalar_one_or_none()
    return {
        "window_days": days,
        "overall": _rates(overall),
        "daily": {d: _rates(v) for d, v in sorted(daily.items())},
        "by_turn": by_turn,
        "by_collection": {c: _rates(v) for c, v in sorted(by_col.items())},
        "retry_rate": round(retried / sum(overall.values()), 4) if overall else None,
        "feedback": {
            "positive": pos,
            "negative": neg,
            "reported_wrong_refusals": wrong_ref,
            "reported_wrong_answers": wrong_ans,
            "reported_wrong_refusal_share": round(wrong_ref / refusals, 4) if refusals else None,
        },
        "latency_ms": {"p50": pct(0.5), "p95": pct(0.95)},
        "last_eval": None
        if last is None
        else {
            "kind": last.kind,
            "passed": last.passed,
            "metrics": last.metrics,
            "versions": last.versions,
            "created_at": last.created_at.isoformat(),
        },
    }


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
