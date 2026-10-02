from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user
from ..db import get_session
from ..models import Conversation, Feedback, Message, User
from ..schemas import FeedbackIn, FeedbackOut
from ..telemetry import get_metrics

router = APIRouter(tags=["feedback"])


@router.post(
    "/feedback",
    response_model=FeedbackOut,
    status_code=201,
    summary="Thumbs up/down on an assistant message (e.g. reason=wrong_refusal); re-posting updates it",
)
async def post_feedback(
    body: FeedbackIn, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    msg = await session.get(Message, body.message_id)
    conv = await session.get(Conversation, msg.conversation_id) if msg else None
    if msg is None or conv is None or conv.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "message not found")
    if msg.role != "assistant":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "feedback is for assistant messages")
    fb = (
        await session.execute(select(Feedback).where(Feedback.message_id == msg.id, Feedback.user_id == user.id))
    ).scalar_one_or_none()
    if fb is None:
        fb = Feedback(message_id=msg.id, user_id=user.id)
        session.add(fb)
    fb.rating, fb.reason, fb.comment = body.rating, body.reason, body.comment
    await session.commit()
    await session.refresh(fb)
    get_metrics().feedback.add(
        1, {"rating": str(body.rating), "decision": msg.decision or "", "reason": body.reason or ""}
    )
    return fb
