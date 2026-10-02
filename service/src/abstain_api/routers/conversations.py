from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..agent_runtime import get_runtime
from ..auth import current_user
from ..db import get_session
from ..models import Conversation, User
from ..schemas import ConversationDetail, ConversationOut
from ..settings import get_settings
from ._access import own_conversation

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationOut], summary="The caller's conversations, newest first")
async def list_conversations(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    q = (
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return (await session.execute(q)).scalars().all()


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    summary="Messages of one conversation (+ the agent's checkpointed state with ?include_state=true)",
)
async def get_conversation(
    conversation_id: str,
    include_state: bool = False,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    await own_conversation(session, conversation_id, user)
    conv = (
        await session.execute(
            select(Conversation).where(Conversation.id == conversation_id).options(selectinload(Conversation.messages))
        )
    ).scalar_one()
    state = None
    if include_state:
        state = await asyncio.to_thread(get_runtime(get_settings()).state, conversation_id)
    return ConversationDetail.model_validate(conv, from_attributes=True).model_copy(update={"agent_state": state})
