from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Query
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..agent_runtime import get_runtime
from ..auth import current_user
from ..db import get_session
from ..models import Conversation, Message, User
from ..schemas import ConversationDetail, ConversationOut
from ..settings import get_settings
from ._access import own_conversation

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationOut], summary="The caller's conversations, newest first")
async def list_conversations(
    collection: str | None = Query(None, description="only this notebook's chats"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    q = select(Conversation).where(Conversation.user_id == user.id)
    if collection:
        q = q.where(Conversation.collection_id == collection)
    q = q.order_by(Conversation.updated_at.desc()).limit(limit).offset(offset)
    return (await session.execute(q)).scalars().all()


@router.delete("/{conversation_id}", status_code=204, summary="Delete a chat")
async def delete_conversation(
    conversation_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    await own_conversation(session, conversation_id, user)
    await session.execute(delete(Message).where(Message.conversation_id == conversation_id))
    await session.execute(delete(Conversation).where(Conversation.id == conversation_id))
    await session.commit()
    try:  # also forget the agent's checkpointed state for this thread
        await asyncio.to_thread(get_runtime(get_settings()).agent.graph.checkpointer.delete_thread, conversation_id)
    except Exception:
        pass


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
