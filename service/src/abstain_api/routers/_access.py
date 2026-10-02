"""Access checks shared by the routers."""

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Collection, Conversation, User


async def readable_collection(session: AsyncSession, cid: str, user: User) -> Collection:
    c = await session.get(Collection, cid)
    if c is None or not (c.is_public or c.owner_id == user.id or user.is_admin):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"collection '{cid}' not found")
    return c


async def writable_collection(session: AsyncSession, cid: str, user: User) -> Collection:
    c = await readable_collection(session, cid, user)
    if not (c.owner_id == user.id or user.is_admin):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "only the owner can add documents to this collection")
    return c


async def own_conversation(session: AsyncSession, conv_id: str, user: User) -> Conversation:
    c = await session.get(Conversation, conv_id)
    if c is None or c.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return c
