"""Access checks shared by the routers.

Roles on a notebook: owner (or an admin), editor and viewer (shared by e-mail / user id), viewer through
"anyone with the link", and "public" for the built-in corpora. Viewers can read sources and Studio outputs and
chat (their chats and notes are their own); editors can also add sources and generate Studio outputs; only
the owner can share, rename the sharing settings or delete the notebook.
"""

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Collection, CollectionMember, Conversation, User


def principals(user: User) -> list[str]:
    return [p.lower() for p in (user.email, user.external_id) if p]


async def role_of(session: AsyncSession, c: Collection, user: User) -> str | None:
    if c.owner_id == user.id or user.is_admin:
        return "owner"
    m = (
        (
            await session.execute(
                select(CollectionMember.role).where(
                    CollectionMember.collection_id == c.id, CollectionMember.principal.in_(principals(user))
                )
            )
        )
        .scalars()
        .all()
    )
    if "editor" in m:
        return "editor"
    if m or c.link_access == "viewer":
        return "viewer"
    return "public" if c.is_public else None


async def readable_collection(session: AsyncSession, cid: str, user: User) -> Collection:
    c = await session.get(Collection, cid)
    if c is None or await role_of(session, c, user) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"collection '{cid}' not found")
    return c


async def writable_collection(session: AsyncSession, cid: str, user: User) -> Collection:
    c = await readable_collection(session, cid, user)
    if await role_of(session, c, user) not in ("owner", "editor"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "you can view this notebook but not change it")
    return c


async def owned_collection(session: AsyncSession, cid: str, user: User) -> Collection:
    c = await readable_collection(session, cid, user)
    if await role_of(session, c, user) != "owner":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "only the owner can do this")
    return c


async def own_conversation(session: AsyncSession, conv_id: str, user: User) -> Conversation:
    c = await session.get(Conversation, conv_id)
    if c is None or c.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return c
