"""Share a notebook with people (viewer / editor, by e-mail or user id) or with anyone who has the link."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user
from ..db import get_session
from ..models import CollectionMember, User
from ..schemas import MemberIn, SharingOut, SharingUpdate
from ._access import owned_collection, readable_collection

router = APIRouter(prefix="/collections/{collection_id}/sharing", tags=["sharing"])


async def _state(session: AsyncSession, cid: str, link_access: str, owner_id: str | None) -> dict:
    ms = (
        (
            await session.execute(
                select(CollectionMember)
                .where(CollectionMember.collection_id == cid)
                .order_by(CollectionMember.created_at)
            )
        )
        .scalars()
        .all()
    )
    owner = await session.get(User, owner_id) if owner_id else None
    return {
        "link_access": link_access,
        "owner": (owner.email or owner.external_id) if owner else None,
        "members": [{"principal": m.principal, "role": m.role} for m in ms],
    }


@router.get("", response_model=SharingOut, summary="Who this notebook is shared with")
async def get_sharing(
    collection_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    c = await readable_collection(session, collection_id, user)
    return await _state(session, c.id, c.link_access, c.owner_id)


@router.put("/members", response_model=SharingOut, summary="Share with a person, or change their role")
async def put_member(
    collection_id: str, body: MemberIn, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    c = await owned_collection(session, collection_id, user)
    if c.is_public:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "built-in collections are already public")
    p = body.principal.strip().lower()
    if p in {x.lower() for x in (user.email, user.external_id) if x}:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "you already own this notebook")
    m = (
        await session.execute(
            select(CollectionMember).where(CollectionMember.collection_id == c.id, CollectionMember.principal == p)
        )
    ).scalar_one_or_none()
    if m is None:
        session.add(CollectionMember(collection_id=c.id, principal=p, role=body.role, added_by=user.id))
    else:
        m.role = body.role
    await session.commit()
    return await _state(session, c.id, c.link_access, c.owner_id)


@router.delete("/members/{principal}", response_model=SharingOut, summary="Stop sharing with a person")
async def delete_member(
    collection_id: str, principal: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    c = await owned_collection(session, collection_id, user)
    await session.execute(
        delete(CollectionMember).where(
            CollectionMember.collection_id == c.id, CollectionMember.principal == principal.strip().lower()
        )
    )
    await session.commit()
    return await _state(session, c.id, c.link_access, c.owner_id)


@router.patch("", response_model=SharingOut, summary="General access: restricted, or anyone signed in with the link")
async def update_sharing(
    collection_id: str,
    body: SharingUpdate,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    c = await owned_collection(session, collection_id, user)
    if c.is_public:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "built-in collections are already public")
    c.link_access = body.link_access
    await session.commit()
    return await _state(session, c.id, c.link_access, c.owner_id)
