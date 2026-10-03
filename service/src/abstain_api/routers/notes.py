"""Notes in a notebook: written by the user or saved from an answer (with its citations)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user
from ..db import get_session
from ..models import Note, User
from ..schemas import NoteIn, NoteOut, NoteUpdate
from ._access import readable_collection

router = APIRouter(prefix="/collections/{collection_id}/notes", tags=["notes"])


async def _own_note(session: AsyncSession, collection_id: str, note_id: str, user: User) -> Note:
    n = await session.get(Note, note_id)
    if n is None or n.collection_id != collection_id or n.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "note not found")
    return n


@router.get("", response_model=list[NoteOut], summary="The caller's notes in this notebook (pinned first)")
async def list_notes(
    collection_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    await readable_collection(session, collection_id, user)
    q = (
        select(Note)
        .where(Note.collection_id == collection_id, Note.user_id == user.id)
        .order_by(Note.pinned.desc(), Note.updated_at.desc())
    )
    return (await session.execute(q)).scalars().all()


@router.post("", response_model=NoteOut, status_code=201, summary="Create a note (or save an answer as a note)")
async def create_note(
    collection_id: str, body: NoteIn, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    await readable_collection(session, collection_id, user)
    n = Note(collection_id=collection_id, user_id=user.id, **body.model_dump())
    session.add(n)
    await session.commit()
    await session.refresh(n)
    return n


@router.patch("/{note_id}", response_model=NoteOut, summary="Edit, rename, pin or unpin a note")
async def update_note(
    collection_id: str,
    note_id: str,
    body: NoteUpdate,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    n = await _own_note(session, collection_id, note_id, user)
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(n, k, v)
    await session.commit()
    await session.refresh(n)
    return n


@router.delete("/{note_id}", status_code=204, summary="Delete a note")
async def delete_note(
    collection_id: str, note_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    n = await _own_note(session, collection_id, note_id, user)
    await session.delete(n)
    await session.commit()
