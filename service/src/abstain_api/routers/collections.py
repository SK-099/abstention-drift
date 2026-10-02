from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user
from ..db import get_session
from ..guardrails import limiter, validate_upload
from ..models import Collection, Document, User
from ..schemas import CollectionIn, CollectionOut, DocumentOut
from ..settings import get_settings
from ._access import readable_collection, writable_collection

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionOut, status_code=201, summary="Create a document collection")
async def create_collection(
    body: CollectionIn, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    if await session.get(Collection, body.id):
        raise HTTPException(status.HTTP_409_CONFLICT, f"collection '{body.id}' already exists")
    if body.is_public and not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "only admins can create public collections")
    c = Collection(id=body.id, name=body.name, description=body.description, is_public=body.is_public, owner_id=user.id)
    session.add(c)
    await session.commit()
    await session.refresh(c)
    return c


@router.get("", response_model=list[CollectionOut], summary="Collections the caller can search")
async def list_collections(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    q = select(Collection).order_by(Collection.created_at)
    if not user.is_admin:
        q = q.where(or_(Collection.is_public.is_(True), Collection.owner_id == user.id))
    return (await session.execute(q)).scalars().all()


def _safe_name(name: str) -> str:
    name = Path(name or "upload").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:150] or "upload"


@router.post(
    "/{collection_id}/documents",
    response_model=DocumentOut,
    status_code=202,
    summary="Upload a PDF/HTML/DOCX/MD/TXT file; it is parsed, screened and indexed in the background",
)
@limiter.limit(lambda: get_settings().rate_limit_upload)
async def upload_document(
    request: Request,
    collection_id: str,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    s = get_settings()
    await writable_collection(session, collection_id, user)
    data = await file.read(s.max_upload_mb * 1024 * 1024 + 1)
    validate_upload(file, len(data))
    doc = Document(
        collection_id=collection_id,
        owner_id=user.id,
        filename=_safe_name(file.filename),
        content_type=file.content_type,
        size_bytes=len(data),
        path="",
    )
    session.add(doc)
    await session.flush()
    dest = Path(s.upload_dir) / collection_id / f"{doc.id}{Path(doc.filename).suffix.lower()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    doc.path = str(dest)
    await session.commit()

    from ..worker import index_document_task

    if s.celery_eager:  # tests / no-Redis dev: index synchronously in a thread
        import asyncio

        await asyncio.to_thread(index_document_task.apply, args=[doc.id])
    else:
        index_document_task.delay(doc.id)
    await session.refresh(doc)
    return doc


@router.get(
    "/{collection_id}/documents", response_model=list[DocumentOut], summary="Documents and their indexing status"
)
async def list_documents(
    collection_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    await readable_collection(session, collection_id, user)
    q = select(Document).where(Document.collection_id == collection_id).order_by(Document.created_at)
    return (await session.execute(q)).scalars().all()
