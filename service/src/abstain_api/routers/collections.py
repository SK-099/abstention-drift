from __future__ import annotations

import asyncio
import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user
from ..db import get_session
from ..guardrails import limiter, validate_upload
from ..models import Collection, Conversation, Document, User
from ..schemas import CollectionIn, CollectionOut, CollectionUpdate, DocumentOut, PassageOut
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
    from ..telemetry import get_metrics

    get_metrics().prime_collection(c.id)
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


async def _document(session: AsyncSession, collection_id: str, document_id: str) -> Document:
    d = await session.get(Document, document_id)
    if d is None or d.collection_id != collection_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return d


@router.get(
    "/{collection_id}/documents/{document_id}/passages",
    response_model=list[PassageOut],
    summary="The indexed passages of one document, in reading order (source viewer)",
)
async def document_passages(
    collection_id: str,
    document_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    await readable_collection(session, collection_id, user)
    await _document(session, collection_id, document_id)
    from ..retrieval import get_store

    return await asyncio.to_thread(get_store(get_settings()).document_passages, document_id)


@router.get("/{collection_id}/documents/{document_id}/file", summary="Download / view the original file")
async def document_file(
    collection_id: str,
    document_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    await readable_collection(session, collection_id, user)
    d = await _document(session, collection_id, document_id)
    if not d.path or not Path(d.path).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "file no longer stored")
    return FileResponse(
        d.path,
        media_type=d.content_type or "application/octet-stream",
        filename=d.filename,
        content_disposition_type="inline",
    )


@router.delete("/{collection_id}/documents/{document_id}", status_code=204, summary="Remove a source")
async def delete_document(
    collection_id: str,
    document_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    await writable_collection(session, collection_id, user)
    d = await _document(session, collection_id, document_id)
    from ..retrieval import get_store

    await asyncio.to_thread(get_store(get_settings()).delete_document, d.id)
    if d.path:
        Path(d.path).unlink(missing_ok=True)
    await session.delete(d)
    await session.commit()


@router.patch("/{collection_id}", response_model=CollectionOut, summary="Rename a notebook")
async def update_collection(
    collection_id: str,
    body: CollectionUpdate,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    c = await writable_collection(session, collection_id, user)
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(c, k, v)
    await session.commit()
    await session.refresh(c)
    return c


@router.delete("/{collection_id}", status_code=204, summary="Delete a notebook with its sources, chats and notes")
async def delete_collection(
    collection_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    c = await writable_collection(session, collection_id, user)
    if c.is_public:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "built-in collections cannot be deleted")
    from ..retrieval import get_store

    store = get_store(get_settings())
    docs = (await session.execute(select(Document).where(Document.collection_id == collection_id))).scalars().all()
    for d in docs:
        await asyncio.to_thread(store.delete_document, d.id)
        if d.path:
            Path(d.path).unlink(missing_ok=True)
    await session.execute(delete(Conversation).where(Conversation.collection_id == collection_id))
    await session.delete(c)
    await session.commit()
