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
from ..models import Collection, CollectionMember, Conversation, Document, User
from ..schemas import CollectionIn, CollectionOut, CollectionUpdate, DocumentOut, PassageOut, SourceIn
from ..settings import get_settings
from ._access import owned_collection, principals, readable_collection, role_of, writable_collection

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


@router.get("", response_model=list[CollectionOut], summary="Collections the caller can search (with their role)")
async def list_collections(user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    mine = dict(
        (
            await session.execute(
                select(CollectionMember.collection_id, CollectionMember.role).where(
                    CollectionMember.principal.in_(principals(user))
                )
            )
        ).all()
    )
    q = select(Collection).order_by(Collection.created_at)
    if not user.is_admin:
        q = q.where(or_(Collection.is_public.is_(True), Collection.owner_id == user.id, Collection.id.in_(list(mine))))
    shared = set((await session.execute(select(CollectionMember.collection_id).distinct())).scalars().all())
    out = []
    for c in (await session.execute(q)).scalars().all():
        if c.owner_id == user.id:
            role = "owner"
        elif c.id in mine:
            role = mine[c.id]
        else:
            role = "public" if c.is_public else "owner"  # admins see every notebook
        out.append(
            CollectionOut.model_validate(c).model_copy(
                update={"role": role, "shared": c.id in shared or c.link_access != "none"}
            )
        )
    return out


@router.get("/{collection_id}", response_model=CollectionOut, summary="One notebook (also one shared by link)")
async def get_collection(
    collection_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    c = await readable_collection(session, collection_id, user)
    role = await role_of(session, c, user)
    n = (await session.execute(select(CollectionMember.id).where(CollectionMember.collection_id == c.id))).first()
    return CollectionOut.model_validate(c).model_copy(
        update={"role": role, "shared": n is not None or c.link_access != "none"}
    )


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

    await _start_indexing(doc.id)
    await session.refresh(doc)
    return doc


async def _start_indexing(doc_id: str) -> None:
    from ..worker import index_document_task

    if get_settings().celery_eager:  # tests / no-Redis dev: index synchronously in a thread
        await asyncio.to_thread(index_document_task.apply, args=[doc_id])
    else:
        index_document_task.delay(doc_id)


@router.post(
    "/{collection_id}/sources",
    response_model=DocumentOut,
    status_code=202,
    summary="Add a web page, a YouTube video (its transcript) or pasted text as a source",
)
@limiter.limit(lambda: get_settings().rate_limit_upload)
async def add_source(
    request: Request,
    collection_id: str,
    body: SourceIn,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    from ..sources import SourceError, check_public_url, youtube_id

    s = get_settings()
    await writable_collection(session, collection_id, user)
    doc = Document(collection_id=collection_id, owner_id=user.id, source_type=body.type, path="", size_bytes=0)
    if body.type == "text":
        data = body.text.strip().encode("utf-8")
        if len(data) > s.max_upload_mb * 1024 * 1024:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"text is larger than {s.max_upload_mb} MB")
        doc.filename = re.sub(r"[\x00-\x1f<>]", "", (body.title or "").strip())[:200] or "Pasted text"
        doc.content_type, doc.size_bytes = "text/plain", len(data)
    else:
        url = body.url.strip()
        if body.type == "url" and youtube_id(url):
            doc.source_type = "youtube"
        try:
            if doc.source_type == "youtube" and not youtube_id(url):
                raise SourceError("not a YouTube video link")
            if doc.source_type == "url":
                await asyncio.to_thread(check_public_url, url, s)
        except SourceError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
        doc.source_url = url
        doc.filename = (body.title or url)[:255]
        doc.content_type = "text/html" if doc.source_type == "url" else "text/markdown"
    session.add(doc)
    await session.flush()
    if body.type == "text":
        dest = Path(s.upload_dir) / collection_id / f"{doc.id}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        doc.path = str(dest)
    await session.commit()
    await _start_indexing(doc.id)
    await session.refresh(doc)
    return doc


@router.post(
    "/{collection_id}/documents/{document_id}/guide",
    response_model=DocumentOut,
    status_code=202,
    summary="Write the source guide (summary, topics, suggested questions) again",
)
async def regenerate_guide(
    collection_id: str,
    document_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    from ..worker import source_guide_task

    await writable_collection(session, collection_id, user)
    d = await _document(session, collection_id, document_id)
    if d.status != "indexed":
        raise HTTPException(status.HTTP_409_CONFLICT, "the source is not indexed yet")
    d.guide_status = "pending"
    await session.commit()
    if get_settings().celery_eager:
        await asyncio.to_thread(source_guide_task.apply, args=[d.id])
    else:
        source_guide_task.delay(d.id)
    await session.refresh(d)
    return d


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
    inline = Path(d.path).suffix.lower() in {".pdf", ".txt", ".md"}
    return FileResponse(
        d.path,
        media_type=(d.content_type or "application/octet-stream") if inline else "application/octet-stream",
        filename=d.filename if Path(d.filename).suffix else d.filename + Path(d.path).suffix,
        content_disposition_type="inline" if inline else "attachment",
        # an uploaded or fetched HTML page must never run scripts on the API's origin
        headers={"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff"},
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
    c = await owned_collection(session, collection_id, user)
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
