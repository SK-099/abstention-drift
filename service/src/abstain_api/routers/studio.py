"""Studio outputs of a notebook: generated in the background from the (ticked) sources."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user
from ..db import get_session
from ..models import Artifact, Document, User
from ..schemas import ArtifactIn, ArtifactOut
from ..settings import get_settings
from ._access import readable_collection, writable_collection

router = APIRouter(prefix="/collections/{collection_id}/studio", tags=["studio"])


async def _artifact(session: AsyncSession, collection_id: str, artifact_id: str) -> Artifact:
    a = await session.get(Artifact, artifact_id)
    if a is None or a.collection_id != collection_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "studio output not found")
    return a


def _run(task, artifact_id: str):
    if get_settings().celery_eager:
        return asyncio.to_thread(task.apply, args=[artifact_id])
    task.delay(artifact_id)
    return asyncio.sleep(0)


@router.post("", response_model=ArtifactOut, status_code=202, summary="Generate a Studio output from the sources")
async def create_artifact(
    collection_id: str,
    body: ArtifactIn,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    from ..model_registry import get_registry
    from ..worker import studio_task

    s = get_settings()
    c = await writable_collection(session, collection_id, user)
    if c.is_public:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "the Studio works on your own notebooks, not built-in corpora")
    q = select(Document.id).where(Document.collection_id == collection_id, Document.status == "indexed")
    if body.document_ids is not None:
        q = q.where(Document.id.in_(body.document_ids))
    ids = (await session.execute(q)).scalars().all()
    if not ids:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "tick at least one indexed source")
    try:
        model = get_registry(s).resolve(body.model) if body.model else None
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    a = Artifact(
        collection_id=collection_id,
        user_id=user.id,
        kind=body.kind,
        document_ids=sorted(ids) if body.document_ids is not None else [],
        model=model,
        instructions=body.instructions,
        data={},
    )
    session.add(a)
    await session.commit()
    await _run(studio_task, a.id)
    await session.refresh(a)
    return a


@router.get("", response_model=list[ArtifactOut], summary="Studio outputs of this notebook (newest first)")
async def list_artifacts(
    collection_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
):
    await readable_collection(session, collection_id, user)
    q = select(Artifact).where(Artifact.collection_id == collection_id).order_by(Artifact.created_at.desc())
    return (await session.execute(q)).scalars().all()


@router.get("/{artifact_id}", response_model=ArtifactOut, summary="One Studio output")
async def get_artifact(
    collection_id: str,
    artifact_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    await readable_collection(session, collection_id, user)
    return await _artifact(session, collection_id, artifact_id)


@router.get("/{artifact_id}/audio", summary="The audio overview as WAV")
async def artifact_audio(
    collection_id: str,
    artifact_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    from ..studio import audio_path

    await readable_collection(session, collection_id, user)
    a = await _artifact(session, collection_id, artifact_id)
    p = audio_path(get_settings(), a)
    if a.kind != "audio" or not p.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no audio for this output")
    return FileResponse(p, media_type="audio/wav", filename=f"{a.title or 'audio overview'}.wav")


@router.post("/{artifact_id}/speak", response_model=ArtifactOut, status_code=202, summary="Speak the script again")
async def speak_again(
    collection_id: str,
    artifact_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    from ..worker import speak_task

    await writable_collection(session, collection_id, user)
    a = await _artifact(session, collection_id, artifact_id)
    if a.kind != "audio" or not (a.data or {}).get("turns"):
        raise HTTPException(status.HTTP_409_CONFLICT, "this output has no script to speak")
    await _run(speak_task, a.id)
    await session.refresh(a)
    return a


@router.delete("/{artifact_id}", status_code=204, summary="Delete a Studio output")
async def delete_artifact(
    collection_id: str,
    artifact_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    from ..studio import audio_path

    await writable_collection(session, collection_id, user)
    a = await _artifact(session, collection_id, artifact_id)
    Path(audio_path(get_settings(), a)).unlink(missing_ok=True)
    await session.delete(a)
    await session.commit()
