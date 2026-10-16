"""Studio: generate briefing docs, study guides, FAQs, timelines, mind maps, flashcards, quizzes and audio
overviews from a notebook's (ticked) sources. Runs in the worker; one LLM call per output (plus Kokoro for audio)."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .generation import KINDS, get_llm, notebook_material, run_studio, structure
from .models import Artifact, Collection, Document
from .retrieval import QdrantStore, get_store
from .settings import Settings

log = logging.getLogger(__name__)


def audio_path(s: Settings, a: Artifact) -> Path:
    return Path(s.upload_dir) / a.collection_id / "audio" / f"{a.id}.wav"


def generate_artifact(session: Session, artifact_id: str, s: Settings, store: QdrantStore | None = None) -> Artifact:
    a = session.get(Artifact, artifact_id)
    if a is None:
        raise ValueError(f"artifact {artifact_id} not found")
    a.status, a.error = "running", None
    session.commit()
    try:
        q = select(Document).where(Document.collection_id == a.collection_id, Document.status == "indexed")
        if a.document_ids:
            q = q.where(Document.id.in_(a.document_ids))
        docs = session.execute(q.order_by(Document.created_at)).scalars().all()
        if not docs:
            raise ValueError("no indexed sources to work from")
        store = store or get_store(s)
        material, sources_ = notebook_material(
            [(d.id, d.filename, d.guide, store.document_passages(d.id)) for d in docs[:20]], s.notebook_context_chars
        )
        a.data = {"stage": "writing", "sources": sources_}  # progress for the Studio view
        session.commit()
        from .model_registry import REQUEST_MODEL

        tok = REQUEST_MODEL.set(a.model)
        try:
            text = run_studio(get_llm(s, a.model), a.kind, material, a.instructions)
        finally:
            REQUEST_MODEL.reset(tok)
        md, data = structure(a.kind, text)
        nb = session.get(Collection, a.collection_id)
        a.title = a.title or f"{KINDS[a.kind]} · {nb.name if nb else a.collection_id}"
        a.content, a.data = md, {**data, "sources": sources_, "raw": text[:20000]}
        empty = {
            "faq": not data.get("items"),
            "flashcards": not data.get("cards"),
            "quiz": not data.get("questions"),
            "audio": not data.get("turns"),
            "mindmap": not data.get("tree", {}).get("children"),
        }.get(a.kind, not md.strip())
        if empty:
            raise ValueError("the model's output could not be read; try again or choose a larger model")
        if a.kind == "audio":
            a.data = {**a.data, "stage": "speaking", "progress": [0, len(data["turns"])]}
            session.commit()  # the script can be read while it is being spoken
            speak_artifact(a, s, session)
        a.data = {k: v for k, v in a.data.items() if k not in ("stage", "progress")}
        a.status = "ready"
    except Exception as e:
        log.exception("studio %s %s failed", a.kind, artifact_id)
        a.status, a.error = "failed", str(e)[:1000]
    session.commit()
    return a


def speak_artifact(a: Artifact, s: Settings, session: Session | None = None) -> None:
    """Turn an audio overview's script into speech. If the voice service is not running the script stays
    readable and the reason is stored (data.audio_error); the user can try again later."""
    from .audio import TTSUnavailable, synthesize

    try:

        def progress(done: int, total: int) -> None:
            if session is not None:
                a.data = {**a.data, "stage": "speaking", "progress": [done, total]}
                session.commit()

        secs = synthesize(a.data.get("turns") or [], audio_path(s, a), s, on_turn=progress)
        a.data = {**a.data, "audio": True, "duration_s": round(secs, 1), "audio_error": None}
    except TTSUnavailable as e:
        a.data = {**a.data, "audio": False, "audio_error": str(e)}
    except Exception as e:
        log.exception("speaking %s failed", a.id)
        a.data = {**a.data, "audio": False, "audio_error": f"speech synthesis failed: {str(e)[:300]}"}


def respeak(session: Session, artifact_id: str, s: Settings) -> Artifact:
    a = session.get(Artifact, artifact_id)
    if a is not None and a.kind == "audio" and a.data.get("turns"):
        a.status = "running"
        a.data = {**a.data, "stage": "speaking", "progress": [0, len(a.data["turns"])]}
        session.commit()
        speak_artifact(a, s, session)
        a.data = {k: v for k, v in a.data.items() if k not in ("stage", "progress")}
        a.status = "ready"
        session.commit()
    return a
