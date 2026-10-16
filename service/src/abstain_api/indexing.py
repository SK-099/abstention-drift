"""Index one uploaded document: parse -> chunk -> injection screen -> embed -> upsert into Qdrant."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from .guardrails import is_injection
from .models import Document
from .parsing import parse_and_chunk
from .retrieval import Embedder, QdrantStore, get_embedder, get_store
from .settings import Settings
from .sources import SourceError, fetch_web_page, fetch_youtube

log = logging.getLogger(__name__)


def index_document(
    session: Session, doc_id: str, s: Settings, store: QdrantStore | None = None, embedder: Embedder | None = None
) -> Document:
    doc = session.get(Document, doc_id)
    if doc is None:
        raise ValueError(f"document {doc_id} not found")
    doc.status, doc.error = "processing", None
    session.commit()
    try:
        if not doc.path:  # web page / YouTube video: fetch it first
            fetch_source(doc, s)
            session.commit()
        chunks = parse_and_chunk(doc.path, s)
        if not chunks:
            raise ValueError("no text could be extracted")
        kept = [c for c in chunks if not is_injection(c["text"])]
        doc.n_flagged = len(chunks) - len(kept)
        if doc.n_flagged and doc.n_flagged / len(chunks) > 0.5:
            doc.status, doc.error = (
                "rejected",
                (
                    f"{doc.n_flagged} of {len(chunks)} passages look like instructions "
                    "to the AI (possible prompt injection)"
                ),
            )
            session.commit()
            return doc
        store = store or get_store(s)
        store.delete_document(doc.id)  # re-indexing replaces old passages
        stem = Path(doc.path).stem  # stored as <doc id>.<ext>: show the original file name instead
        passages = [
            {
                "id": f"{doc.id}-{i}",
                "title": doc.filename if c["title"] == stem else c["title"],
                "text": c["text"],
                "source_document": doc.id,
                "page": c.get("page"),
                "section": c.get("section") or None,
                "seq": i,
            }
            for i, c in enumerate(kept)
        ]
        vecs = (embedder or get_embedder(s)).embed_documents([p["text"] for p in passages])
        doc.n_passages = store.upsert(doc.collection_id, passages, vecs)
        doc.status = "indexed"
        doc.guide_status = "pending" if s.source_guides else "none"
    except SourceError as e:
        doc.status, doc.error = "failed", str(e)
    except Exception as e:
        log.exception("indexing %s failed", doc_id)
        doc.status, doc.error = "failed", str(e)[:1000]
    session.commit()
    return doc


def fetch_source(doc: Document, s: Settings) -> None:
    """Download a web page or a YouTube transcript into the upload folder and point the document at it."""
    dest = Path(s.upload_dir) / doc.collection_id / doc.id
    dest.parent.mkdir(parents=True, exist_ok=True)
    fetch = fetch_youtube if doc.source_type == "youtube" else fetch_web_page
    path, title = fetch(doc.source_url or "", dest, s)
    doc.path, doc.size_bytes = str(path), path.stat().st_size
    doc.filename = (title or doc.filename)[:255]


def make_guide(
    session: Session, doc_id: str, s: Settings, store: QdrantStore | None = None, model: str | None = None
) -> Document:
    """Source guide (summary, key topics, suggested questions) from the indexed passages."""
    from .generation import get_llm, source_guide

    doc = session.get(Document, doc_id)
    if doc is None or doc.status != "indexed":
        return doc
    try:
        passages = (store or get_store(s)).document_passages(doc.id)
        doc.guide = source_guide(get_llm(s, model), doc.filename, passages, s)
        doc.guide_status = "ready"
    except Exception as e:
        log.warning("source guide for %s failed: %s", doc_id, e)
        doc.guide_status = "failed"
    session.commit()
    return doc
