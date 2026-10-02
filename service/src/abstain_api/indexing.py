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
            }
            for i, c in enumerate(kept)
        ]
        vecs = (embedder or get_embedder(s)).embed_documents([p["text"] for p in passages])
        doc.n_passages = store.upsert(doc.collection_id, passages, vecs)
        doc.status = "indexed"
    except Exception as e:
        log.exception("indexing %s failed", doc_id)
        doc.status, doc.error = "failed", str(e)[:1000]
    session.commit()
    return doc
