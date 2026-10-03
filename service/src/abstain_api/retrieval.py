"""Hybrid retrieval on Qdrant: dense (BGE-small) + sparse BM25 vectors fused with RRF, filtered by
collection, then re-ranked by the BGE cross-encoder (in-process or Text Embeddings Inference).

Replaces the local FAISS + bm25s indexes of the research code. One Qdrant collection holds every
passage; the `collection` payload field (indexed keyword) separates MTRAG corpora and user uploads.
Sparse vectors carry BM25 term-frequency weights; Qdrant applies IDF itself (Modifier.IDF).
"""

from __future__ import annotations

import re
import uuid
import zlib
from collections import Counter
from collections.abc import Iterable

import httpx

from .settings import Settings

# ---------------------------------------------------------------- sparse BM25
_STOP = set(
    """a about above after again against all am an and any are as at be because been before being below between
both but by can did do does doing down during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which while who whom why
will with you your yours yourself yourselves""".split()
)
_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _tid(tok: str) -> int:
    return zlib.crc32(tok.encode()) & 0x7FFFFFFF


def bm25_document_vector(text: str, avgdl: float, k1: float = 1.2, b: float = 0.75) -> tuple[list[int], list[float]]:
    toks = tokenize(text)
    dl = max(len(toks), 1)
    tf: Counter = Counter()
    for t in toks:
        tf[_tid(t)] += 1
    idx = sorted(tf)
    return idx, [tf[i] * (k1 + 1) / (tf[i] + k1 * (1 - b + b * dl / avgdl)) for i in idx]


def bm25_query_vector(text: str) -> tuple[list[int], list[float]]:
    idx = sorted({_tid(t) for t in tokenize(text)})
    return idx, [1.0] * len(idx)


# ---------------------------------------------------------------- dense embeddings
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, s: Settings):
        self.s = s
        self._model = None

    def _local(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.s.dense_model, device=self.s.device)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.s.tei_embed_url:
            r = httpx.post(
                f"{self.s.tei_embed_url.rstrip('/')}/embed", json={"inputs": texts, "normalize": True}, timeout=60
            )
            r.raise_for_status()
            return r.json()
        return self._local().encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._encode([BGE_QUERY_PREFIX + text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)


class Reranker:
    """Relevance in [0, 1] for (query, passage) pairs."""

    def __init__(self, s: Settings):
        self.s = s
        self._model = None

    def scores(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if self.s.tei_rerank_url:
            r = httpx.post(
                f"{self.s.tei_rerank_url.rstrip('/')}/rerank",
                json={"query": query, "texts": texts, "raw_scores": False, "truncate": True},
                timeout=60,
            )
            r.raise_for_status()
            out = [0.0] * len(texts)
            for item in r.json():
                out[item["index"]] = float(item["score"])
            return out
        if self._model is None:
            from abstain.retrieval import Reranker as LocalReranker

            self._model = LocalReranker(self.s.reranker_model, self.s.device, 256)
        ranked = self._model.rerank(query, [{"text": t, "i": i} for i, t in enumerate(texts)])
        out = [0.0] * len(texts)
        for p in ranked:
            out[p["i"]] = p["score"]
        return out


# ---------------------------------------------------------------- Qdrant
def point_id(collection: str, passage_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}/{passage_id}"))


class QdrantStore:
    DENSE, SPARSE = "dense", "bm25"

    def __init__(self, s: Settings, client=None):
        from qdrant_client import QdrantClient

        self.s = s
        if client is not None:
            self.client = client
        elif s.qdrant_url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60)
        self.name = s.qdrant_collection

    def ensure_collection(self) -> None:
        from qdrant_client import models as m

        if self.client.collection_exists(self.name):
            return
        self.client.create_collection(
            self.name,
            vectors_config={self.DENSE: m.VectorParams(size=self.s.dense_dim, distance=m.Distance.COSINE)},
            sparse_vectors_config={self.SPARSE: m.SparseVectorParams(modifier=m.Modifier.IDF)},
        )
        for field in ("collection", "source_document"):
            self.client.create_payload_index(self.name, field, m.PayloadSchemaType.KEYWORD)

    def upsert(self, collection: str, passages: list[dict], dense: list[list[float]], batch: int = 256) -> int:
        """passages: dicts with id, text, title (+ optional source_document)."""
        from qdrant_client import models as m

        pts = []
        for p, vec in zip(passages, dense):
            idx, val = bm25_document_vector(f"{p.get('title', '')} {p['text']}", self.s.bm25_avgdl)
            pts.append(
                m.PointStruct(
                    id=point_id(collection, p["id"]),
                    vector={self.DENSE: vec, self.SPARSE: m.SparseVector(indices=idx, values=val)},
                    payload={
                        "collection": collection,
                        "passage_id": p["id"],
                        "title": p.get("title", ""),
                        "text": p["text"],
                        "source_document": p.get("source_document"),
                        "page": p.get("page"),
                        "section": p.get("section"),
                        "seq": p.get("seq"),
                    },
                )
            )
        for i in range(0, len(pts), batch):
            self.client.upsert(self.name, pts[i : i + batch], wait=True)
        return len(pts)

    def delete_document(self, document_id: str) -> None:
        from qdrant_client import models as m

        self.client.delete(
            self.name,
            points_selector=m.FilterSelector(
                filter=m.Filter(must=[m.FieldCondition(key="source_document", match=m.MatchValue(value=document_id))])
            ),
        )

    def hybrid_search(
        self, query: str, query_vec: list[float], collection: str, limit: int, documents=None
    ) -> list[dict]:
        from qdrant_client import models as m

        must = [m.FieldCondition(key="collection", match=m.MatchValue(value=collection))]
        if documents:  # the notebook's ticked sources only
            must.append(m.FieldCondition(key="source_document", match=m.MatchAny(any=list(documents))))
        flt = m.Filter(must=must)
        idx, val = bm25_query_vector(query)
        prefetch = [m.Prefetch(query=query_vec, using=self.DENSE, filter=flt, limit=self.s.prefetch_k)]
        if idx:
            prefetch.append(
                m.Prefetch(
                    query=m.SparseVector(indices=idx, values=val),
                    using=self.SPARSE,
                    filter=flt,
                    limit=self.s.prefetch_k,
                )
            )
        res = self.client.query_points(
            self.name, prefetch=prefetch, query=m.FusionQuery(fusion=m.Fusion.RRF), limit=limit, with_payload=True
        )
        return [
            {
                "document_id": p.payload["passage_id"],
                "title": p.payload.get("title", ""),
                "text": p.payload["text"],
                "rrf": p.score,
                **{k: p.payload.get(k) for k in ("source_document", "page", "section") if p.payload.get(k) is not None},
            }
            for p in res.points
        ]

    def document_passages(self, document_id: str) -> list[dict]:
        """All passages of one uploaded document, in reading order (for the source viewer)."""
        from qdrant_client import models as m

        flt = m.Filter(must=[m.FieldCondition(key="source_document", match=m.MatchValue(value=document_id))])
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(self.name, scroll_filter=flt, limit=256, offset=offset, with_payload=True)
            out += [
                {
                    "passage_id": q.payload["passage_id"],
                    "text": q.payload["text"],
                    "page": q.payload.get("page"),
                    "section": q.payload.get("section"),
                    "seq": q.payload.get("seq"),
                }
                for q in pts
            ]
            if offset is None:
                break
        return sorted(out, key=_reading_order)

    def count(self, collection: str | None = None) -> int:
        from qdrant_client import models as m

        flt = (
            None
            if collection is None
            else m.Filter(must=[m.FieldCondition(key="collection", match=m.MatchValue(value=collection))])
        )
        return self.client.count(self.name, count_filter=flt, exact=True).count


def _reading_order(p: dict):
    if p.get("seq") is not None:
        return (0, p["seq"])
    tail = p["passage_id"].rsplit("-", 1)[-1]
    return (1, int(tail)) if tail.isdigit() else (2, p["passage_id"])


class HybridQdrantRetriever:
    """search(query, collection, k) -> top-k passages with re-ranker scores (same shape as the research code)."""

    def __init__(self, s: Settings, store: QdrantStore, embedder: Embedder, reranker: Reranker):
        self.s, self.store, self.embedder, self.reranker = s, store, embedder, reranker

    def search(self, query: str, collection: str, k: int | None = None, documents=None) -> list[dict]:
        if documents is None:
            from .model_registry import REQUEST_DOCS

            documents = REQUEST_DOCS.get()
        cands = self.store.hybrid_search(
            query, self.embedder.embed_query(query), collection, self.s.rerank_top_n, documents=documents
        )
        scores = self.reranker.scores(query, [c["text"] for c in cands])
        ranked = sorted((dict(c, score=float(sc)) for c, sc in zip(cands, scores)), key=lambda p: -p["score"])
        return ranked[: k or self.s.final_k]


def make_lc_retriever(retriever: HybridQdrantRetriever):
    """LangChain BaseRetriever used by the LangGraph agent (agent.retriever.invoke(query, collection=...))."""
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class QdrantLCRetriever(BaseRetriever):
        backend: object

        def _get_relevant_documents(self, query: str, *, run_manager=None, collection: str = "") -> list:
            return [
                Document(
                    page_content=p["text"],
                    metadata={
                        "document_id": p["document_id"],
                        "title": p["title"],
                        "score": p["score"],
                        **{k: p[k] for k in ("source_document", "page", "section") if p.get(k) is not None},
                    },
                )
                for p in self.backend.search(query, collection)
            ]

    return QdrantLCRetriever(backend=retriever)


def batched(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


class PgVectorStore:
    """Alternative to Qdrant: pgvector (HNSW, cosine) + PostgreSQL full-text search, fused with RRF.

    Same interface as QdrantStore. Uses the service database (DATABASE_URL); needs the `vector` extension
    (the pgvector/pgvector Postgres image, or `CREATE EXTENSION vector` on managed Postgres)."""

    def __init__(self, s: Settings):
        self.s, self.table = s, f"{s.qdrant_collection}_pgv"

    def _conn(self):
        import psycopg

        return psycopg.connect(self.s.checkpoint_conninfo, autocommit=True)

    @staticmethod
    def _vec(v) -> str:
        return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"

    def ensure_collection(self) -> None:
        t = self.table
        with self._conn() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute(f"""CREATE TABLE IF NOT EXISTS {t} (
                id uuid PRIMARY KEY, collection text NOT NULL, passage_id text NOT NULL, title text,
                text text NOT NULL, source_document text, embedding vector({self.s.dense_dim}) NOT NULL,
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || text)) STORED)""")
            c.execute(f"CREATE INDEX IF NOT EXISTS {t}_hnsw ON {t} USING hnsw (embedding vector_cosine_ops)")
            c.execute(f"CREATE INDEX IF NOT EXISTS {t}_tsv ON {t} USING gin (tsv)")
            c.execute(f"CREATE INDEX IF NOT EXISTS {t}_col ON {t} (collection)")
            c.execute(f"CREATE INDEX IF NOT EXISTS {t}_doc ON {t} (source_document)")
            c.execute(f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS page integer")
            c.execute(f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS section text")
            c.execute(f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS seq integer")

    def upsert(self, collection: str, passages: list[dict], dense: list[list[float]], batch: int = 256) -> int:
        rows = [
            (
                str(point_id(collection, p["id"])),
                collection,
                p["id"],
                p.get("title", ""),
                p["text"],
                p.get("source_document"),
                self._vec(v),
                p.get("page"),
                p.get("section"),
                p.get("seq"),
            )
            for p, v in zip(passages, dense)
        ]
        with self._conn() as c, c.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table}
                    (id, collection, passage_id, title, text, source_document, embedding, page, section, seq)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, text = EXCLUDED.text,
                    source_document = EXCLUDED.source_document, embedding = EXCLUDED.embedding,
                    page = EXCLUDED.page, section = EXCLUDED.section, seq = EXCLUDED.seq""",
                rows,
            )
        return len(rows)

    def delete_document(self, document_id: str) -> None:
        with self._conn() as c:
            c.execute(f"DELETE FROM {self.table} WHERE source_document = %s", (document_id,))

    def hybrid_search(
        self, query: str, query_vec: list[float], collection: str, limit: int, documents=None
    ) -> list[dict]:
        k, t = self.s.prefetch_k, self.table
        cols = "passage_id, title, text, source_document, page, section"
        where, args = "collection = %s", [collection]
        if documents:  # the notebook's ticked sources only
            where += " AND source_document = ANY(%s)"
            args.append(list(documents))
        tsq = " OR ".join(tokenize(query)) or query
        with self._conn() as c:
            dense = c.execute(
                f"SELECT {cols} FROM {t} WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s",
                (*args, self._vec(query_vec), k),
            ).fetchall()
            sparse = c.execute(
                f"SELECT {cols} FROM {t} WHERE {where} AND tsv @@ websearch_to_tsquery('english', %s) "
                f"ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) DESC LIMIT %s",
                (*args, tsq, tsq, k),
            ).fetchall()
        score, info = {}, {}
        for ranking in (dense, sparse):
            for r, row in enumerate(ranking):
                score[row[0]] = score.get(row[0], 0.0) + 1.0 / (60 + r + 1)
                info[row[0]] = row
        top = sorted(score, key=lambda p: -score[p])[:limit]
        out = []
        for pid in top:
            _, title, text, src, page, section = info[pid]
            d = {"document_id": pid, "title": title or "", "text": text, "rrf": score[pid]}
            d.update(
                {k: v for k, v in (("source_document", src), ("page", page), ("section", section)) if v is not None}
            )
            out.append(d)
        return out

    def document_passages(self, document_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                f"SELECT passage_id, text, page, section, seq FROM {self.table} WHERE source_document = %s",
                (document_id,),
            ).fetchall()
        return sorted(
            ({"passage_id": r[0], "text": r[1], "page": r[2], "section": r[3], "seq": r[4]} for r in rows),
            key=_reading_order,
        )

    def count(self, collection: str | None = None) -> int:
        with self._conn() as c:
            if collection:
                return c.execute(f"SELECT count(*) FROM {self.table} WHERE collection = %s", (collection,)).fetchone()[
                    0
                ]
            return c.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]


# ---------------------------------------------------------------- process-wide shared instances
_shared: dict = {}


def get_store(s: Settings) -> QdrantStore | PgVectorStore:
    if "store" not in _shared:
        st = PgVectorStore(s) if s.vector_store == "pgvector" else QdrantStore(s)
        st.ensure_collection()
        _shared["store"] = st
    return _shared["store"]


def get_embedder(s: Settings) -> Embedder:
    return _shared.setdefault("embedder", Embedder(s))


def set_shared(store: QdrantStore | None = None, embedder=None) -> None:
    """Dependency injection for tests (and to drop cached clients)."""
    _shared.clear()
    if store is not None:
        _shared["store"] = store
    if embedder is not None:
        _shared["embedder"] = embedder
