"""Hybrid retrieval: BM25 + dense (BGE) fused with reciprocal rank fusion, then cross-encoder re-ranking.

One index per collection is built once by scripts/build_indexes.py and saved under indexes/<collection>/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import iter_corpus, parent_doc

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum_r 1 / (k + rank_r(d))."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))


class PassageStore:
    def __init__(self, passages: dict[str, dict]):
        self.p = passages

    @classmethod
    def build(cls, mtrag_dir, collection, out_dir: Path) -> "PassageStore":
        out_dir.mkdir(parents=True, exist_ok=True)
        passages = {}
        with open(out_dir / "passages.jsonl", "w", encoding="utf-8") as f:
            for r in iter_corpus(mtrag_dir, collection):
                passages[r["id"]] = r
                f.write(json.dumps(r) + "\n")
        return cls(passages)

    @classmethod
    def load(cls, d: Path) -> "PassageStore":
        with open(d / "passages.jsonl", encoding="utf-8") as f:
            return cls({r["id"]: r for r in map(json.loads, f)})

    def __getitem__(self, pid):
        return self.p[pid]

    def ids(self):
        return list(self.p)


class BM25Index:
    def __init__(self, retriever, ids):
        self.r, self.ids = retriever, ids

    @classmethod
    def build(cls, store: PassageStore, out_dir: Path) -> "BM25Index":
        import bm25s

        ids = store.ids()
        texts = [(store[i].get("title", "") + " " + store[i]["text"]) for i in ids]
        r = bm25s.BM25()
        r.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
        r.save(str(out_dir / "bm25"))
        json.dump(ids, open(out_dir / "bm25_ids.json", "w"))
        return cls(r, ids)

    @classmethod
    def load(cls, d: Path) -> "BM25Index":
        import bm25s

        return cls(bm25s.BM25.load(str(d / "bm25")), json.load(open(d / "bm25_ids.json")))

    def search(self, query: str, k: int = 100) -> list[str]:
        import bm25s

        k = min(k, len(self.ids))
        idx, _ = self.r.retrieve(bm25s.tokenize([query], stopwords="en", show_progress=False), k=k,
                                 show_progress=False)
        return [self.ids[i] for i in idx[0]]


class DenseIndex:
    def __init__(self, model_name: str, index, ids):
        from sentence_transformers import SentenceTransformer

        self.model, self.index, self.ids = SentenceTransformer(model_name), index, ids

    @classmethod
    def build(cls, store: PassageStore, model_name: str, out_dir: Path, batch_size: int = 256) -> "DenseIndex":
        import faiss
        from sentence_transformers import SentenceTransformer

        ids = store.ids()
        model = SentenceTransformer(model_name)
        emb = model.encode([store[i]["text"] for i in ids], batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(out_dir / "dense.faiss"))
        json.dump(ids, open(out_dir / "dense_ids.json", "w"))
        return cls(model_name, index, ids)

    @classmethod
    def load(cls, d: Path, model_name: str) -> "DenseIndex":
        import faiss

        return cls(model_name, faiss.read_index(str(d / "dense.faiss")), json.load(open(d / "dense_ids.json")))

    def search(self, query: str, k: int = 100) -> list[str]:
        q = self.model.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True).astype(np.float32)
        _, idx = self.index.search(q, k)
        return [self.ids[i] for i in idx[0] if i >= 0]


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.m = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, passages: list[dict]) -> list[dict]:
        if not passages:
            return []
        scores = self.m.predict([(query, p["text"]) for p in passages], show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        return [dict(passages[i], score=float(scores[i])) for i in order]


class HybridRetriever:
    """search(query) -> top-k passages. Any component can be switched off for ablations."""

    def __init__(self, collection: str, store: PassageStore, bm25: BM25Index | None = None,
                 dense: DenseIndex | None = None, reranker: Reranker | None = None, cfg: dict | None = None):
        self.collection, self.store, self.bm25, self.dense, self.reranker = collection, store, bm25, dense, reranker
        c = cfg or {}
        self.bm25_k, self.dense_k = c.get("bm25_top_k", 100), c.get("dense_top_k", 100)
        self.rrf_k, self.rerank_n, self.final_k = c.get("rrf_k", 60), c.get("rerank_top_n", 50), c.get("final_k", 5)

    def candidates(self, query: str) -> list[str]:
        rankings = []
        if self.bm25:
            rankings.append(self.bm25.search(query, self.bm25_k))
        if self.dense:
            rankings.append(self.dense.search(query, self.dense_k))
        if not rankings:
            raise ValueError("enable at least one of bm25 / dense")
        return rrf_fuse(rankings, self.rrf_k) if len(rankings) > 1 else rankings[0]

    def search(self, query: str, k: int | None = None, exclude_parents: set[str] | None = None) -> list[dict]:
        k = k or self.final_k
        ids = self.candidates(query)
        if exclude_parents:
            ids = [i for i in ids if parent_doc(i, self.collection) not in exclude_parents]
        passages = [dict(self.store[i], document_id=i) for i in ids[: self.rerank_n]]
        if self.reranker:
            passages = self.reranker.rerank(query, passages)
        return passages[:k]


def load_retriever(collection: str, cfg: dict, use_bm25=True, use_dense=True, use_rerank=True) -> HybridRetriever:
    d = Path(cfg["data"]["index_dir"]) / collection
    r = cfg["retrieval"]
    return HybridRetriever(
        collection, PassageStore.load(d),
        bm25=BM25Index.load(d) if use_bm25 else None,
        dense=DenseIndex.load(d, r["dense_model"]) if use_dense else None,
        reranker=Reranker(r["reranker_model"]) if use_rerank else None,
        cfg=r,
    )
