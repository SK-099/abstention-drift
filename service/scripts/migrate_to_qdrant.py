"""Load the MTRAG corpora (ClapNQ, Cloud, FiQA, Govt; ~366K passages) into Qdrant for the service.

Reuses the research indexes: passage texts from indexes/<collection>/passages.jsonl and the BGE-small
vectors already stored in indexes/<collection>/dense.faiss (no re-encoding); the sparse BM25 vector is
computed here. Without a FAISS index the passages are encoded with the configured embedder.

    python service/scripts/migrate_to_qdrant.py --index-dir indexes                 # all four collections
    python service/scripts/migrate_to_qdrant.py --collections cloud --limit 2000    # quick test
    python service/scripts/migrate_to_qdrant.py --snapshot                          # + create a Qdrant snapshot

QDRANT_URL / QDRANT_API_KEY select the server (local Docker or Qdrant Cloud). Re-running is safe:
point ids are deterministic, so existing points are overwritten.
Restore a snapshot elsewhere (e.g. Qdrant Cloud) with:
    curl -X POST "$QDRANT_URL/collections/passages/snapshots/upload?priority=snapshot" \
         -H "api-key: $QDRANT_API_KEY" -F "snapshot=@passages-....snapshot"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE.parents[1] / "src")]

from abstain_api.retrieval import Embedder, get_store
from abstain_api.settings import get_settings

ap = argparse.ArgumentParser()
ap.add_argument("--index-dir", default="indexes")
ap.add_argument("--collections", nargs="+", default=["clapnq", "cloud", "fiqa", "govt"])
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--limit", type=int, help="first N passages per collection (testing)")
ap.add_argument("--snapshot", action="store_true", help="create a snapshot of the Qdrant collection afterwards")
a = ap.parse_args()

s = get_settings()
store = get_store(s)  # Qdrant, or pgvector with VECTOR_STORE=pgvector
emb = None
for col in a.collections:
    d = Path(a.index_dir) / col
    passages = [json.loads(l) for l in open(d / "passages.jsonl", encoding="utf-8")][: a.limit]
    vec_of = None
    if (d / "dense.faiss").exists():
        import faiss

        index = faiss.read_index(str(d / "dense.faiss"))
        ids = json.load(open(d / "dense_ids.json", encoding="utf-8"))
        if index.d != s.dense_dim:
            sys.exit(f"{col}: FAISS vectors have dim {index.d} but DENSE_DIM={s.dense_dim}")
        pos = {pid: i for i, pid in enumerate(ids)}
        vec_of = lambda batch, index=index, pos=pos: [index.reconstruct(pos[p["id"]]).tolist() for p in batch]
        print(f"{col}: {len(passages)} passages, reusing FAISS vectors")
    else:
        emb = emb or Embedder(s)
        vec_of = lambda batch, emb=emb: emb.embed_documents([p["text"] for p in batch])
        print(f"{col}: {len(passages)} passages, encoding with {s.dense_model}")
    t0, n = time.time(), 0
    for i in range(0, len(passages), a.batch):
        batch = [{"id": p["id"], "title": p.get("title", ""), "text": p["text"]} for p in passages[i : i + a.batch]]
        n += store.upsert(col, batch, vec_of(batch))
        if (i // a.batch) % 20 == 0:
            rate = n / max(time.time() - t0, 1e-6)
            print(f"  {n}/{len(passages)} ({rate:.0f}/s)", flush=True)
    print(f"{col}: {store.count(col)} points in Qdrant ({time.time() - t0:.0f}s)")

if a.snapshot and s.vector_store == "qdrant":
    snap = store.client.create_snapshot(store.name, wait=True)
    print(f"snapshot: {snap.name} (download: GET {s.qdrant_url}/collections/{store.name}/snapshots/{snap.name})")
_ = np  # numpy is required by faiss
