"""Build passage store + BM25 (+ dense) index per collection.

    python scripts/build_indexes.py                       # all 4 collections, BM25 + dense (GPU recommended)
    python scripts/build_indexes.py --collections fiqa --no-dense   # quick CPU-only start

Dense encoding of all ~366K passages with bge-small takes ~15-25 min on a T4.
"""
import argparse
from pathlib import Path

from abstain.config import load_config
from abstain.data import COLLECTIONS
from abstain.retrieval import BM25Index, DenseIndex, PassageStore

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--collections", nargs="+", default=list(COLLECTIONS))
ap.add_argument("--no-dense", action="store_true")
ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. --set llm.model=Qwen/Qwen2.5-1.5B-Instruct")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
for c in a.collections:
    d = Path(cfg["data"]["index_dir"]) / c
    print(f"[{c}] passages ...", flush=True)
    store = PassageStore.build(cfg["data"]["mtrag_dir"], c, d)
    print(f"[{c}] {len(store.p):,} passages; BM25 ...", flush=True)
    BM25Index.build(store, d)
    if not a.no_dense:
        print(f"[{c}] dense ({cfg['retrieval']['dense_model']}) ...", flush=True)
        DenseIndex.build(store, cfg["retrieval"]["dense_model"], d)
    print(f"[{c}] done -> {d}")
