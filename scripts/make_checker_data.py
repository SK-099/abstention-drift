"""Build answerability-checker examples (real + synthetic evidence-removed) from MTRAG.

    python scripts/make_checker_data.py --rewrite oracle     # fast: no LLM needed (human rewrites)
    python scripts/make_checker_data.py --rewrite llm        # uses the LLM rewriter (GPU)

Splits by conversation: train/dev from MTRAG. MTRAG-UN is kept as an untouched test set.
Writes outputs/checker_{train,dev}.jsonl
"""
import argparse
import json
from functools import partial
from pathlib import Path

from abstain.config import load_config
from abstain.data import conversation_split, load_benchmark, load_rewrites
from abstain.retrieval import load_retriever
from abstain.rewrite import QueryRewriter
from abstain.synth import checker_examples, llm_still_answerable

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--rewrite", default="oracle", choices=["last_turn", "full_history", "oracle", "llm"])
ap.add_argument("--no-dense", action="store_true")
ap.add_argument("--no-rerank", action="store_true")
ap.add_argument("--verify-llm", action="store_true", help="drop synthetic examples the LLM says are still answerable")
ap.add_argument("--collections", nargs="+", help="restrict to some collections (quick runs)")
ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. --set llm.model=Qwen/Qwen2.5-1.5B-Instruct")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
tasks = load_benchmark(cfg["data"]["mtrag_dir"])["mtrag"]
if a.collections:
    tasks = [t for t in tasks if t.collection in a.collections]
train, dev = conversation_split(tasks, 0.2, cfg["synthetic"]["seed"])

llm = None
if a.rewrite == "llm" or a.verify_llm:
    from abstain.llm import make_llm
    llm = make_llm(cfg["llm"])
from abstain.pipeline import prompt_limits

rewriter = QueryRewriter(a.rewrite, llm=llm, oracle=load_rewrites(cfg["data"]["mtrag_dir"]), **prompt_limits(cfg["llm"])[0])
rets = {c: load_retriever(c, cfg, use_dense=not a.no_dense, use_rerank=not a.no_rerank)
        for c in sorted({t.collection for t in tasks})}


def retrieve(t, q, exclude):
    return rets[t.collection].search(q, k=cfg["synthetic"]["k"] + 10, exclude_parents=exclude)


out = Path(cfg["data"]["output_dir"]); out.mkdir(exist_ok=True)
for name, split in (("train", train), ("dev", dev)):
    exs = checker_examples(split, retrieve, rewriter, k=cfg["synthetic"]["k"], seed=cfg["synthetic"]["seed"],
                           verify=partial(llm_still_answerable, llm) if a.verify_llm else None)
    with open(out / f"checker_{name}.jsonl", "w", encoding="utf-8") as f:
        for e in exs:
            f.write(json.dumps(e) + "\n")
    n_syn = sum(e["kind"] == "synthetic" for e in exs); n_pos = sum(e["label"] for e in exs)
    print(f"{name}: {len(exs)} examples | unanswerable {n_pos} (synthetic {n_syn}) | answerable {len(exs) - n_pos}")
