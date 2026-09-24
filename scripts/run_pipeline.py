"""Run one system over MTRAG (dev conversations) or MTRAG-UN and write MTRAG-format predictions.

    python scripts/run_pipeline.py --system S4 --split mtragun
    python scripts/run_pipeline.py --system S0 --split dev --limit 50      # quick check

Output: outputs/<system>_<split>.jsonl  (compatible with the official MTRAG eval scripts,
plus an 'abstain' field used by scripts/evaluate_abstention.py)
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from abstain.config import load_config
from abstain.data import conversation_split, load_benchmark, load_rewrites
from abstain.pipeline import AbstainRAG, to_mtrag_prediction
from abstain.retrieval import load_retriever
from abstain.rewrite import QueryRewriter

SYSTEMS = {  # rewrite, dense, rerank, checker (None | "real" = trained on real labels | "full" = real + synthetic)
    "S0": ("last_turn", False, False, None),
    "S1": ("llm", False, False, None),
    "S2": ("llm", True, True, None),
    "S3": ("llm", True, True, "real"),
    "S4": ("llm", True, True, "full"),
    "S4-oracle": ("oracle", True, True, "full"),   # upper bound: human rewrites
}

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--system", default="S4", choices=list(SYSTEMS))
ap.add_argument("--split", default="dev", choices=["dev", "mtragun"])
ap.add_argument("--limit", type=int)
ap.add_argument("--collections", nargs="+", help="restrict to some collections (quick runs)")
ap.add_argument("--no-generate", action="store_true", help="only rewrite/retrieve/check (fast)")
ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. --set llm.model=Qwen/Qwen2.5-1.5B-Instruct")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
mode, dense, rerank, ckpt = SYSTEMS[a.system]
d = Path(cfg["data"]["mtrag_dir"])

bench = load_benchmark(d)
if a.split == "dev":
    tasks = conversation_split(bench["mtrag"], 0.2, cfg["synthetic"]["seed"])[1]
    raw_path = d / "mtrag-human/generation_tasks/reference.jsonl"
else:
    tasks = bench["mtragun"]
    raw_path = d / "mtragun-human/generation_tasks/reference.jsonl"
raw = {r["task_id"]: r for r in map(json.loads, open(raw_path, encoding="utf-8"))}
tasks = [t for t in tasks if t.abstain_label is not None and (not a.collections or t.collection in a.collections)][: a.limit]

llm = None
if mode == "llm" or not a.no_generate:
    from abstain.llm import ChatLLM
    llm = ChatLLM(cfg["llm"]["model"], cfg["llm"]["load_in_4bit"], cfg["llm"]["max_new_tokens"])
checker = None
if ckpt:
    from abstain.checker import AnswerabilityChecker
    base = cfg["checker"]["output_dir"] + ("-real" if ckpt == "real" else "")
    checker = AnswerabilityChecker(f"{base}/best", device=cfg["checker"].get("infer_device"))
rets = {c: load_retriever(c, cfg, use_dense=dense, use_rerank=rerank) for c in sorted({t.collection for t in tasks})}
system = AbstainRAG(QueryRewriter(mode, llm=llm, oracle=load_rewrites(d)), rets, llm=llm, checker=checker,
                    k=cfg["retrieval"]["final_k"])

out = Path(cfg["data"]["output_dir"]); out.mkdir(exist_ok=True)
path = out / f"{a.system}_{a.split}.jsonl"
with open(path, "w", encoding="utf-8") as f:
    for t in tqdm(tasks, desc=a.system):
        f.write(json.dumps(to_mtrag_prediction(raw[t.task_id], system(t, generate=not a.no_generate))) + "\n")
print("wrote", path)
