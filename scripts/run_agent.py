"""Run S5, the LangGraph self-correcting abstention agent, and write MTRAG-format predictions.

Fast mode (default): reuse the finished S2 run. The agent's first attempt (rewrite, retrieval,
generation) is identical to S2 and the gate score is the one S4 (or S2-rr) already computed, so those
are loaded from outputs/S2_<split>.jsonl and outputs/S4_<split>.jsonl. Only the new agent steps run:
the retry on weak/refused turns (~120 turns) and the self-check on answered turns (~310 turns).

    python scripts/run_agent.py --config configs/laptop.yaml                     # S5 on MTRAG-UN, ~2 h
    python scripts/run_agent.py --config configs/laptop.yaml --underspecified    # + the 78 underspecified turns (live)
    python scripts/run_agent.py --config configs/laptop.yaml --live --limit 10   # everything live (quick check)
    python scripts/run_agent.py ... --no-self-check / --max-retries 2 / --gate reranker   (ablations)

The run can be interrupted and resumed: finished task ids are skipped (use --restart to start over).
Output: outputs/S5-agent_<split>.jsonl (+ S5-agent_underspecified.jsonl); evaluate with
scripts/evaluate_abstention.py like any other system.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from abstain.agent import AbstentionAgent, make_agent_llm, make_retriever, to_prediction
from abstain.config import load_config
from abstain.data import conversation_split, load_benchmark
from abstain.pipeline import prompt_limits
from abstain.retrieval import load_retriever

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--split", default="mtragun", choices=["dev", "mtragun"])
ap.add_argument("--gate", default="checker", choices=["checker", "reranker", "none"],
                help="evidence grader: the S4 checker, the S2-rr re-ranker score, or none")
ap.add_argument("--max-retries", type=int, default=1)
ap.add_argument("--no-self-check", action="store_true")
ap.add_argument("--no-clarify", action="store_true")
ap.add_argument("--self-check-min-p", default="0",
                help="only self-check answers whose gate score p >= this value; 'auto' reads the value chosen "
                     "on dev by scripts/calibrate_self_check.py (checkpoints/self_check.json)")
ap.add_argument("--live", action="store_true", help="do not reuse the S2/S4 runs (much slower)")
ap.add_argument("--underspecified", action="store_true", help="also run the underspecified turns (live)")
ap.add_argument("--name", default="S5-agent", help="output name (use a different one for ablations)")
ap.add_argument("--limit", type=int)
ap.add_argument("--collections", nargs="+")
ap.add_argument("--restart", action="store_true", help="ignore an existing output file")
ap.add_argument("--set", nargs="*", default=[], help="config overrides")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
out_dir = Path(cfg["data"]["output_dir"])
d = Path(cfg["data"]["mtrag_dir"])

# ---------------------------------------------------------------- tasks
bench = load_benchmark(d)
if a.split == "dev":
    tasks = conversation_split(bench["mtrag"], 0.2, cfg["synthetic"]["seed"])[1]
    raw_path = d / "mtrag-human/generation_tasks/reference.jsonl"
else:
    tasks = bench["mtragun"]
    raw_path = d / "mtragun-human/generation_tasks/reference.jsonl"
raw = {r["task_id"]: r for r in map(json.loads, open(raw_path, encoding="utf-8"))}
keep = lambda t: not a.collections or t.collection in a.collections
main = [t for t in tasks if t.abstain_label is not None and keep(t)][: a.limit]
under = [t for t in tasks if t.answerability == "UNDERSPECIFIED" and keep(t)][: a.limit] if a.underspecified else []

# ---------------------------------------------------------------- components
llm = make_agent_llm(cfg["llm"])
c = cfg["checker"]
gate = None
if a.gate == "checker":
    from abstain.checker import AnswerabilityChecker
    gate = AnswerabilityChecker(c["output_dir"] + "/best", device=c.get("infer_device"), max_length=c.get("max_length", 512))
elif a.gate == "reranker":
    from abstain.checker import RerankerScoreChecker
    rr = RerankerScoreChecker(str(Path(c["output_dir"]).parent / "reranker_checker.json"))
    gate = lambda q, ps: rr.p_unanswerable([p.get("score", 0.0) for p in ps])
    gate.threshold = rr.threshold
if a.self_check_min_p == "auto":
    f = Path(c["output_dir"]).parent / "self_check.json"
    if not f.exists():
        raise SystemExit(f"{f} not found: run scripts/calibrate_self_check.py first")
    min_p = json.load(open(f, encoding="utf-8"))["min_p"]
else:
    min_p = float(a.self_check_min_p)
rets = {col: load_retriever(col, cfg) for col in sorted({t.collection for t in main + under})}
rw_kw, gen_kw = prompt_limits(cfg["llm"])
agent = AbstentionAgent(llm, make_retriever(rets, cfg["retrieval"]["final_k"]), gate, max_retries=a.max_retries,
                        rewrite_kwargs=rw_kw, gen_kwargs=gen_kw, self_check=not a.no_self_check,
                        clarify=not a.no_clarify, self_check_min_p=min_p)
print(f"agent: gate={a.gate} (tau={agent.threshold:.3f}), max_retries={a.max_retries}, "
      f"self_check={not a.no_self_check} (p >= {min_p:.3f}), clarify={not a.no_clarify}, mode={'live' if a.live else 'reuse S2'}")

# ---------------------------------------------------------------- caches from S2 / S4 (fast mode)
cache = {}
if not a.live:
    s2 = out_dir / f"S2_{a.split}.jsonl"
    gfile = {"checker": f"S4_{a.split}.jsonl", "reranker": f"S2-rr_{a.split}.jsonl"}.get(a.gate)
    if not s2.exists() or (gfile and not (out_dir / gfile).exists()):
        raise SystemExit(f"Fast mode needs {s2}" + (f" and {out_dir / gfile}" if gfile else "")
                         + " (run run_pipeline.py --system S2 and apply_checker.py first), or pass --live.")
    pmap = {}
    if gfile:
        for r in map(json.loads, open(out_dir / gfile, encoding="utf-8")):
            pmap[r["task_id"]] = r["abstain"]["p_unanswerable"]
    for r in map(json.loads, open(s2, encoding="utf-8")):
        cache[r["task_id"]] = {"query": r["abstain"]["query"], "answer": r["predictions"][0]["text"],
                               "passages": [{"document_id": p["document_id"], "title": p.get("title", ""),
                                             "text": p["text"], "score": p.get("score", 0.0)} for p in r["contexts"]],
                               "p": pmap.get(r["task_id"]) if gfile else None}


def run(tasks, path, use_cache):
    done = set()
    if path.exists() and not a.restart:
        done = {json.loads(l)["task_id"] for l in open(path, encoding="utf-8")}
    todo = [t for t in tasks if t.task_id not in done]
    if done:
        print(f"{path.name}: resuming, {len(done)} done, {len(todo)} to go")
    with open(path, "a" if done else "w", encoding="utf-8") as f:
        for t in tqdm(todo, desc=path.stem):
            out = agent(t, cache.get(t.task_id) if use_cache else None)
            f.write(json.dumps(to_prediction(raw[t.task_id], out)) + "\n")
            f.flush()
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    lab = {t.task_id: t.abstain_label for t in tasks}
    dec = Counter((r["agent"]["decision"], {1: "unans", 0: "ans", None: "under"}[lab.get(r["task_id"])]) for r in rows)
    print(f"wrote {path}  decisions (decision, label): {dict(sorted(dec.items()))}")
    print("retried turns:", sum(r["agent"]["attempts"] > 0 for r in rows),
          "| abstained by:", dict(Counter(r["abstain"]["abstained_by"] for r in rows if r["abstain"]["abstained"])))


run(main, out_dir / f"{a.name}_{a.split}.jsonl", not a.live)
if under:
    run(under, out_dir / f"{a.name}_underspecified.jsonl", False)
