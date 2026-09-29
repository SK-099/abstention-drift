"""Controlled drift test on the synthetic dev set (RQ1 with question type held fixed).

Real unanswerable questions are not spread evenly over turns (see results/dataset_stats.md), so a drop
in recall could come from *which* questions appear late rather than from the turn itself. Evidence removal
creates an unanswerable twin of every answerable dev turn, at every turn position, so the passages are the
only thing that changes. We feed the fixed passages to:
    generator  (prompt-only abstention, systems S0-S2)
    checker    (S3 / S4)

The generator sees the real conversation history of each turn (as in the pipeline), so a turn effect
here is an effect of the conversation, not of the retrieved passages.

    python scripts/run_synthetic_probe.py --who generator --split all      # generator is not trained: use train + dev
    python scripts/run_synthetic_probe.py --who checker                     # checker: dev only (it was trained on train)
Output: outputs/probe_<who>[_<split>].jsonl  -> evaluate with scripts/evaluate_abstention.py
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from abstain.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--who", choices=["generator", "checker"], required=True)
ap.add_argument("--ckpt", help="checker dir (default: <checker.output_dir>/best)")
ap.add_argument("--limit", type=int)
ap.add_argument("--split", choices=["dev", "train", "all"], default="dev",
                help="checker examples to probe with (generator only; the checker must use dev)")
ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. --set llm.model=Qwen/Qwen2.5-1.5B-Instruct")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
out_dir = Path(cfg["data"]["output_dir"])
if a.who == "checker" and a.split != "dev":
    ap.error("the checker was trained on the train split: probe it on --split dev")
splits = ["train", "dev"] if a.split == "all" else [a.split]
exs = [json.loads(l) for s in splits for l in open(out_dir / f"checker_{s}.jsonl", encoding="utf-8")][: a.limit]

if a.who == "checker":
    from abstain.checker import AnswerabilityChecker
    ch = AnswerabilityChecker(a.ckpt or cfg["checker"]["output_dir"] + "/best", device=cfg["checker"].get("infer_device"),
                                     max_length=cfg["checker"].get("max_length", 512))
    probs = ch.predict_proba(exs)
    preds = [int(p >= ch.threshold) for p in probs]
else:
    from abstain.data import load_benchmark
    from abstain.generate import generate_answer
    from abstain.idk import is_idk
    from abstain.llm import make_llm
    llm = make_llm(cfg["llm"])
    from abstain.pipeline import prompt_limits
    gen_kw = prompt_limits(cfg["llm"])[1]
    tasks = {t.task_id: t for ts in load_benchmark(cfg["data"]["mtrag_dir"]).values() for t in ts}
    preds = []
    for e in tqdm(exs):
        t = tasks[e["task_id"]]          # real history + question, as the pipeline's generator sees them
        preds.append(int(is_idk(generate_answer(llm, t, e["passages"], **gen_kw))))

path = out_dir / (f"probe_{a.who}.jsonl" if a.split == "dev" else f"probe_{a.who}_{a.split}.jsonl")
with open(path, "w", encoding="utf-8") as f:
    for e, p in zip(exs, preds):
        f.write(json.dumps({k: e[k] for k in ("task_id", "conversation_id", "turn", "collection", "multi_turn",
                                              "label", "kind")} | {"pred": p}) + "\n")
print("wrote", path)
