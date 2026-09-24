"""Controlled drift test on the synthetic dev set (RQ1 with question type held fixed).

Real unanswerable questions are not spread evenly over turns (see results/dataset_stats.md), so a drop
in recall could come from *which* questions appear late rather than from the turn itself. Evidence removal
creates an unanswerable twin of every answerable dev turn, at every turn position, so the passages are the
only thing that changes. We feed the fixed passages to:
    generator  (prompt-only abstention, systems S0-S2)
    checker    (S3 / S4)

    python scripts/run_synthetic_probe.py --who generator --limit 200
    python scripts/run_synthetic_probe.py --who checker --ckpt checkpoints/checker/best
Output: outputs/probe_<who>.jsonl  -> evaluate with scripts/evaluate_abstention.py
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
ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. --set llm.model=Qwen/Qwen2.5-1.5B-Instruct")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
out_dir = Path(cfg["data"]["output_dir"])
exs = [json.loads(l) for l in open(out_dir / "checker_dev.jsonl", encoding="utf-8")][: a.limit]

if a.who == "checker":
    from abstain.checker import AnswerabilityChecker
    ch = AnswerabilityChecker(a.ckpt or cfg["checker"]["output_dir"] + "/best", device=cfg["checker"].get("infer_device"))
    probs = ch.predict_proba(exs)
    preds = [int(p >= ch.threshold) for p in probs]
else:
    from abstain.data import Task
    from abstain.generate import generate_answer
    from abstain.idk import is_idk
    from abstain.llm import ChatLLM
    llm = ChatLLM(cfg["llm"]["model"], cfg["llm"]["load_in_4bit"], cfg["llm"]["max_new_tokens"])
    preds = []
    for e in tqdm(exs):
        # history is not stored in the example; use the rewritten standalone query as the question
        t = Task(e["task_id"], e["conversation_id"], e["turn"], e["collection"], [], e["query"], "")
        preds.append(int(is_idk(generate_answer(llm, t, e["passages"]))))

path = out_dir / f"probe_{a.who}.jsonl"
with open(path, "w", encoding="utf-8") as f:
    for e, p in zip(exs, preds):
        f.write(json.dumps({k: e[k] for k in ("task_id", "conversation_id", "turn", "collection", "multi_turn",
                                              "label", "kind")} | {"pred": p}) + "\n")
print("wrote", path)
