"""Choose the selective self-check cut-off on the MTRAG dev split (never on the test set).

The agent self-checks an answer only when the gate score p >= min_p. This script uses the same dev set
that set the gate threshold: the 224 checker-dev examples (150 answerable, 6 real + 68 synthetic
unanswerable), each with its fixed passages and the real conversation history. For every example it
records the gate score p, the generator's answer and the self-check verdict (cached, resumable), then
evaluates every cut-off offline:

    abstain = (p >= tau) or generator refuses or (p >= min_p and self-check says "not supported")

and picks the min_p with the best abstention F1 among those that add at most `--max-extra-fa`
(default 5 points) of false abstention over no self-check. Writes checkpoints/self_check.json,
which run_agent.py reads with --self-check-min-p auto.

    python scripts/calibrate_self_check.py --config configs/laptop.yaml        # ~40 min on a GTX 1650
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from abstain.agent import make_agent_llm, self_check_supported
from abstain.checker import AnswerabilityChecker
from abstain.config import load_config
from abstain.data import load_benchmark
from abstain.generate import generate_answer
from abstain.idk import is_idk
from abstain.pipeline import prompt_limits

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--max-extra-fa", type=float, default=0.05)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, a.set)
out = Path(cfg["data"]["output_dir"])
c = cfg["checker"]
dev = [json.loads(l) for l in open(out / "checker_dev.jsonl", encoding="utf-8")]
tasks = {t.task_id: t for ts in load_benchmark(cfg["data"]["mtrag_dir"]).values() for t in ts}
gate = AnswerabilityChecker(c["output_dir"] + "/best", device=c.get("infer_device"), max_length=c.get("max_length", 512))
llm = make_agent_llm(cfg["llm"])
gen_kw = prompt_limits(cfg["llm"])[1]

cache_f = out / "selfcheck_calib_dev.jsonl"
cache = {}
if cache_f.exists():
    for l in open(cache_f, encoding="utf-8"):
        r = json.loads(l)
        cache[r["key"]] = r
with open(cache_f, "a", encoding="utf-8") as f:
    for e in tqdm(dev, desc="dev examples"):
        key = f"{e['task_id']}|{e['kind']}|{e['label']}"
        if key in cache:
            continue
        p = float(gate(e["query"], e["passages"]))
        ans = generate_answer(llm, tasks[e["task_id"]], e["passages"], **gen_kw)
        refused = is_idk(ans)
        sc = None if refused else bool(self_check_supported(llm, e["query"], e["passages"], ans, gen_kw.get("passage_chars")))
        cache[key] = {"key": key, "label": e["label"], "kind": e["kind"], "p": p, "refused": refused, "supported": sc}
        f.write(json.dumps(cache[key]) + "\n")
        f.flush()

rows = list(cache.values())
y = np.array([r["label"] for r in rows])
p = np.array([r["p"] for r in rows])
ref = np.array([r["refused"] for r in rows])
unsup = np.array([r["supported"] is False for r in rows])
tau = gate.threshold


def score(min_p):
    pred = (p >= tau) | ref | ((p >= min_p) & unsup)
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    rec, fa = tp / max((y == 1).sum(), 1), fp / max((y == 0).sum(), 1)
    prec = tp / max(pred.sum(), 1)
    return {"min_p": float(min_p), "recall": rec, "fa": fa, "precision": prec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


print(f"dev: {len(rows)} examples ({int((y == 1).sum())} unanswerable), gate tau = {tau:.3f}")
print(f"self-check says 'not supported' for {unsup[y == 0].mean():.2f} of answered answerable and "
      f"{unsup[y == 1].mean():.2f} of answered unanswerable examples")
grid = [1.01] + sorted(set(np.round(np.arange(0.0, tau + 1e-9, 0.05), 2)), reverse=True)
res = [score(m) for m in grid]
base = res[0]
print("\n min_p  recall    FA   prec    F1")
for r in res:
    tag = "  (no self-check)" if r["min_p"] > 1 else ""
    print(f" {r['min_p']:5.2f}  {r['recall']:.3f}  {r['fa']:.3f}  {r['precision']:.3f}  {r['f1']:.3f}{tag}")
ok = [r for r in res if r["fa"] <= base["fa"] + a.max_extra_fa]
best = max(ok, key=lambda r: (r["f1"], r["min_p"]))
print(f"\nchosen min_p = {best['min_p']:.2f} (best dev F1 with FA <= {base['fa'] + a.max_extra_fa:.3f})")
dst = Path(c["output_dir"]).parent / "self_check.json"
json.dump(best | {"tau": tau, "n_dev": len(rows)}, open(dst, "w", encoding="utf-8"), indent=2)
print("wrote", dst)
