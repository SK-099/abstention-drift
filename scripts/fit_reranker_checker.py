"""Training-free abstention baseline (system S2-rr): abstain when even the best retrieved passage gets a
low cross-encoder re-ranker score. Scores the checker dev set, reports AUROC and picks the threshold
under the same false-abstention budget as the trained checkers.

    python scripts/fit_reranker_checker.py --config configs/laptop.yaml
Writes checkpoints/reranker_checker.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from abstain.checker import RerankerScoreChecker, choose_threshold
from abstain.config import load_config
from abstain.retrieval import Reranker

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--set", nargs="*", default=[], help="config overrides")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
r = cfg["retrieval"]
dev = [json.loads(l) for l in open(Path(cfg["data"]["output_dir"]) / "checker_dev.jsonl", encoding="utf-8")]
rr = Reranker(r["reranker_model"], r.get("reranker_device"), r.get("reranker_max_length", 512))

p = np.array([RerankerScoreChecker.p_unanswerable([x["score"] for x in rr.rerank(e["query"], e["passages"])])
              for e in tqdm(dev, desc="scoring dev")])
y = np.array([e["label"] for e in dev])
real = np.array([e["kind"] == "real" for e in dev])


def auroc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


ans = y == 0
print(f"AUROC  all: {auroc(p, y):.3f} | real only: {auroc(p[real], y[real]):.3f} | "
      f"synthetic unanswerable vs answerable: {auroc(p[~real | ans], y[~real | ans]):.3f}")
th = choose_threshold(p, y, cfg["checker"]["max_false_abstention"])
print("threshold:", th)
out = Path(cfg["checker"]["output_dir"]).parent / "reranker_checker.json"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(th, open(out, "w", encoding="utf-8"), indent=2)
print("wrote", out)
