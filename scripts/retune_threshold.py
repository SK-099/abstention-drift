"""Re-choose a trained checker's threshold on a larger dev set, and report how well it separates classes.

The default training run tunes the threshold on *real* dev examples only. MTRAG has very few real
unanswerable dev turns (6 with the default split), so that threshold is noisy. This script lets you
tune it on all dev examples (real + synthetic unanswerables; the false-abstention budget is always
measured on real answerable turns) and prints AUROC per subset, with no retraining.

    python scripts/retune_threshold.py --config configs/laptop.yaml --on all            # S4 checker
    python scripts/retune_threshold.py --config configs/laptop.yaml --on all --real-only  # S3 checker

The previous threshold.json is kept as threshold_prev.json.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from abstain.checker import AnswerabilityChecker, choose_threshold
from abstain.config import load_config


def auroc(p, y):
    """Probability that a random unanswerable example scores above a random answerable one."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--real-only", action="store_true", help="retune the S3 (real-only) checker")
ap.add_argument("--on", choices=["all", "real"], default="all", help="dev examples used to choose the threshold")
ap.add_argument("--set", nargs="*", default=[], help="config overrides")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
c = cfg["checker"]
ckpt = Path(c["output_dir"] + ("-real" if a.real_only else "")) / "best"
dev = [json.loads(l) for l in open(Path(cfg["data"]["output_dir"]) / "checker_dev.jsonl", encoding="utf-8")]

checker = AnswerabilityChecker(str(ckpt), device=c.get("infer_device"), max_length=c.get("max_length", 512))
probs = checker.predict_proba(dev)
y = np.array([e["label"] for e in dev])
real = np.array([e["kind"] == "real" for e in dev])

print(f"checker: {ckpt}")
print(f"dev: {len(dev)} examples | real unanswerable {int((y[real] == 1).sum())} | "
      f"synthetic unanswerable {int((y[~real] == 1).sum())} | answerable {int((y == 0).sum())}")
ans = y == 0
print(f"AUROC  all: {auroc(probs, y):.3f} | real only: {auroc(probs[real], y[real]):.3f} | "
      f"synthetic unanswerable vs answerable: {auroc(probs[~real | ans], y[~real | ans]):.3f}")

sel = np.ones(len(dev), bool) if a.on == "all" else real
th = choose_threshold(probs[sel], y[sel], c["max_false_abstention"])
th["tuned_on"] = a.on
pred = probs >= th["threshold"]
th["recall_real"] = float(pred[real & (y == 1)].mean()) if (real & (y == 1)).any() else None
th["recall_synthetic"] = float(pred[~real & (y == 1)].mean()) if (~real & (y == 1)).any() else None
print("new threshold:", th)

f = ckpt / "threshold.json"
if f.exists():
    shutil.copy(f, ckpt / "threshold_prev.json")
json.dump(th, open(f, "w", encoding="utf-8"), indent=2)
print(f"wrote {f}")
