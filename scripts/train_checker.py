"""Train the answerability checker.

    python scripts/train_checker.py                   # S4: real + synthetic unanswerables
    python scripts/train_checker.py --real-only       # S3 ablation
    python scripts/train_checker.py --set checker.mode=joint   # older single-pass checker

Saves checkpoints/checker[-real]/best (+ threshold.json). The threshold is chosen on all dev examples
(real + synthetic unanswerables; answerable turns are always real) under the false-abstention budget.
MTRAG has only ~6 real unanswerable dev turns, too few to tune on alone (--tune-on real does that).
"""
import argparse
import json

from abstain.checker import train_checker
from abstain.config import load_config
from abstain.data import load_benchmark

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--real-only", action="store_true")
ap.add_argument("--tune-on", choices=["all", "real"], default="all", help="dev examples used to choose the threshold")
ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. --set checker.epochs=2")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
out = cfg["data"]["output_dir"]
train = [json.loads(l) for l in open(f"{out}/checker_train.jsonl", encoding="utf-8")]
dev = [json.loads(l) for l in open(f"{out}/checker_dev.jsonl", encoding="utf-8")]
if a.real_only:
    train = [e for e in train if e["kind"] == "real"]
    cfg["checker"]["output_dir"] += "-real"
gold = {t.task_id: set(t.gold_ids) for ts in load_benchmark(cfg["data"]["mtrag_dir"]).values() for t in ts}
th_dev = dev if a.tune_on == "all" else [e for e in dev if e["kind"] == "real"]
print(f"mode {cfg['checker'].get('mode', 'passage')} | train {len(train)} | dev {len(dev)} | threshold tuned on {a.tune_on}")
print("chosen threshold:", train_checker(train, dev, cfg, gold=gold, threshold_dev=th_dev))
