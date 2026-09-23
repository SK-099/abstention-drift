"""Train the answerability checker.

    python scripts/train_checker.py                   # S4: real + synthetic unanswerables
    python scripts/train_checker.py --real-only       # S3 ablation
Saves checkpoints/checker[-real]/best (+ threshold.json chosen on dev under the false-abstention budget).
"""
import argparse
import json

from abstain.checker import train_checker
from abstain.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--real-only", action="store_true")
ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. --set llm.model=Qwen/Qwen2.5-1.5B-Instruct")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
out = cfg["data"]["output_dir"]
train = [json.loads(l) for l in open(f"{out}/checker_train.jsonl")]
dev = [json.loads(l) for l in open(f"{out}/checker_dev.jsonl")]
if a.real_only:
    train = [e for e in train if e["kind"] == "real"]
    cfg["checker"]["output_dir"] += "-real"
dev_real = [e for e in dev if e["kind"] == "real"]   # threshold is tuned on real data only
print(f"train {len(train)} | dev (real) {len(dev_real)}")
print("chosen threshold:", train_checker(train, dev_real, cfg))
