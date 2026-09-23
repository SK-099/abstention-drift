"""Answerability by turn position for MTRAG and MTRAG-UN (CPU only, ~2 s).

Usage: python scripts/dataset_stats.py [--config configs/default.yaml]
Writes results/dataset_stats.md
"""
import argparse
from collections import Counter
from pathlib import Path

from abstain.config import load_config
from abstain.data import load_benchmark, turn_bucket

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. --set llm.model=Qwen/Qwen2.5-1.5B-Instruct")
args = ap.parse_args()
cfg = load_config(args.config, args.set)
B, N = cfg["eval"]["buckets"], cfg["eval"]["bucket_names"]

lines = ["# Dataset statistics", ""]
for name, tasks in load_benchmark(cfg["data"]["mtrag_dir"]).items():
    labels = Counter(t.answerability for t in tasks)
    lines += [f"## {name}: {len(tasks)} tasks, {len({t.conversation_id for t in tasks})} conversations", "",
              "Labels: " + ", ".join(f"{k} {v}" for k, v in labels.most_common()), "",
              "| Turn bucket | Tasks | Unanswerable | Answerable + partial | % unanswerable |",
              "| --- | --- | --- | --- | --- |"]
    for n in N:
        g = [t for t in tasks if turn_bucket(t.turn, B, N) == n]
        u = sum(t.abstain_label == 1 for t in g)
        a = sum(t.abstain_label == 0 for t in g)
        lines.append(f"| {n} | {len(g)} | {u} | {a} | {100 * u / max(len(g), 1):.1f}% |")
    lines.append("")
out = "\n".join(lines)
Path("results").mkdir(exist_ok=True)
Path("results/dataset_stats.md").write_text(out)
print(out)
