"""CI quality gate: score predictions on the fixed MTRAG-UN sample and fail if quality regressed.

    python service/eval/quality_gate.py                                   # committed fixture (no LLM needed)
    python service/eval/quality_gate.py --predictions eval_out/live.jsonl # a live run (run_sample.py)
    python service/eval/quality_gate.py ... --max-f1-drop 0.03 --max-false-abstention 0.25

Exit code 1 if abstain-F1 drops more than --max-f1-drop below baseline.json, or the false-abstention
(wrong refusal) rate exceeds --max-false-abstention or rises more than --max-false-abstention-rise.
Writes a Markdown summary (for the GitHub job summary / PR comment) with --summary.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from abstain_api.evaluation import (
    MAX_F1_DROP,
    MAX_FALSE_ABSTENTION,
    MAX_FALSE_ABSTENTION_RISE,
    compute_metrics,
    quality_gate,
    records_from_predictions,
)

env = os.environ.get
ap = argparse.ArgumentParser()
ap.add_argument("--predictions", default=str(HERE / "fixtures/predictions_sample.jsonl"))
ap.add_argument("--sample", default=str(HERE / "sample_ids.json"))
ap.add_argument("--baseline", default=str(HERE / "baseline.json"))
ap.add_argument("--max-f1-drop", type=float, default=float(env("GATE_MAX_F1_DROP", MAX_F1_DROP)))
ap.add_argument(
    "--max-false-abstention", type=float, default=float(env("GATE_MAX_FALSE_ABSTENTION", MAX_FALSE_ABSTENTION))
)
ap.add_argument(
    "--max-false-abstention-rise",
    type=float,
    default=float(env("GATE_MAX_FALSE_ABSTENTION_RISE", MAX_FALSE_ABSTENTION_RISE)),
)
ap.add_argument("--summary", help="append a Markdown summary to this file (e.g. $GITHUB_STEP_SUMMARY)")
ap.add_argument("--json-out")
a = ap.parse_args()

sample = json.loads(Path(a.sample).read_text())
baseline = json.loads(Path(a.baseline).read_text())
rows = [json.loads(l) for l in open(a.predictions, encoding="utf-8")]
m = compute_metrics(records_from_predictions(rows, sample))
g = quality_gate(m, baseline["metrics"], a.max_f1_drop, a.max_false_abstention, a.max_false_abstention_rise)

b = baseline["metrics"]
md = [
    f"### Abstention quality gate: {'PASSED' if g['passed'] else 'FAILED'}",
    "",
    f"Sample: {m['n']} MTRAG-UN turns ({m['n_unanswerable']} unanswerable); baseline: {baseline['system']}",
    "",
    "| Metric | This run | Baseline |",
    "| --- | --- | --- |",
]
for k in ("abstain_f1", "abstain_recall", "abstain_precision", "false_abstention_rate", "late_abstain_recall"):
    md.append(f"| {k} | {m[k]} | {b.get(k)} |")
md += ["", "| Check | Value | Limit | Result |", "| --- | --- | --- | --- |"]
md += [f"| {c['rule']} | {c['value']} | {c['limit']} | {'ok' if c['ok'] else 'FAIL'} |" for c in g["checks"]]
text = "\n".join(md)
print(text)
if a.summary:
    with open(a.summary, "a", encoding="utf-8") as f:
        f.write(text + "\n")
if a.json_out:
    Path(a.json_out).write_text(json.dumps({"metrics": m, "gate": g}, indent=1))
sys.exit(0 if g["passed"] else 1)
