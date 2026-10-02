"""Build the fixed MTRAG-UN evaluation sample, the baseline metrics and the CI prediction fixture.

    python service/eval/make_sample.py --mtrag-dir data/mt-rag-benchmark \
        --predictions outputs/S5-selective_mtragun.jsonl [--n 150 --seed 13]

Writes service/eval/sample_ids.json (stratified by label x collection), service/eval/baseline.json
(metrics of the reference system on that sample) and service/eval/fixtures/predictions_sample.jsonl
(the reference predictions restricted to the sample, slimmed, so the CI gate runs without an LLM).
Re-run with --predictions <new run> to move the baseline when a release is accepted.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from abstain.data import load_benchmark
from abstain_api.evaluation import compute_metrics, records_from_predictions

ap = argparse.ArgumentParser()
ap.add_argument("--mtrag-dir", default="data/mt-rag-benchmark")
ap.add_argument("--predictions", required=True, help="reference run (MTRAG prediction format)")
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--keep-sample", action="store_true", help="reuse sample_ids.json, only refresh the baseline")
ap.add_argument("--system", default="S5-selective")
a = ap.parse_args()

if a.keep_sample:
    sample = json.loads((HERE / "sample_ids.json").read_text())
else:
    tasks = [t for t in load_benchmark(a.mtrag_dir)["mtragun"] if t.abstain_label is not None]
    strata = defaultdict(list)
    for t in sorted(tasks, key=lambda t: t.task_id):
        strata[(t.abstain_label, t.collection)].append(t)
    rng, chosen = random.Random(a.seed), []
    for key in sorted(strata):
        group = strata[key]
        k = max(1, round(a.n * len(group) / len(tasks)))
        chosen += rng.sample(group, min(k, len(group)))
    chosen.sort(key=lambda t: (t.conversation_id, t.turn))
    sample = {
        "source": "MTRAG-UN (mtragun-human)",
        "seed": a.seed,
        "n": len(chosen),
        "n_unanswerable": sum(t.abstain_label for t in chosen),
        "tasks": [
            {
                "task_id": t.task_id,
                "conversation_id": t.conversation_id,
                "turn": t.turn,
                "collection": t.collection,
                "label": t.abstain_label,
            }
            for t in chosen
        ],
    }
    (HERE / "sample_ids.json").write_text(json.dumps(sample, indent=1))

ids = {t["task_id"] for t in sample["tasks"]}
rows = [r for r in map(json.loads, open(a.predictions, encoding="utf-8")) if r["task_id"] in ids]
slim = [
    {
        "task_id": r["task_id"],
        "abstain": r["abstain"],
        "agent": {
            "decision": r.get("agent", {}).get("decision", "abstain" if r["abstain"]["abstained"] else "answer"),
            "attempts": r.get("agent", {}).get("attempts", 0),
        },
        "predictions": [{"text": r["predictions"][0]["text"]}],
    }
    for r in rows
]
(HERE / "fixtures").mkdir(exist_ok=True)
with open(HERE / "fixtures/predictions_sample.jsonl", "w", encoding="utf-8") as f:
    for r in slim:
        f.write(json.dumps(r) + "\n")
m = compute_metrics(records_from_predictions(slim, sample))
(HERE / "baseline.json").write_text(
    json.dumps({"system": a.system, "predictions": Path(a.predictions).name, "metrics": m}, indent=1)
)
print(
    f"sample: {sample['n']} tasks ({sample['n_unanswerable']} unanswerable) | baseline {a.system}: "
    f"F1 {m['abstain_f1']:.3f} recall {m['abstain_recall']:.3f} FA {m['false_abstention_rate']:.3f}"
)
