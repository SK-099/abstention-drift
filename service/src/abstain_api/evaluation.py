"""Release evaluation on a fixed MTRAG-UN sample: abstention metrics + a quality gate vs a stored baseline.

Used by service/eval/quality_gate.py (CI), service/eval/run_sample.py (live run against the deployed
stack) and the nightly Celery job. A *record* is {task_id, conversation_id, turn, collection, label, pred,
decision}; label 1 = the turn is unanswerable (system should abstain).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from abstain.metrics import abstention_report

from .settings import Settings

# Quality-gate limits (overridable from the CLI / env through quality_gate.py).
MAX_F1_DROP = 0.03  # PR fails if abstain-F1 falls more than this below the baseline
MAX_FALSE_ABSTENTION = 0.25  # ... or if wrong refusals on answerable turns exceed this rate
MAX_FALSE_ABSTENTION_RISE = 0.05  # ... or rise more than this above the baseline


def load_sample(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_metrics(records: list[dict]) -> dict:
    rep = abstention_report([r["label"] for r in records], [r["pred"] for r in records])
    late = [r for r in records if int(r["turn"]) >= 5]
    late_rep = abstention_report([r["label"] for r in late], [r["pred"] for r in late]) if late else {}
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}
    out["late_abstain_recall"] = round(late_rep.get("abstain_recall", float("nan")), 4) if late else None
    out["clarify_rate"] = round(sum(r.get("decision") == "clarify" for r in records) / max(len(records), 1), 4)
    return out


def quality_gate(
    metrics: dict,
    baseline: dict,
    max_f1_drop: float = MAX_F1_DROP,
    max_false_abstention: float = MAX_FALSE_ABSTENTION,
    max_false_abstention_rise: float = MAX_FALSE_ABSTENTION_RISE,
) -> dict:
    f1, base_f1 = metrics["abstain_f1"], baseline["abstain_f1"]
    fa, base_fa = metrics["false_abstention_rate"], baseline["false_abstention_rate"]
    checks = [
        {
            "name": "abstain_f1",
            "value": f1,
            "limit": round(base_f1 - max_f1_drop, 4),
            "ok": f1 >= base_f1 - max_f1_drop - 1e-9,
            "rule": f"F1 must not drop more than {max_f1_drop} below the baseline {base_f1:.3f}",
        },
        {
            "name": "false_abstention_rate",
            "value": fa,
            "limit": max_false_abstention,
            "ok": fa <= max_false_abstention + 1e-9,
            "rule": f"wrong refusals must stay <= {max_false_abstention}",
        },
        {
            "name": "false_abstention_rise",
            "value": round(fa - base_fa, 4),
            "limit": max_false_abstention_rise,
            "ok": fa - base_fa <= max_false_abstention_rise + 1e-9,
            "rule": f"wrong refusals must not rise more than {max_false_abstention_rise} above the baseline {base_fa:.3f}",
        },
    ]
    return {"passed": all(c["ok"] for c in checks), "checks": checks}


def records_from_predictions(rows: list[dict], sample: dict) -> list[dict]:
    """Rows in MTRAG prediction format (as written by run_agent.py / run_sample.py) -> records."""
    labels = {t["task_id"]: t for t in sample["tasks"]}
    recs = []
    for r in rows:
        meta = labels.get(r["task_id"])
        if meta is None:
            continue
        recs.append(
            {
                "task_id": r["task_id"],
                "conversation_id": meta["conversation_id"],
                "turn": meta["turn"],
                "collection": meta["collection"],
                "label": meta["label"],
                "pred": int(r["abstain"]["abstained"]),
                "decision": r.get("agent", {}).get("decision", "abstain" if r["abstain"]["abstained"] else "answer"),
            }
        )
    missing = len(labels) - len(recs)
    if missing:
        raise ValueError(f"{missing} sample tasks have no prediction")
    return recs


def run_live(s: Settings, sample: dict, limit: int | None = None, runtime=None, out_path: str | Path | None = None):
    """Run the deployed agent (LLM + Qdrant + checker) on the sample; returns MTRAG-format rows."""
    from abstain.agent import to_prediction
    from abstain.data import load_benchmark

    from .agent_runtime import get_runtime

    rt = runtime or get_runtime(s)
    ids = [t["task_id"] for t in sample["tasks"]][:limit]
    bench = {t.task_id: t for t in load_benchmark(s.mtrag_dir)["mtragun"]}
    raw_path = Path(s.mtrag_dir) / "mtragun-human/generation_tasks/reference.jsonl"
    raw = {r["task_id"]: r for r in map(json.loads, open(raw_path, encoding="utf-8"))}
    rows = []
    f = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for tid in ids:
            out = rt.agent(bench[tid], thread_id=f"eval-{uuid.uuid4()}")
            row = to_prediction(raw[tid], out)
            rows.append(row)
            if f:
                f.write(json.dumps(row) + "\n")
                f.flush()
    finally:
        if f:
            f.close()
    return rows


def run_sample_eval(s: Settings, limit: int | None = None, predictions: str | Path | None = None, runtime=None) -> dict:
    """Nightly / release evaluation. With `predictions`, score an existing file instead of running live."""
    from .prompts import versions

    sample = load_sample(s.eval_sample_file)
    if limit:
        sample = {**sample, "tasks": sample["tasks"][:limit]}
    t0 = time.time()
    if predictions:
        rows = [json.loads(l) for l in open(predictions, encoding="utf-8")]
    else:
        rows = run_live(s, sample, runtime=runtime)
    m = compute_metrics(records_from_predictions(rows, sample))
    m["seconds"] = round(time.time() - t0, 1)
    baseline = json.loads(Path(s.eval_baseline_file).read_text(encoding="utf-8"))["metrics"]
    return {"metrics": m, "gate": quality_gate(m, baseline), "versions": versions(s)}
