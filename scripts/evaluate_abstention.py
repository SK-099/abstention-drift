"""Abstention metrics per turn bucket + abstention drift (RQ1) + false-abstention (RQ2).

    python scripts/evaluate_abstention.py outputs/S0_mtragun.jsonl outputs/S4_mtragun.jsonl

For answer quality (faithfulness, RB_llm, IDK-conditioned scores) run the official MTRAG scripts
on the same files: data/mt-rag-benchmark/scripts/evaluation/run_generation_eval.py
"""
import argparse
import glob
import json
from pathlib import Path

from abstain.config import load_config
from abstain.data import task_from_json
from abstain.metrics import abstention_report, controlled_logit, drift, report_by_bucket

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
ap.add_argument("--plot", default="results/drift.png")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
# expand wildcards ourselves: Windows shells pass "outputs/S*.jsonl" through literally
a.files = sorted({f for pat in a.files for f in (glob.glob(pat) or [pat])})
B, N, NB = cfg["eval"]["buckets"], cfg["eval"]["bucket_names"], cfg["eval"]["n_boot"]


def records(path):
    rs = []
    for row in map(json.loads, open(path, encoding="utf-8")):
        if "label" in row and "pred" in row:          # synthetic-probe format
            rs.append(row)
            continue
        t = task_from_json(row | {"targets": row.get("targets", [])}, "eval")
        if t.abstain_label is None:
            continue
        rs.append({"conversation_id": t.conversation_id, "turn": t.turn, "label": t.abstain_label,
                   "pred": int(row["abstain"]["abstained"]), "collection": t.collection, "multi_turn": t.multi_turn})
    return rs


lines, curves = ["# Abstention results", ""], {}
for f in a.files:
    rs, name = records(f), Path(f).stem
    tot = abstention_report([r["label"] for r in rs], [r["pred"] for r in rs])
    byb = report_by_bucket(rs, B, N)
    dr, fa, lg = drift(rs, "recall", NB), drift(rs, "false_abstain", NB), controlled_logit(rs, 1)
    curves[name] = [byb[n]["abstain_recall"] for n in N]
    lines += [f"## {name}", "",
              f"Overall: abstain F1 {tot['abstain_f1']:.3f} | recall {tot['abstain_recall']:.3f} | "
              f"precision {tot['abstain_precision']:.3f} | false-abstention {tot['false_abstention_rate']:.3f} (n={tot['n']})", "",
              "| Bucket | n unans. | Abstain recall | n ans. | False-abstention |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {n} | {byb[n]['n_unanswerable']} | {byb[n]['abstain_recall']:.3f} | {byb[n]['n_answerable']} | "
              f"{byb[n]['false_abstention_rate']:.3f} |" for n in N]
    lines += ["", f"**Abstention drift (recall slope / turn):** {dr['slope']:+.4f} "
                  f"[95% CI {dr['ci_low']:+.4f}, {dr['ci_high']:+.4f}], {dr['n_conversations']} conversations",
              f"**False-abstention slope / turn:** {fa['slope']:+.4f} [{fa['ci_low']:+.4f}, {fa['ci_high']:+.4f}]",
              f"**Controlled logit (domain + follow-up type):** odds ratio per turn {lg['odds_ratio_per_turn']:.3f}", ""]
text = "\n".join(lines)
Path("results").mkdir(exist_ok=True)
Path("results/abstention.md").write_text(text)
print(text)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name, ys in curves.items():
        ax.plot(N, ys, marker="o", label=name)
    ax.set_xlabel("Turn position"); ax.set_ylabel("Abstention recall (unanswerable)")
    ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(a.plot, dpi=200); print("plot ->", a.plot)
except Exception as e:  # plotting is optional
    print("plot skipped:", e)
