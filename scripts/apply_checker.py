"""Derive the checker systems (S2-rr, S3, S4) from a finished S2 run without re-running the LLM.

S2, S2-rr, S3 and S4 share the same rewrite, retrieval and (deterministic, temperature-0) generation;
they differ only in the answerability gate in front of the generator. So for each S2 prediction we
run the gate on the stored (query, passages): if it fires, the answer becomes the IDK response;
otherwise S2's answer is kept. This gives the same outputs as running each system end to end, in
minutes instead of ~3 hours per system.

    python scripts/apply_checker.py --config configs/laptop.yaml                  # S2-rr, S3, S4 from outputs/S2_mtragun.jsonl
    python scripts/apply_checker.py --config configs/laptop.yaml --split dev --systems S4
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from abstain.checker import AnswerabilityChecker, RerankerScoreChecker
from abstain.config import load_config
from abstain.idk import IDK_RESPONSE

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--split", default="mtragun", choices=["dev", "mtragun"])
ap.add_argument("--systems", nargs="+", default=["S2-rr", "S3", "S4"], choices=["S2-rr", "S3", "S4"])
ap.add_argument("--set", nargs="*", default=[], help="config overrides")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
c = cfg["checker"]
out = Path(cfg["data"]["output_dir"])
src = out / f"S2_{a.split}.jsonl"
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
print(f"{len(rows)} S2 predictions from {src}")

for name in a.systems:
    if name == "S2-rr":
        gate = RerankerScoreChecker(str(Path(c["output_dir"]).parent / "reranker_checker.json"))
        probs = [gate.p_unanswerable([p.get("score", 0.0) for p in r["contexts"]]) for r in rows]
    else:
        path = c["output_dir"] + ("-real" if name == "S3" else "") + "/best"
        gate = AnswerabilityChecker(path, device=c.get("infer_device"), max_length=c.get("max_length", 512))
        probs = []
        for r in tqdm(rows, desc=name):
            probs.append(gate(r["abstain"]["query"], r["contexts"]))
    n_gate = 0
    with open(out / f"{name}_{a.split}.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(rows, probs):
            r2 = dict(r, abstain=dict(r["abstain"], p_unanswerable=float(p)))
            if p >= gate.threshold:
                n_gate += 1
                r2["predictions"] = [{"text": IDK_RESPONSE}]
                r2["abstain"].update(abstained=True, abstained_by="checker")
            f.write(json.dumps(r2) + "\n")
    print(f"{name}: checker abstained on {n_gate}/{len(rows)} (threshold {gate.threshold:.3f}) -> {out / f'{name}_{a.split}.jsonl'}")
