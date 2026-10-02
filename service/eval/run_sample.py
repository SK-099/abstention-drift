"""Run the deployed agent stack live on the fixed sample and write MTRAG-format predictions.

Needs the LLM endpoint (LLM_BASE_URL / LLM_MODEL / LLM_API_KEY), Qdrant with the MTRAG corpora
(scripts/migrate_to_qdrant.py), the checker checkpoint and MTRAG_DIR. Then gate the result:

    python service/eval/run_sample.py --out eval_out/live.jsonl [--limit 20]
    python service/eval/quality_gate.py --predictions eval_out/live.jsonl

The baseline in baseline.json comes from the research run (Qwen2.5-3B via Ollama). When you change
the serving model on purpose, accept the new numbers with:
    python service/eval/make_sample.py --keep-sample --predictions eval_out/live.jsonl --system <name>
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
os.environ.setdefault("CHECKPOINTER", "memory")

from abstain_api.evaluation import load_sample, run_live
from abstain_api.settings import get_settings

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="eval_out/live.jsonl")
ap.add_argument("--limit", type=int)
a = ap.parse_args()
s = get_settings()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
rows = run_live(s, load_sample(s.eval_sample_file), limit=a.limit, out_path=a.out)
print(f"wrote {len(rows)} predictions to {a.out}")
