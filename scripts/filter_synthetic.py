"""Drop synthetic 'unanswerable' examples whose passages still answer the question (LLM judge).

Evidence removal drops the gold *documents*, but the corpora contain near-duplicates (several NASA
pages on the same mission, several Wikipedia pages on one team), so many synthetic examples are in
fact answerable. This filters the existing checker data in place (originals kept as
*_unfiltered.jsonl); no re-retrieval needed. Decisions are cached, so the script can be re-run
after an interruption.

    python scripts/filter_synthetic.py --config configs/laptop.yaml
"""
import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from abstain.config import load_config
from abstain.data import load_benchmark
from abstain.llm import make_llm
from abstain.synth import llm_still_answerable

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--set", nargs="*", default=[], help="config overrides")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
out = Path(cfg["data"]["output_dir"])
tasks = {t.task_id: t for ts in load_benchmark(cfg["data"]["mtrag_dir"]).values() for t in ts}
llm = make_llm(cfg["llm"], max_new_tokens=3)

cache_f = out / "verify_cache.jsonl"
cache = {}
if cache_f.exists():
    for line in open(cache_f, encoding="utf-8"):
        r = json.loads(line)
        cache[r["key"]] = r["answerable"]

with open(cache_f, "a", encoding="utf-8") as cf:
    for split in ("train", "dev"):
        f, orig = out / f"checker_{split}.jsonl", out / f"checker_{split}_unfiltered.jsonl"
        if not orig.exists():
            shutil.copy(f, orig)
        rows = [json.loads(l) for l in open(orig, encoding="utf-8")]
        kept, dropped, failed = [], Counter(), 0
        for e in tqdm(rows, desc=split):
            if e["kind"] == "synthetic":
                key = f"{e['task_id']}|{'|'.join(p['document_id'] for p in e['passages'])}"
                if key not in cache:
                    for attempt in range(4):          # Ollama occasionally returns HTTP 500; retry, then skip
                        try:
                            cache[key] = llm_still_answerable(llm, tasks[e["task_id"]], e["passages"])
                            break
                        except Exception as ex:
                            print(f"\nLLM call failed ({ex}); retrying in 20 s")
                            time.sleep(20)
                    else:
                        failed += 1                   # not cached: judged on the next run
                        continue
                    cf.write(json.dumps({"key": key, "answerable": cache[key]}) + "\n")
                    cf.flush()
                if cache[key]:
                    dropped[e["collection"]] += 1
                    continue
            kept.append(e)
        with open(f, "w", encoding="utf-8") as fh:
            for e in kept:
                fh.write(json.dumps(e) + "\n")
        n_syn = sum(e["kind"] == "synthetic" for e in rows)
        print(f"{split}: synthetic {n_syn} -> kept {n_syn - sum(dropped.values()) - failed} "
              f"(dropped as still answerable, by collection: {dict(dropped)}) | total rows {len(rows)} -> {len(kept)}"
              + (f" | {failed} not judged (LLM errors): re-run to finish" if failed else ""))
