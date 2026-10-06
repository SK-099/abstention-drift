"""Calibrate the refusal thresholds (gate tau, self-check min_p) for every model in the model menu.

Same data and rule family as the research calibration (scripts/retune_threshold.py + calibrate_self_check.py):
the 224 MTRAG *dev* examples of the checker (150 answerable, 74 unanswerable: 6 real + 68 synthetic), each
with its fixed passages and real conversation history; never the test set. For every model and example it
records, through exactly the client the web service uses (so reasoning models think, and <think> text is
removed, as in the app):

    p          gate score P(unanswerable)         (model-independent; computed once)
    refused    the generator's answer is a refusal
    supported  the self-check verdict on the answer (when not refused)

and then chooses, offline, the pair (tau, min_p) with the best abstention F1 among those whose false-abstention
rate on answerable questions is at most the model's own refusal rate + FA_BUDGET (default 0.10: the research
procedure allowed 5 points for the gate and 5 for the self-check):

    abstain = (p >= tau) or refused or (p >= min_p and not supported)

The research model (LLM_MODEL, qwen2.5:3b) keeps the thresholds of the research runs (tau 0.611, min_p 0.5),
so the report's numbers stay valid; its file records the dev metrics at those values and what the joint search
would choose (pass --retune-research to use that instead). Results: CALIBRATION_DIR/<model>.json (read by the
API without a restart; the model gets the shield in the menu), plus summary.md / summary.csv.

Run with Docker (writes into ./checkpoints/calibration):
    docker compose run --rm calibrate                          # every local (Ollama) model
    docker compose run --rm calibrate --models llama3.2:3b gemma3:4b
    docker compose run --rm calibrate --models openrouter:openai/gpt-4.1-mini   # API models cost money

Resumable: answers are cached per model (CALIBRATION_DIR/cache/), so an interrupted run continues where it
stopped. About 30-60 min per 3-4B model on a GTX 1650; reasoning models (qwen3, deepseek-r1) take hours.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "service/src"), str(ROOT / "src")]

from abstain.agent import self_check_supported
from abstain.data import load_benchmark
from abstain.generate import generate_answer
from abstain.idk import is_idk
from abstain_api.calibration import slug
from abstain_api.llm import RoutingChat
from abstain_api.model_registry import default_model_id, is_reasoning, split_id
from abstain_api.settings import get_settings

PROCEDURE = "joint (tau, min_p) grid on MTRAG checker dev; max F1 s.t. FA <= own refusal FA + budget"


# ------------------------------------------------------------------------------------------- selection
def evaluate(rows: list[dict], tau: float, min_p: float) -> dict:
    y = np.array([r["label"] for r in rows])
    p = np.array([r["p"] for r in rows])
    ref = np.array([bool(r["refused"]) for r in rows])
    unsup = np.array([r["supported"] is False for r in rows])
    pred = (p >= tau) | ref | ((p >= min_p) & unsup)
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    rec, fa = tp / max(int((y == 1).sum()), 1), fp / max(int((y == 0).sum()), 1)
    prec = tp / max(int(pred.sum()), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tau": float(tau), "min_p": float(min_p), "f1": f1, "recall": rec, "fa": fa, "precision": prec}


def choose(rows: list[dict], fa_budget: float) -> tuple[dict, dict]:
    """Best F1 within the false-abstention budget; ties -> the less eager setting (higher tau, then min_p)."""
    base = evaluate(rows, 1.01, 1.01)  # generator refusals only
    best = None
    for tau in np.round(np.arange(0.30, 0.951, 0.01), 2):
        for min_p in [1.01, *np.round(np.arange(0.0, tau + 1e-9, 0.05), 2)]:
            r = evaluate(rows, tau, min(min_p, 1.01))
            if r["fa"] > base["fa"] + fa_budget + 1e-9:
                continue
            key = (round(r["f1"], 6), r["tau"], r["min_p"])
            if best is None or key > best[0]:
                best = (key, r)
    return best[1], base


def stats(rows: list[dict]) -> dict:
    y = np.array([r["label"] for r in rows])
    ref = np.array([bool(r["refused"]) for r in rows])
    unsup = np.array([r["supported"] is False for r in rows])
    ans = ~ref
    return {
        "n_dev": len(rows),
        "n_unanswerable": int((y == 1).sum()),
        "refusal_rate_answerable": float(ref[y == 0].mean()) if (y == 0).any() else None,
        "refusal_rate_unanswerable": float(ref[y == 1].mean()) if (y == 1).any() else None,
        "unsupported_rate_answerable": float(unsup[ans & (y == 0)].mean()) if (ans & (y == 0)).any() else None,
        "unsupported_rate_unanswerable": float(unsup[ans & (y == 1)].mean()) if (ans & (y == 1)).any() else None,
        "seconds_per_example": float(np.mean(secs))
        if (secs := [r["seconds"] for r in rows if "seconds" in r])
        else None,
    }


# ------------------------------------------------------------------------------------------- data
def read_jsonl(f: Path) -> list[dict]:
    return [json.loads(line) for line in open(f, encoding="utf-8")] if f.exists() else []


def key_of(e: dict) -> str:
    return f"{e['task_id']}|{e['kind']}|{e['label']}"


def gate_scores(s, dev: list[dict], out: Path, research_cache: Path) -> dict[str, float]:
    f = out / "gate_dev_scores.jsonl"
    scores = {r["key"]: r["p"] for r in read_jsonl(f)}
    scores.update({r["key"]: r["p"] for r in read_jsonl(research_cache) if r["key"] not in scores})
    missing = [e for e in dev if key_of(e) not in scores]
    if missing:
        from abstain_api.checker import make_gate

        gate = make_gate(s)
        if gate is None:
            sys.exit(f"no answerability checker at {s.checker_path}: train it first (see README)")
        print(f"gate: scoring {len(missing)} dev examples", flush=True)
        with open(f, "a", encoding="utf-8") as fh:
            for e in missing:
                scores[key_of(e)] = float(gate(e["query"], e["passages"]))
                fh.write(json.dumps({"key": key_of(e), "p": scores[key_of(e)]}) + "\n")
    return scores


def run_model(s, llm, mid: str, dev, tasks, scores, cache_f: Path, limit: int | None) -> tuple[list[dict], int]:
    cache = {r["key"]: r for r in read_jsonl(cache_f)}
    todo = [e for e in dev if key_of(e) not in cache][: None if limit is None else max(0, limit - len(cache))]
    gen_kw = {"history_turns": s.history_turns, "history_chars": s.history_chars, "passage_chars": s.passage_chars}
    errors, t_start = 0, time.time()
    if todo:
        print(f"  {len(cache)} cached, {len(todo)} to run", flush=True)
    with open(cache_f, "a", encoding="utf-8") as fh:
        for i, e in enumerate(todo, 1):
            t0 = time.time()
            try:
                ans = generate_answer(llm, tasks[e["task_id"]], e["passages"], **gen_kw)
                refused = is_idk(ans)
                sc = None
                if not refused:
                    sc = bool(self_check_supported(llm, e["query"], e["passages"], ans, s.passage_chars))
            except Exception as ex:  # timeout / server error: not cached, retried on the next run
                errors += 1
                print(f"  ! {e['task_id']}: {str(ex)[:160]}", flush=True)
                if errors >= 10 and errors > i // 2:
                    print("  too many errors, skipping this model for now", flush=True)
                    break
                continue
            r = {
                "key": key_of(e),
                "label": e["label"],
                "kind": e["kind"],
                "p": scores[key_of(e)],
                "refused": refused,
                "supported": sc,
                "answer": ans[:300],
                "seconds": round(time.time() - t0, 2),
            }
            cache[r["key"]] = r
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t_start
                eta = el / i * (len(todo) - i)
                print(f"  {i}/{len(todo)}  {el / i:5.1f} s/example  ETA {eta / 60:5.1f} min", flush=True)
    keys = {key_of(e) for e in dev}
    return [r for k, r in cache.items() if k in keys], errors


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--models",
        nargs="*",
        default=["all"],
        help="model ids ('local:llama3.2:3b', 'openrouter:...') or Ollama tags; 'all' = every local model",
    )
    ap.add_argument("--dev", default="outputs/checker_dev.jsonl")
    ap.add_argument(
        "--research-cache",
        default="outputs/selfcheck_calib_dev.jsonl",
        help="answers of the research model from scripts/calibrate_self_check.py (reused, not re-run)",
    )
    ap.add_argument("--mtrag-dir", default=s.mtrag_dir)
    ap.add_argument("--out", default=s.calibration_dir)
    ap.add_argument("--fa-budget", type=float, default=0.10)
    ap.add_argument(
        "--limit", type=int, default=None, help="only N dev examples per model (smoke test; not saved as a calibration)"
    )
    ap.add_argument("--retune-research", action="store_true", help="also use the joint search for the research model")
    ap.add_argument("--summary-only", action="store_true", help="rebuild summary.md from existing files")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    research_id = default_model_id(s) if default_model_id(s).startswith("local:") else f"local:{s.llm_model}"

    if not a.summary_only:
        dev = read_jsonl(Path(a.dev))
        if not dev:
            sys.exit(f"{a.dev} not found: it is written by scripts/train_checker.py (research pipeline)")
        tasks = {t.task_id: t for ts in load_benchmark(a.mtrag_dir).values() for t in ts}
        scores = gate_scores(s, dev, out, Path(a.research_cache))

        if a.models == ["all"]:
            from abstain_api.model_registry import _SKIP, ollama_base, ollama_tags

            try:
                tags = ollama_tags(ollama_base(s))
            except Exception as e:
                sys.exit(
                    f"could not list the Ollama models at {ollama_base(s)}/api/tags ({e}).\n"
                    "Is Ollama running (`ollama list`)? Or name the models: --models llama3.2:3b gemma3:4b ..."
                )
            names = [m.get("name") or m.get("model") for m in tags.get("models", [])]
            ids = [research_id] + [f"local:{n}" for n in names if n and not _SKIP.search(n)]
        else:
            ids = [
                m
                if ":" in m
                and m.split(":", 1)[0]
                in {"local", "openrouter", "openai", "anthropic", "gemini", "groq", "mistral", "deepseek", "together"}
                else f"local:{m}"
                for m in a.models
            ]
        # quick models first: reasoning models take hours
        ids = sorted(dict.fromkeys(ids), key=lambda m: (m != research_id, is_reasoning(split_id(m)[1]), m))
        print(f"dev: {len(dev)} examples; models: {', '.join(ids)}", flush=True)

        rc = RoutingChat(s)
        for mid in ids:
            print(f"\n== {mid}", flush=True)
            cache_f = out / "cache" / f"{slug(mid)}.jsonl"
            if mid == research_id and not cache_f.exists() and Path(a.research_cache).exists():
                rows = [r for r in read_jsonl(Path(a.research_cache))]
                cache_f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                print(f"  reusing {len(rows)} answers from {a.research_cache}", flush=True)
            try:
                llm = rc.client(mid)
                rows, errors = run_model(s, llm, mid, dev, tasks, scores, cache_f, a.limit)
            except KeyboardInterrupt:
                print("\ninterrupted: answers so far are cached, run again to continue")
                return
            complete = len(rows) == len(dev)
            if not rows:
                continue
            best, base = choose(rows, a.fa_budget)
            research = mid == research_id and not a.retune_research
            chosen = evaluate(rows, s.resolved_gate_threshold(), s.resolved_min_p()) if research else best
            rec = {
                "model_id": mid,
                **chosen,
                "source": "research" if research else "calibrated",
                "complete": complete,
                **stats(rows),
                "no_gate_no_selfcheck": base,
                "joint_search_choice": best,
                "fa_budget": a.fa_budget,
                "procedure": PROCEDURE,
                "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            dst = out / f"{slug(mid)}.json" if complete else out / "partial" / f"{slug(mid)}.json"
            dst.parent.mkdir(exist_ok=True)
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
            os.replace(tmp, dst)  # the API never reads a half-written file
            print(
                f"  tau={rec['tau']:.2f} min_p={rec['min_p']:.2f}  F1={rec['f1']:.3f} recall={rec['recall']:.3f} "
                f"FA={rec['fa']:.3f} (own refusals {base['fa']:.3f})  -> {dst}"
                + ("" if complete else f"  [partial: {len(rows)}/{len(dev)}, not used by the API]"),
                flush=True,
            )
    write_summary(out)


def write_summary(out: Path) -> None:
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(out.glob("*.json"))]
    recs = [r for r in recs if r.get("model_id")]
    recs.sort(key=lambda r: -r["f1"])
    cols = [
        "model_id",
        "source",
        "tau",
        "min_p",
        "f1",
        "recall",
        "fa",
        "precision",
        "refusal_rate_answerable",
        "refusal_rate_unanswerable",
        "seconds_per_example",
        "n_dev",
    ]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in recs:
            w.writerow([r.get(c) for c in cols])
    f = lambda v: "-" if v is None else f"{v:.2f}"
    lines = [
        "# Per-model refusal thresholds (MTRAG checker dev, 150 answerable / 74 unanswerable)",
        "",
        "Abstain if p >= tau, or the model refuses, or (p >= min_p and the self-check says unsupported).",
        "FA = answerable questions refused. Own refusals = FA with no gate and no self-check.",
        "",
        "| Model | Thresholds | tau | min_p | F1 | Recall | FA | Own refusals | s / question |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        lines.append(
            f"| {r['model_id']} | {r['source']} | {f(r['tau'])} | {f(r['min_p']) if r['min_p'] <= 1 else 'off'} "
            f"| {f(r['f1'])} | {f(r['recall'])} | {f(r['fa'])} | {f(r.get('refusal_rate_answerable'))} "
            f"| {f(r.get('seconds_per_example'))} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsummary: {out / 'summary.md'}\n" + "\n".join(lines[5:]))


if __name__ == "__main__":
    main()
