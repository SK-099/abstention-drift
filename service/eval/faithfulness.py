"""Faithfulness of the answered turns (Ragas, or DeepEval) on a predictions file.

    pip install "abstain-service[eval]"          # ragas
    python service/eval/faithfulness.py --predictions eval_out/live.jsonl --judge-model gpt-4o-mini \
        [--judge-base-url https://openrouter.ai/api/v1] [--backend ragas|deepeval] [--min 0.7]

Only turns where the agent answered are scored (refusals have nothing to be faithful to). The judge
is any OpenAI-compatible model (JUDGE_API_KEY). Exit code 1 if the mean is below --min.
Needs the full prediction rows (with "contexts"), i.e. the output of run_sample.py, not the slim fixture.
"""

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--predictions", required=True)
ap.add_argument("--backend", default="ragas", choices=["ragas", "deepeval"])
ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "gpt-4o-mini"))
ap.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL"))
ap.add_argument("--min", type=float, default=0.0)
ap.add_argument("--limit", type=int)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.predictions, encoding="utf-8")]
rows = [r for r in rows if not r["abstain"]["abstained"] and r.get("contexts")][: a.limit]
if not rows:
    sys.exit("no answered turns with contexts in the file")


def question(r):
    return next(m["text"] for m in reversed(r["input"]) if m["speaker"] == "user")


items = [(question(r), r["predictions"][0]["text"], [c["text"] for c in r["contexts"]]) for r in rows]
key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")

if a.backend == "ragas":
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    judge = LangchainLLMWrapper(ChatOpenAI(model=a.judge_model, base_url=a.judge_base_url, api_key=key, temperature=0))
    ds = EvaluationDataset.from_list(
        [{"user_input": q, "response": ans, "retrieved_contexts": ctx} for q, ans, ctx in items]
    )
    res = evaluate(ds, metrics=[Faithfulness(llm=judge)])
    scores = [float(x) for x in res.to_pandas()["faithfulness"].fillna(0)]
else:
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.models import GPTModel
    from deepeval.test_case import LLMTestCase

    model = GPTModel(model=a.judge_model, base_url=a.judge_base_url, _openai_api_key=key)
    scores = []
    for q, ans, ctx in items:
        m = FaithfulnessMetric(model=model, include_reason=False)
        m.measure(LLMTestCase(input=q, actual_output=ans, retrieval_context=ctx))
        scores.append(m.score)

mean = sum(scores) / len(scores)
print(f"faithfulness ({a.backend}, judge {a.judge_model}) on {len(scores)} answered turns: {mean:.3f}")
sys.exit(0 if mean >= a.min else 1)
