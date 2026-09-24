"""End-to-end system: rewrite -> retrieve -> check answerability -> generate (or abstain).

Systems compared in the paper (see README, 'Experiments'):
    S0  last-turn query, BM25, prompt-only abstention            (weak baseline)
    S1  + LLM query rewriting
    S2  + hybrid retrieval (BM25 + BGE, RRF) + re-ranking
    S3  S2 + answerability checker trained on real labels only
    S4  S2 + checker trained on real + synthetic (evidence-removed) examples   <- proposed
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import Task
from .generate import generate_answer
from .idk import IDK_RESPONSE, is_idk


@dataclass
class Output:
    task_id: str
    query: str
    passages: list[dict]
    p_unanswerable: float | None
    abstained: bool
    answer: str
    abstained_by: str            # checker | generator | none


class AbstainRAG:
    def __init__(self, rewriter, retrievers: dict, llm=None, checker=None, k: int = 5):
        self.rewriter, self.retrievers, self.llm, self.checker, self.k = rewriter, retrievers, llm, checker, k

    def __call__(self, task: Task, generate: bool = True) -> Output:
        query = self.rewriter(task)
        passages = self.retrievers[task.collection].search(query, k=self.k)
        p_un = None
        if self.checker is not None:
            p_un = self.checker(query, passages)
            if p_un >= self.checker.threshold:
                return Output(task.task_id, query, passages, p_un, True, IDK_RESPONSE, "checker")
        if not generate or self.llm is None:
            return Output(task.task_id, query, passages, p_un, False, "", "none")
        ans = generate_answer(self.llm, task, passages)
        refused = is_idk(ans)
        return Output(task.task_id, query, passages, p_un, refused, ans, "generator" if refused else "none")


def to_mtrag_prediction(task_json: dict, out: Output) -> dict:
    """Prediction row compatible with the official MTRAG evaluation scripts (Task C format)."""
    # 'targets' is kept: the official MTRAG generation/algorithmic evaluators compare against it
    row = {k: v for k, v in task_json.items() if k not in ("predictions", "contexts")}
    row["contexts"] = [{"document_id": p["document_id"], "text": p["text"], "title": p.get("title", ""),
                        "score": p.get("score", 0.0), "source": ""} for p in out.passages]
    row["predictions"] = [{"text": out.answer}]
    row["abstain"] = {"query": out.query, "p_unanswerable": out.p_unanswerable,
                      "abstained": out.abstained, "abstained_by": out.abstained_by}
    return row
