"""Answerability gate: P(unanswerable) for (query, passages), from the fine-tuned S4 checker.

In-process (abstain.checker.AnswerabilityChecker) or served by Text Embeddings Inference: the checker
is a fine-tuned bge-reranker, so TEI's /rerank endpoint returns relevance r_i and
P(unanswerable) = min_i (1 - r_i), exactly as in passage mode.
"""

from __future__ import annotations

import httpx

from .settings import Settings


class TEIChecker:
    def __init__(self, url: str, threshold: float):
        self.url, self.threshold = url.rstrip("/"), threshold

    def __call__(self, query: str, passages: list[dict]) -> float:
        if not passages:
            return 1.0
        r = httpx.post(
            f"{self.url}/rerank",
            json={"query": query, "texts": [p["text"] for p in passages], "raw_scores": False, "truncate": True},
            timeout=60,
        )
        r.raise_for_status()
        return min(1.0 - float(x["score"]) for x in r.json())


def make_gate(s: Settings):
    tau = s.resolved_gate_threshold()
    if s.tei_checker_url:
        return TEIChecker(s.tei_checker_url, tau)
    from pathlib import Path

    path = Path(s.checker_path)
    weights = ("model.safetensors", "pytorch_model.bin")
    if not path.exists() or not any((path / w).exists() for w in weights):
        # no trained checker (a fresh clone has only its threshold.json, the weights are not in git):
        # the agent runs without the gate; the generator's own refusals and the self-check still apply
        import logging

        logging.getLogger(__name__).warning("no checker weights in %s: running without the answerability gate", path)
        return None
    from abstain.checker import AnswerabilityChecker

    return AnswerabilityChecker(s.checker_path, threshold=tau, device=s.device or "cpu", max_length=256)
