"""Evidence removal: turn answerable turns into (synthetic) unanswerable ones for free.

For an answerable turn we know which passages contain the answer (MTRAG reference passages).
Retrieving again while *excluding every passage from those source documents* yields a context
that looks like a normal retrieval result but no longer supports the answer.

Both classes of checker-training examples are built from retrieved passages, so the checker cannot
learn a shortcut such as "gold passages look different from retrieved ones":
    answerable   = retrieved top-k with the gold passages inserted   (label 0)
    unanswerable = retrieved top-k with all gold source documents removed (label 1)

Known limitation: another document may still contain the answer (false 'unanswerable') -- FiQA in
particular has many near-duplicate forum answers. We therefore drop a synthetic example when the
remaining passages still cover most of the reference answer's content words (`answer_support`).
Hand-check ~50 kept examples and report the residual noise rate in the paper.
"""
from __future__ import annotations

import random
import re

from .data import Task, parent_doc

_STOP = set("""a an the and or but if of to in on at for by with from as is are was were be been being it its
this that these those you your i we they he she them his her our their not no yes can could would should will
do does did have has had so than then there here what which who whom how when where why also about into more
most such only just very may might must one any all some other each both""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def answer_support(target: str, passages: list[dict]) -> float:
    """Fraction of the reference answer's content words that appear in the passages (0..1)."""
    tw = content_words(target)
    if not tw:
        return 0.0
    pw = set().union(*(content_words(p["text"]) for p in passages)) if passages else set()
    return len(tw & pw) / len(tw)


VERIFY_PROMPT = (
    "Question: {q}\n\nReference answer: {a}\n\nPassages:\n{p}\n\n"
    "Do the passages contain the main information needed to answer the question, i.e. the key point "
    "of the reference answer? Extra details of the reference answer may be missing. "
    "Reply with exactly one word: yes or no."
)


def llm_still_answerable(llm, task: Task, passages: list[dict], passage_chars: int = 1000) -> bool:
    """LLM check against the *reference answer* (easier and less biased than judging answerability blind).
    Near-duplicate pages (same site, same Wikipedia topic) often still answer after the gold document is
    removed; those examples must be dropped, not labelled unanswerable."""
    ptxt = "\n\n".join(f"[{i + 1}] {p['text'][:passage_chars]}" for i, p in enumerate(passages))
    msg = VERIFY_PROMPT.format(q=task.question, a=task.target[:600], p=ptxt)
    return llm.chat([{"role": "user", "content": msg}], max_new_tokens=3).strip().lower().startswith("yes")


def gold_parents(task: Task) -> set[str]:
    return {parent_doc(pid, task.collection) for pid in task.gold_ids}


def remove_evidence(ranked: list[dict], parents: set[str], collection: str, k: int) -> list[dict]:
    """Keep the top-k retrieved passages that do not come from any gold source document."""
    return [p for p in ranked if parent_doc(p["document_id"], collection) not in parents][:k]


def insert_gold(ranked: list[dict], gold: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Top-k retrieved passages with every gold passage guaranteed to be present, at random positions."""
    gold_ids = {g["document_id"] for g in gold}
    others = [p for p in ranked if p["document_id"] not in gold_ids][: max(k - len(gold), 0)]
    out = others[:]
    for g in gold[:k]:
        out.insert(rng.randint(0, len(out)), g)
    return out[:k]


def checker_examples(tasks: list[Task], retrieve, query_fn, k: int = 5, seed: int = 13,
                     include_real: bool = True, synthesize: bool = True, max_support: float = 0.5,
                     verify=None) -> list[dict]:
    """Build training/dev examples for the answerability checker.

    retrieve(task, query, exclude_parents) -> ranked passages (dicts with document_id, text)
    query_fn(task) -> retrieval query (e.g. the rewritten question)
    verify(task, passages) -> True if the passages STILL answer the question (example is then dropped);
        e.g. functools.partial(llm_still_answerable, llm). Lexical `answer_support` misses paraphrases.
    """
    rng = random.Random(seed)
    out = []
    for t in tasks:
        label = t.abstain_label
        if label is None:
            continue
        q = query_fn(t)
        if label == 1 and include_real:                       # real unanswerable: whatever retrieval returns
            out.append(_ex(t, q, retrieve(t, q, None)[:k], 1, "real"))
        elif label == 0 and t.contexts:
            ranked = retrieve(t, q, None)
            if include_real:
                out.append(_ex(t, q, insert_gold(ranked, t.contexts, k, rng), 0, "real"))
            if synthesize:
                parents = gold_parents(t)
                deep = retrieve(t, q, parents)                # retriever filters gold documents itself
                ctx = remove_evidence(deep, parents, t.collection, k)
                if ctx and answer_support(t.target, ctx) < max_support and not (verify and verify(t, ctx)):
                    out.append(_ex(t, q, ctx, 1, "synthetic"))
    return out


def _ex(t: Task, query: str, passages: list[dict], label: int, kind: str) -> dict:
    return {"task_id": t.task_id, "conversation_id": t.conversation_id, "turn": t.turn,
            "collection": t.collection, "multi_turn": t.multi_turn, "query": query,
            "question": t.question, "passages": [{"document_id": p["document_id"], "text": p["text"]} for p in passages],
            "label": label, "kind": kind, "gold_ids": t.gold_ids}
