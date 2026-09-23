"""Loading MTRAG / MTRAG-UN tasks and corpora into one normalised format."""
from __future__ import annotations

import io
import json
import random
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator

COLLECTIONS = ("clapnq", "cloud", "fiqa", "govt")

# Labels that mean "the system should say I don't know".
ABSTAIN_LABELS = {"UNANSWERABLE"}
# Labels that mean "the system should answer".
ANSWER_LABELS = {"ANSWERABLE", "PARTIAL"}
# CONVERSATIONAL / UNDERSPECIFIED are excluded from the binary abstention analysis.


def short_collection(name: str) -> str:
    """'mt-rag-ibmcloud-elser-512-...' or 'ibmcloud' -> 'cloud'."""
    n = name.lower()
    if "cloud" in n:
        return "cloud"
    for c in ("clapnq", "fiqa", "govt"):
        if c in n:
            return c
    raise ValueError(f"Unknown collection: {name}")


def parent_doc(passage_id: str, collection: str) -> str:
    """Document a passage came from, so evidence removal can drop *all* its passages.

    Passage ids end in '-<start>-<end>' character offsets. ClapNQ ids additionally
    encode the Wikipedia page before '_' (e.g. '822086267_7384-8758-0-1374').
    """
    base = passage_id.rsplit("-", 2)[0]
    if short_collection(collection) == "clapnq":
        base = base.split("_")[0]
    return base


@dataclass
class Task:
    task_id: str
    conversation_id: str
    turn: int
    collection: str                      # short name: clapnq | cloud | fiqa | govt
    history: list[dict]                  # previous turns: {"speaker": "user"|"agent", "text": ...}
    question: str                        # current user question
    answerability: str                   # ANSWERABLE | PARTIAL | UNANSWERABLE | CONVERSATIONAL | UNDERSPECIFIED
    contexts: list[dict] = field(default_factory=list)   # reference (gold) passages
    target: str = ""
    multi_turn: str = "N/A"              # Follow-up | Clarification | N/A
    source: str = "mtrag"                # mtrag | mtragun | synthetic
    synthetic: bool = False

    @property
    def abstain_label(self) -> int | None:
        """1 = should abstain, 0 = should answer, None = excluded from analysis."""
        if self.answerability in ABSTAIN_LABELS:
            return 1
        if self.answerability in ANSWER_LABELS:
            return 0
        return None

    @property
    def gold_ids(self) -> list[str]:
        return [c["document_id"] for c in self.contexts]

    def with_(self, **kw) -> "Task":
        return replace(self, **kw)


def _first(x):
    if isinstance(x, list):
        return x[0] if x else None
    return x


def task_from_json(r: dict, source: str) -> Task:
    ans = _first(r.get("Answerability") or r.get("answerability")) or "UNKNOWN"
    turns = r["input"]
    targets = r.get("targets") or []
    return Task(
        task_id=r["task_id"],
        conversation_id=r["conversation_id"],
        turn=int(r["turn"]),
        collection=short_collection(r["Collection"]),
        history=[{"speaker": t["speaker"], "text": t["text"]} for t in turns[:-1]],
        question=turns[-1]["text"],
        answerability=str(ans).upper(),
        contexts=r.get("contexts") or [],
        target=targets[0]["text"] if targets else "",
        multi_turn=_first(r.get("Multi-Turn")) or "N/A",
        source=source,
    )


def load_tasks(path: str | Path, source: str = "mtrag") -> list[Task]:
    with open(path, encoding="utf-8") as f:
        return [task_from_json(json.loads(line), source) for line in f if line.strip()]


def load_benchmark(mtrag_dir: str | Path) -> dict[str, list[Task]]:
    d = Path(mtrag_dir)
    out = {"mtrag": load_tasks(d / "mtrag-human/generation_tasks/reference.jsonl", "mtrag")}
    un = d / "mtragun-human/generation_tasks/reference.jsonl"
    if un.exists():
        out["mtragun"] = load_tasks(un, "mtragun")
    return out


def load_rewrites(mtrag_dir: str | Path) -> dict[str, str]:
    """Human-written standalone rewrites shipped with MTRAG (used as an oracle rewriter)."""
    rw = {}
    for p in Path(mtrag_dir).glob("mtrag-human/retrieval_tasks/*/*_rewrite.jsonl"):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            rw[r["_id"]] = r["text"].replace("|user|:", "").strip()
    return rw


def iter_corpus(mtrag_dir: str | Path, collection: str) -> Iterator[dict]:
    """Yield passages {'id', 'text', 'title'} straight from the zipped passage-level corpus."""
    zname = {"cloud": "cloud", "clapnq": "clapnq", "fiqa": "fiqa", "govt": "govt"}[collection]
    zpath = Path(mtrag_dir) / "corpora/passage_level" / f"{zname}.jsonl.zip"
    with zipfile.ZipFile(zpath) as z:
        inner = [n for n in z.namelist() if n.endswith(".jsonl")][0]
        with z.open(inner) as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8"):
                r = json.loads(line)
                yield {"id": r.get("_id") or r["id"], "text": r.get("text", ""), "title": r.get("title", "")}


def conversation_split(tasks: list[Task], dev_frac: float = 0.2, seed: int = 13) -> tuple[list[Task], list[Task]]:
    """Split by conversation so no conversation leaks between train and dev."""
    convs = sorted({t.conversation_id for t in tasks})
    random.Random(seed).shuffle(convs)
    dev = set(convs[: int(len(convs) * dev_frac)])
    return [t for t in tasks if t.conversation_id not in dev], [t for t in tasks if t.conversation_id in dev]


def turn_bucket(turn: int, buckets: list[list[int]], names: list[str]) -> str:
    for (lo, hi), n in zip(buckets, names):
        if lo <= turn <= hi:
            return n
    return names[-1]


def format_history(history: list[dict], max_turns: int = 6) -> str:
    h = history[-2 * max_turns:]
    return "\n".join(f"{'User' if t['speaker'] == 'user' else 'Assistant'}: {t['text']}" for t in h)
