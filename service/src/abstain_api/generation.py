"""LLM generation over whole sources (not single questions): source guides and notebook overviews.

Prompts ask for plain labelled sections rather than JSON, which small local models (3B) follow far more
reliably; the parsers are lenient and every function has a fallback, so a weak model degrades the output
instead of failing the request.
"""

from __future__ import annotations

import logging
import re

from .settings import Settings

log = logging.getLogger(__name__)


def generation_model_id(s: Settings) -> str:
    from .model_registry import default_model_id

    return s.generation_model or default_model_id(s)


def get_llm(s: Settings, model_id: str | None = None):
    """An object with .chat(messages, max_new_tokens) for `model_id` (default: GENERATION_MODEL / DEFAULT_MODEL).

    Inside the API the agent runtime's LLM is reused (so tests can replace it); the worker builds its own."""
    from . import agent_runtime

    rt = agent_runtime._runtime
    llm = rt.c.llm if rt is not None else None
    if llm is None:
        from .llm import RoutingChat

        llm = RoutingChat(s)
    if hasattr(llm, "client"):  # RoutingChat: pick the model explicitly (no request context in the worker)
        return llm.client(model_id or generation_model_id(s))
    return llm


def clip_passages(passages: list[dict], max_chars: int) -> str:
    """Passages in reading order, without their contextual header line, up to max_chars."""
    out, n = [], 0
    for p in passages:
        text = p["text"]
        if p.get("section") and text.startswith(p["section"]):
            text = text[len(p["section"]) :].strip()
        if n + len(text) > max_chars:
            text = text[: max(0, max_chars - n)]
        if text:
            out.append(text)
            n += len(text) + 1
        if n >= max_chars:
            break
    return "\n".join(out)


def sample_passages(passages: list[dict], max_chars: int) -> str:
    """Evenly spread passages over a long document (beginning, middle, end) up to max_chars."""
    if sum(len(p["text"]) for p in passages) <= max_chars:
        return clip_passages(passages, max_chars)
    avg = max(1, sum(len(p["text"]) for p in passages) // len(passages))
    k = max(1, max_chars // avg)
    step = len(passages) / k
    picked = [passages[int(i * step)] for i in range(k)]
    return clip_passages(picked, max_chars)


# ----------------------------------------------------------------------------------------- source guide
GUIDE_SYSTEM = (
    "You write short guides to documents. Use only the document text you are given. "
    "Answer in exactly this format:\n"
    "SUMMARY: <3 or 4 sentences on what the document is about>\n"
    "TOPICS: <5 key topics, separated by semicolons>\n"
    "QUESTIONS:\n- <a question the document answers>\n- <another>\n- <another>"
)


def parse_sections(text: str) -> dict[str, str]:
    """'LABEL: value' blocks (value may continue over the following lines) -> {label.lower(): value}."""
    out, cur = {}, None
    for line in (text or "").splitlines():
        m = re.match(r"^\s*[#*]*\s*([A-Z][A-Z _]{1,30})\s*\**\s*:\s*\**\s*(.*)$", line)
        if m:
            cur = m.group(1).strip().lower().replace(" ", "_")
            out[cur] = m.group(2).strip()
        elif cur:
            out[cur] = (out[cur] + "\n" + line).strip()
    return out


def bullets(text: str) -> list[str]:
    items = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in (text or "").splitlines()]
    return [i for i in items if i]


def parse_guide(text: str) -> dict:
    sec = parse_sections(text)
    summary = sec.get("summary") or re.sub(r"\s+", " ", (text or "").strip())[:600]
    topics = [t.strip(" .") for t in re.split(r"[;\n]|,(?=\s*[A-Z])", sec.get("topics", "")) if t.strip(" .-*")]
    questions = [q if q.endswith("?") else q + "?" for q in bullets(sec.get("questions", ""))]
    return {
        "summary": summary.strip(),
        "topics": [re.sub(r"^[-*•]\s*", "", t) for t in topics][:6],
        "questions": [q for q in questions if len(q) > 8][:3],
    }


def source_guide(llm, title: str, passages: list[dict], s: Settings) -> dict:
    text = sample_passages(passages, s.guide_chars)
    reply = llm.chat(
        [
            {"role": "system", "content": GUIDE_SYSTEM},
            {"role": "user", "content": f"Document: {title}\n\n{text}"},
        ],
        max_new_tokens=400,
    )
    return parse_guide(reply)


# ----------------------------------------------------------------------------------------- overview
OVERVIEW = re.compile(
    r"\b(summari[sz]e|summary|overview|outline|gist|main (?:points|ideas|topics|themes|findings|takeaways)|"
    r"key (?:points|ideas|topics|themes|findings|takeaways)|what (?:are|is) (?:these|my|the|this|all) "
    r"(?:sources?|documents?|files?|notebook|papers?|material)\b.*\babout|what do (?:my|the|these) "
    r"(?:sources|documents) (?:say|cover)|compare (?:my|the|these|all) (?:sources|documents))",
    re.I,
)


def is_overview_question(q: str) -> bool:
    """Questions about the notebook as a whole ("summarise my sources"), which retrieval of 5 passages
    answers badly: they are answered from the source guides instead."""
    return bool(OVERVIEW.search(q or "")) and len(q) < 300


OVERVIEW_SYSTEM = (
    "You answer questions about a collection of sources, using ONLY the source summaries given. "
    "Cite every statement with the number of its source in square brackets, e.g. [2]. "
    "Be concise: at most 8 sentences or bullet points. If the summaries do not answer the question, say so."
)


def overview_answer(llm, question: str, guides: list[tuple[str, dict]]) -> str:
    blocks = []
    for i, (title, g) in enumerate(guides, 1):
        topics = "; ".join(g.get("topics") or [])
        blocks.append(f"[{i}] {title}\n{g.get('summary', '')}" + (f"\nTopics: {topics}" if topics else ""))
    return llm.chat(
        [
            {"role": "system", "content": OVERVIEW_SYSTEM},
            {"role": "user", "content": "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"},
        ],
        max_new_tokens=500,
    ).strip()
