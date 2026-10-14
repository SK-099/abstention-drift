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


# ----------------------------------------------------------------------------------------- Studio
KINDS = {
    "briefing": "Briefing doc",
    "study_guide": "Study guide",
    "faq": "FAQ",
    "timeline": "Timeline",
    "mindmap": "Mind map",
    "flashcards": "Flashcards",
    "quiz": "Quiz",
    "audio": "Audio overview",
}

_CITE = "Cite the source of every point with its number in square brackets, e.g. [2]."
STUDIO_PROMPTS = {
    "briefing": (
        "Write a briefing document about the sources in Markdown with these sections:\n"
        "## Executive summary\n(3-4 sentences)\n## Key points\n(5-8 bullet points)\n"
        "## Important details\n(facts, figures, names, quotes)\n## Open questions\n(2-3 bullet points)\n" + _CITE,
        1100,
    ),
    "study_guide": (
        "Write a study guide for the sources in Markdown with these sections:\n"
        "## Key concepts\n(bullet points: **term** - explanation)\n## Short-answer questions\n(5 numbered questions)\n"
        "## Answer key\n(the 5 answers, numbered)\n## Essay questions\n(3 numbered questions)\n" + _CITE,
        1200,
    ),
    "faq": (
        "Write 6 to 8 frequently asked questions about the sources with their answers. Use exactly this format "
        "for each one:\nQ: <question>\nA: <answer of 1-3 sentences>\n" + _CITE,
        1000,
    ),
    "timeline": (
        "List the events in the sources in chronological order, one per line, in exactly this format:\n"
        "<date or period> | <what happened> [source number]\n"
        "Use only dates or periods that appear in the sources. If the sources contain no dates, write only: NO DATES",
        900,
    ),
    "mindmap": (
        "Make a mind map of the sources as an indented Markdown list: the first line is the central topic, "
        "then 3 to 6 main branches, each with 2 to 4 short sub-points (2 spaces of indentation per level). "
        "Use short phrases of at most 6 words. Write only the list.",
        700,
    ),
    "flashcards": (
        "Write 10 flashcards for studying the sources, one per line, in exactly this format:\n"
        "Q: <question or term> | A: <short answer> [source number]",
        900,
    ),
    "quiz": (
        "Write a multiple-choice quiz of 6 questions about the sources. Use exactly this format for each:\n"
        "Q: <question>\nA) <option>\nB) <option>\nC) <option>\nD) <option>\nANSWER: <letter>\n"
        "WHY: <one sentence explanation> [source number]\n",
        1300,
    ),
    "audio": (
        "Write the script of a lively podcast episode in which two hosts, ALEX and SAM, discuss the sources for "
        "about 4 minutes. ALEX introduces the topic; SAM asks questions and adds details; they explain the main "
        "ideas with concrete facts from the sources and end with a short recap. Use exactly this format, one "
        "line per turn, 16 to 24 turns:\nALEX: <what Alex says>\nSAM: <what Sam says>\n"
        "Do not use stage directions, sound effects or citations.",
        1500,
    ),
}


def notebook_material(docs: list[tuple[str, str, dict | None, list[dict]]], max_chars: int) -> tuple[str, list]:
    """docs: (document id, title, guide, passages). Numbered source blocks: guide summary + spread excerpts."""
    if not docs:
        return "", []
    per = max(600, max_chars // len(docs))
    blocks, sources_ = [], []
    for i, (doc_id, title, guide, passages) in enumerate(docs, 1):
        summary = (guide or {}).get("summary", "")
        excerpt = sample_passages(passages, max(300, per - len(summary) - len(title) - 20))
        blocks.append(f"[{i}] {title}\n" + (f"Summary: {summary}\n" if summary else "") + f"Excerpts:\n{excerpt}")
        sources_.append({"index": i, "document_id": doc_id, "title": title})
    return "\n\n".join(blocks)[: max_chars + 2000], sources_


def run_studio(llm, kind: str, material: str, instructions: str | None = None) -> str:
    prompt, max_tokens = STUDIO_PROMPTS[kind]
    system = "You turn the user's sources into study and briefing material. Use ONLY facts from the sources."
    user = f"Sources:\n\n{material}\n\nTask: {prompt}"
    if instructions:
        user += f"\nAdditional instructions from the user: {instructions.strip()[:500]}"
    return llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], max_new_tokens=max_tokens
    )


def _strip_cites(t: str) -> tuple[str, list[int]]:
    nums = [int(n) for n in re.findall(r"\[(\d{1,2})\]", t)]
    return re.sub(r"\s*\[\d{1,2}(?:\s*,\s*\d{1,2})*\]", "", t).strip(), nums


def parse_faq(text: str) -> list[dict]:
    out, q = [], None
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[#*\d.)\s]*)(?:\*\*)?Q(?:uestion)?\s*\d*\s*[:.]\**\s*(.+)$", line, re.I)
        if m:
            q = {"q": m.group(1).strip(" *"), "a": ""}
            out.append(q)
            continue
        m = re.match(r"^\s*(?:\*\*)?A(?:nswer)?\s*[:.]\**\s*(.+)$", line, re.I)
        if m and q is not None:
            q["a"] = m.group(1).strip()
        elif q is not None and q["a"] and line.strip():
            q["a"] += " " + line.strip()
    return [x for x in out if x["q"] and x["a"]]


def parse_timeline(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", line.strip())
        if "|" not in line:
            continue
        date, event = (x.strip(" *") for x in line.split("|", 1))
        event, nums = _strip_cites(event)
        if date and event and len(date) <= 40:
            out.append({"date": date, "event": event, "sources": nums})
    return out


def parse_flashcards(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        m = re.match(
            r"^\s*(?:[-*\d.)]+\s*)?(?:\*\*)?Q\s*[:.]\**\s*(.+?)\s*\|\s*(?:\*\*)?A\s*[:.]\**\s*(.+)$", line, re.I
        )
        if m:
            back, nums = _strip_cites(m.group(2))
            out.append({"front": m.group(1).strip(), "back": back, "sources": nums})
    return out


def parse_quiz(text: str) -> list[dict]:
    out, cur = [], None
    for line in text.splitlines():
        s = line.strip()
        if m := re.match(r"^(?:[#*\d.)\s]*)(?:\*\*)?Q(?:uestion)?\s*\d*\s*[:.]\**\s*(.+)$", s, re.I):
            cur = {"question": m.group(1).strip(" *"), "options": [], "answer": None, "why": "", "sources": []}
            out.append(cur)
        elif cur is not None and (m := re.match(r"^\(?([A-Da-d])[).:]\s*(.+)$", s)):
            cur["options"].append(m.group(2).strip())
        elif cur is not None and (m := re.match(r"^(?:\*\*)?(?:correct\s+)?answer\s*[:.]\**\s*\(?([A-Da-d])", s, re.I)):
            cur["answer"] = "abcd".index(m.group(1).lower())
        elif cur is not None and (m := re.match(r"^(?:\*\*)?(?:why|explanation)\s*[:.]\**\s*(.+)$", s, re.I)):
            cur["why"], cur["sources"] = _strip_cites(m.group(1))
    return [q for q in out if len(q["options"]) >= 2 and q["answer"] is not None and q["answer"] < len(q["options"])]


def parse_mindmap(text: str) -> dict:
    """Indented list -> {"label", "children": [...]} (the first line is the root)."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("```"):
            continue
        m = re.match(r"^(\s*)(?:[-*+]|\d+[.)]|#+)?\s*(.+)$", line.replace("\t", "  "))
        label = re.sub(r"\*\*|__|`", "", m.group(2)).strip(" :")
        label, _ = _strip_cites(label)
        if label:
            rows.append((len(m.group(1)), label[:80]))
    if not rows:
        return {"label": "Sources", "children": []}
    first = re.sub(r"^(?:central(?: topic| idea)?|root|main topic|topic)\s*:\s*", "", rows[0][1], flags=re.I)
    root = {"label": first or rows[0][1], "children": []}
    stack = [(-1, root)]
    for indent, label in rows[1:]:
        node = {"label": label, "children": []}
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        stack[-1][1]["children"].append(node)
        stack.append((indent, node))
    if len(root["children"]) == 1 and root["children"][0]["children"]:  # "- Topic" then everything below it
        only = root["children"][0]
        root = {"label": root["label"], "children": only["children"]} if only["label"] == root["label"] else root
    return root


def parse_script(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*\**\s*(ALEX|SAM|HOST\s*[12AB])\s*\**\s*:\s*\**\s*(.+)$", line.strip(), re.I)
        if m:
            who = m.group(1).upper()
            speaker = "ALEX" if who in ("ALEX", "HOST 1", "HOST1", "HOST A", "HOSTA") else "SAM"
            said = re.sub(r"\[[^\]]*\]|\([^)]*(?:laugh|music|pause)[^)]*\)|\*[^*]+\*", "", m.group(2)).strip()
            said = re.sub(r"\s+([.,!?;:])", r"\1", re.sub(r"\s{2,}", " ", said))
            if said:
                out.append({"speaker": speaker, "text": said})
    return out


def structure(kind: str, text: str) -> tuple[str, dict]:
    """(markdown shown / exported, structured data for the interactive views)."""
    text = (text or "").strip()
    if kind == "faq":
        items = parse_faq(text)
        md = "\n\n".join(f"### {x['q']}\n\n{x['a']}" for x in items) or text
        return md, {"items": items}
    if kind == "timeline":
        events = parse_timeline(text)
        if not events:
            return "The sources contain no dated events.", {"events": []}
        md = "\n".join(f"- **{e['date']}**: {e['event']}" + "".join(f" [{n}]" for n in e["sources"]) for e in events)
        return md, {"events": events}
    if kind == "flashcards":
        cards = parse_flashcards(text)
        return "\n".join(f"- **{c['front']}**: {c['back']}" for c in cards) or text, {"cards": cards}
    if kind == "quiz":
        qs = parse_quiz(text)
        md = "\n\n".join(
            f"**{i}. {q['question']}**\n\n"
            + "\n".join(f"{'ABCD'[j]}) {o}" for j, o in enumerate(q["options"]))
            + f"\n\n*Answer: {'ABCD'[q['answer']]}. {q['why']}*"
            for i, q in enumerate(qs, 1)
        )
        return md or text, {"questions": qs}
    if kind == "mindmap":
        tree = parse_mindmap(text)
        return text, {"tree": tree}
    if kind == "audio":
        turns = parse_script(text)
        return "\n\n".join(f"**{t['speaker'].title()}:** {t['text']}" for t in turns) or text, {"turns": turns}
    return text, {}
