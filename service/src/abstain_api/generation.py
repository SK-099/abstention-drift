"""LLM generation over whole sources (not single questions): source guides and notebook overviews.

Prompts ask for plain labelled sections rather than JSON, which small local models (3B) follow far more
reliably; the parsers are lenient and every function has a fallback, so a weak model degrades the output
instead of failing the request.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from .parsing import is_back_matter
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
        from .llm import OpenAICompatChat

        c = llm.client(model_id or generation_model_id(s))
        # one long attempt instead of 3 x LLM_TIMEOUT_S: a 3B model on a small GPU needs minutes for these
        # prompts, and Ollama answers one request at a time (a chat question may be ahead in the queue)
        long = s.model_copy(
            update={"llm_timeout_s": s.generation_timeout_s, "reasoning_timeout_s": s.generation_timeout_s}
        )
        return OpenAICompatChat(
            long,
            retries=2,
            model=c.model,
            base_url=c.base_url,
            api_key=c.api_key,
            provider=c.provider,
            max_tokens_field=c.max_tokens_field,
            reasoning=c.reasoning,
            long_prompts=True,
        )
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
    body = [p for p in passages if not is_back_matter(p)]
    passages = body or passages
    if not passages:
        return ""
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
    r"(?:sources|documents) (?:say|cover)|compare (?:my|the|these|all) (?:sources|documents)|"
    # the source as a whole: "what does this video explain?", "what is the paper about?"
    r"what (?:is|does|do) (?:this|the|these) (?:video|videos|paper|papers|article|talk|lecture|document|book|"
    r"chapter|podcast)(?: \w+ly)? (?:about|explain|cover|discuss|describe|teach|show|present)\s*(?:[?.!,]|$)|"
    r"what (?:does|do) the (?:speaker|speakers|author|authors|presenter|lecturer) (?:explain|talk about|discuss|"
    r"cover|describe) in (?:this|the) (?:video|talk|lecture|podcast|paper))",
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
        "about 2 minutes. ALEX introduces the topic; SAM asks questions and adds details; they explain the main "
        "ideas with concrete facts from the sources and end with a short recap. Use exactly this format, one "
        "line per turn, 10 to 14 turns of one to three sentences:\nALEX: <what Alex says>\nSAM: <what Sam says>\n"
        "Do not use stage directions, sound effects or citations.",
        900,
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


_FILLER = set(
    """about above after again against also although always among another anything around because before being
    below between both cannot could different during each either enough especially every example exactly first
    following further great having however important including interesting into itself just known later least
    less like little made make many maybe might more most much must never next often other others over really
    right same should since some something still such than thank thanks that their them then there these they
    thing things think this those though three through today together under until using very want well were what
    when where whether which while will with within without would years your yeah okay absolutely welcome episode
    joining listen listeners podcast show talk talking discuss going sounds agree fascinating amazing alex sam
    sources source document documents paper study research question questions answer answers point points""".split()
)


def _stems(text: str) -> list[str]:
    return [w[:6] for w in re.findall(r"[a-z]{5,}", (text or "").lower()) if w not in _FILLER]


def on_topic(output: str, material: str) -> bool:
    """False when a small model wrote about something else entirely (its context lost the sources): it uses
    almost none of the material's most frequent words and few of its words occur in the material at all."""
    out = _stems(output)
    if len(out) < 15:
        return True  # too short to judge
    mat = _stems(material)
    have = set(mat)
    top = [w for w, _ in Counter(mat).most_common(25)]
    mentioned = sum(1 for w in top if w in set(out))
    overlap = sum(1 for w in out if w in have) / len(out)
    return mentioned >= 3 or overlap >= 0.45


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


_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?"
    r"|nov(?:ember)?|dec(?:ember)?"
)
_DATE_WORD = re.compile(
    rf"\d|\b(?:{_MONTHS}|spring|summer|autumn|fall|winter|century|decade|era|age|period|bc|ad|today|now)\b", re.I
)


def parse_timeline(text: str) -> list[dict]:
    """ "date | event [n]" lines. A row only counts when its date looks like one (a number, a month, a season,
    a century...): small models answer papers without dates with "NO DATES | ..." rows, numbered lists or a
    copied "<date or period>" template, which must give "no dated events" rather than fake events."""
    out, seen = [], set()
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", line.strip())
        if "|" not in line:
            continue
        date, event = line.split("|", 1)
        date = date.strip(" *<>[]")
        event, nums = _strip_cites(event.strip(" *<>"))
        bad = re.search(r"no dates?|n/?a\b|unknown|not (?:given|stated|specified)|date or period", date, re.I)
        if (
            date
            and event
            and len(date) <= 40
            and not bad
            and _DATE_WORD.search(date)
            and not re.fullmatch(r"\d{1,2}", date)  # a row number, not a date
            and not re.search(r"what happened|no dates?", event, re.I)
            and (date.lower(), event.lower()) not in seen
        ):
            seen.add((date.lower(), event.lower()))
            out.append({"date": date, "event": event, "sources": nums})
    return out


_FRONT = r"(?:Q(?:uestion)?|Front|Term|Concept)\s*\d*\s*[:.]"
_BACK = r"(?:A(?:nswer)?|Back|Definition|Meaning)\s*[:.]"


def parse_flashcards(text: str) -> list[dict]:
    """Flashcards in the asked format "Q: front | A: back", or in the shapes small models drift into:
    Q and A on separate lines, "Front: ... Back: ...", "front | back", or a bulleted "**Term**: definition"."""
    strict, loose, pending = [], [], None

    def card(front: str, back: str) -> dict:
        front = re.sub(r"^(?:\d+\s*[:.)]\s*|Q\s*[:.]\s*)", "", front.strip(" *"), flags=re.I)
        back = re.sub(r"^(?:A(?:nswer)?\s*[:.]\s*)", "", back.strip(" *"), flags=re.I)
        back, nums = _strip_cites(back)
        return {"front": front.strip(" *"), "back": back, "sources": nums}

    def placeholder(c: dict) -> bool:  # the format template copied back: "[Term or question]: [Answer ...]"
        return any(re.fullmatch(r"[\[<].*[\]>]", c[k].strip()) for k in ("front", "back")) or not c["back"]

    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        s = re.sub(r"^(?:[-*\u2022]|\d+[.)])\s+", "", raw).replace("**", "").replace("__", "").strip()
        bullet = s != raw.replace("**", "").replace("__", "").strip()
        s = re.sub(r"^\d{1,2}\s*:\s*", "", s)  # "1: What is ...?"
        if m := re.match(rf"^{_FRONT}\s*(.+?)\s*(?:\||\s[-\u2013]\s|\t)\s*{_BACK}\s*(.+)$", s, re.I):
            strict.append(card(m.group(1), m.group(2)))
            pending = None
        elif m := re.match(rf"^{_FRONT}\s*(.+)$", s, re.I):
            pending = m.group(1)
        elif (m := re.match(rf"^{_BACK}\s*(.+)$", s, re.I)) and pending:
            strict.append(card(pending, m.group(1)))
            pending = None
        elif s.startswith("|"):  # a Markdown table: | front | back |
            cells = [c.strip() for c in s.strip("|").split("|")]
            header = cells[0].lower() in ("front", "question", "term", "q", "concept", "#")
            if len(cells) >= 2 and cells[0] and cells[1] and not header and not set("".join(cells)) <= set("-: "):
                loose.append(card(cells[0], cells[1]))
        elif "|" in s:
            a, b = (x.strip() for x in s.split("|", 1))
            if 2 <= len(a) <= 200 and len(b) >= 2 and not set(b) <= set("-: |"):
                loose.append(card(a, b))
        elif bullet and (m := re.match(r"^([^:]{2,120}?)\s*:\s+(.{2,})$", s)):
            loose.append(card(m.group(1), m.group(2)))
    cards = strict if len(strict) >= 3 or len(strict) >= len(loose) else strict + loose
    seen, out = set(), []
    for c in cards:
        key = c["front"].lower()
        if c["front"] and c["back"] and key not in seen and not placeholder(c):
            seen.add(key)
            out.append(c)
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


_SPEAKER = re.compile(
    r"^\s*(?:[-*\u2022>]\s*)?[*_]*\s*\[?([A-Za-z][A-Za-z0-9 .'-]{0,24}?)\]?\s*(?:\([^)]{0,40}\))?\s*[*_]*\s*[:\-\u2013\u2014]\s*[*_]*\s*(.+)$"
)
_FIRST = {"ALEX", "HOST 1", "HOST1", "HOST A", "HOSTA", "HOST", "SPEAKER 1", "SPEAKER1", "SPEAKER A", "PERSON 1", "A"}
_SECOND = {"SAM", "HOST 2", "HOST2", "HOST B", "HOSTB", "GUEST", "SPEAKER 2", "SPEAKER2", "SPEAKER B", "PERSON 2", "B"}
_NOT_NAMES = {"NOTE", "TITLE", "EPISODE", "SOURCES", "SOURCE", "RECAP", "SUMMARY", "INTRO", "OUTRO", "TOPIC", "SCENE"}


def _clean_turn(said: str) -> str:
    said = re.sub(r"\[[^\]]*\]|\([^)]*(?:laugh|music|pause|sigh|smil|excit|chuckl)[^)]*\)|\*[^*]+\*", "", said)
    said = said.replace("**", "").replace("__", "").strip(' "\u201c\u201d')
    return re.sub(r"\s+([.,!?;:])", r"\1", re.sub(r"\s{2,}", " ", said)).strip()


def parse_script(text: str) -> list[dict]:
    """Two-host script -> [{"speaker": "ALEX"|"SAM", "text": ...}].

    Small models rarely keep to "ALEX: ..." exactly: "**Alex (excited):**", "Host 1 -", "Speaker 2:" or two
    names of their own are all read. Without any speaker labels, the paragraphs are shared out alternately so
    the episode can still be spoken."""
    rows = []
    for line in text.splitlines():
        m = _SPEAKER.match(line.strip())
        if not m:
            continue
        name = re.sub(r"\s+", " ", m.group(1)).strip(" .'-").upper()
        said = _clean_turn(m.group(2))
        if not said or name in _NOT_NAMES or len(name.split()) > 3 or len(said) < 2:
            continue
        rows.append((name, said))
    # the two most frequent names are the hosts (a stray "Note:" line is ignored)
    counts: dict[str, int] = {}
    for name, _ in rows:
        counts[name] = counts.get(name, 0) + 1
    hosts = [n for n, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c >= 1][:2]
    out = []
    if len(hosts) == 2 or (len(hosts) == 1 and hosts[0] in _FIRST | _SECOND):
        first = next((h for h in hosts if h in _FIRST), None) or next((h for h in hosts if h not in _SECOND), hosts[0])
        for name, said in rows:
            if name in hosts:
                out.append({"speaker": "ALEX" if name == first else "SAM", "text": said})
    if len(out) >= 2:
        return out
    # no usable labels: alternate the paragraphs (or pairs of sentences) between the two voices
    body = re.sub(r"(?m)^\s*#+.*$", "", text)
    paras = [_clean_turn(p) for p in re.split(r"\n\s*\n", body)]
    paras = [p for p in paras if len(p.split()) >= 4]
    if len(paras) < 3:
        sents = re.split(r"(?<=[.!?])\s+", " ".join(paras))
        paras = [" ".join(sents[i : i + 2]) for i in range(0, len(sents), 2)]
        paras = [p for p in paras if len(p.split()) >= 4]
    if len(paras) < 2:
        return []
    return [{"speaker": "ALEX" if i % 2 == 0 else "SAM", "text": p} for i, p in enumerate(paras[:16])]


def tidy_script(turns: list[dict], max_turns: int = 16) -> list[dict]:
    """Small models ignore "10 to 14 turns": they repeat themselves ("That's a great point. What do you
    think...") until the token budget cuts the last line off. Keep the episode short and clean: drop turns that
    nearly repeat an earlier one, cap the length, never end on a cut-off sentence or an unanswered question."""

    def words(s: str) -> set[str]:
        return set(re.findall(r"[a-z]{3,}", s.lower()))

    out: list[dict] = []
    for turn in turns:
        w = words(turn["text"])
        if w and any(len(w & words(o["text"])) / len(w | words(o["text"])) >= 0.6 for o in out):
            continue
        out.append(turn)
    if out and not re.search(r"[.!?\"'\u201d)]\s*$", out[-1]["text"]):  # cut off by the token budget
        out.pop()
    out = out[:max_turns]
    while len(out) > 2 and out[-1]["text"].rstrip().endswith("?"):
        out.pop()
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
        turns = tidy_script(parse_script(text))
        return "\n\n".join(f"**{t['speaker'].title()}:** {t['text']}" for t in turns) or text, {"turns": turns}
    return text, {}
