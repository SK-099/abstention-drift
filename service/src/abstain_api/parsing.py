"""Turn uploaded PDF / HTML / Word / Markdown / text files into clean passages for indexing.

PARSER=docling uses IBM Docling (layout-aware PDF parsing, tables, reading order; its HybridChunker
keeps chunks inside section boundaries). PARSER=basic uses pypdf / BeautifulSoup / python-docx and a
paragraph-aware word-window chunker. PARSER=auto picks Docling when it is installed.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .settings import Settings

log = logging.getLogger(__name__)


def _docling_available() -> bool:
    try:
        import docling  # noqa: F401

        return True
    except ImportError:
        return False


def parse_and_chunk(path: str | Path, s: Settings) -> list[dict]:
    """Return [{"title", "section", "text"}] chunks.

    Contextual chunk headers: chunks never cross a section boundary, and each chunk's text starts with
    its section path (e.g. "Appendix A. Prompts" or "5. System Design › 5.4 Stage 3: answer generation"),
    so a chunk deep inside a section is still found by questions that name the section, and the LLM
    knows where the text comes from.
    """
    path = Path(path)
    use_docling = s.parser == "docling" or (
        s.parser == "auto" and _docling_available() and path.suffix.lower() in {".pdf", ".docx", ".html", ".htm"}
    )
    chunks = None
    if use_docling:
        try:
            chunks = _docling_chunks(path)
        except Exception:
            if s.parser == "docling":
                raise
            log.exception("Docling failed on %s; falling back to the basic parser", path.name)
    if chunks is None:
        title, text = basic_extract(path, page_markers=True)
        chunks = with_pages(
            [
                {"title": title, "section": section, "text": c}
                for section, body in split_sections(text, markdown=path.suffix.lower() != ".pdf")
                for c in chunk_text(body, s.chunk_words, s.chunk_overlap)
            ]
        )
    return [piece for c in chunks for piece in with_header(c, s.passage_chars, s.chunk_overlap)]


PAGE_MARK = re.compile(r"\u27e6p:(\d+)\u27e7")  # "⟦p:N⟧" inserted at the start of every PDF page


def with_pages(chunks: list[dict]) -> list[dict]:
    """Give every chunk the page its first word is on, and remove the page markers from the text.
    (Chunks are in reading order; a chunk without a marker lies on the page of the last marker seen.)"""
    current, out = None, []
    for c in chunks:
        marks = list(PAGE_MARK.finditer(c["text"]))
        page = current
        if marks:
            first = marks[0]
            page = int(first.group(1)) if not c["text"][: first.start()].strip() else int(first.group(1)) - 1
            current = int(marks[-1].group(1))
        text = re.sub(r"\s+", " ", PAGE_MARK.sub(" ", c["text"])).strip()
        if text:
            out.append({**c, "text": text, "page": page})
    return out


def with_header(chunk: dict, max_chars: int, overlap: int) -> list[dict]:
    """Prefix the section path; split so that header + text fits PASSAGE_CHARS (the generator and the
    self-check only read that many characters of a passage, so longer text would be retrievable but unseen)."""
    section = (chunk.get("section") or "").strip()
    header = f"{section}\n" if section else ""
    budget = max(200, max_chars - len(header)) if max_chars else 0
    return [{**chunk, "text": header + t} for t in fit_chars(chunk["text"], budget, overlap)]


# ---------------------------------------------------------------- section detection (basic parser)
_MD_HEAD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_NUM_HEAD = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3}|[A-Z])(\.?)\s+([A-Z(][^\n]{1,90})$")


_NAMED_HEADS = {"abstract", "acknowledgements", "acknowledgments", "references", "bibliography", "appendix"}


def _heading(line: str, top: list, markdown: bool = True) -> tuple[int, str] | None:
    """(level, text) if the line looks like a section heading, else None.

    Markdown '#' headings, or numbered ones ("5. System Design", "5.4 Stage 3: …", "A. Prompts").
    Rejects table-of-contents lines (trailing page number), list items and captions (sentence
    punctuation), and numbering that does not continue the current section (e.g. "3.1" under section 8)."""
    line = line.strip()
    m = _MD_HEAD.match(line) if markdown else None  # (in PDFs '#' starts shell comments, not headings)
    if m:
        return len(m.group(1)), m.group(2)
    if line.lower() in _NAMED_HEADS:
        return 1, line
    m = _NUM_HEAD.match(line)
    if not m:
        return None
    num, dot, rest = m.groups()
    if re.search(r"(\s\d+|[.,;:\-–])$", rest) or ". " in rest or len(rest.split()) > 12:
        return None
    parts = num.split(".")
    if len(parts) == 1:  # top level: "5. Title" / "A. Title" (needs the dot) and must move forward
        if not dot:
            return None
        prev = top[0]
        if num.isdigit():
            if prev is not None and (not str(prev).isdigit() or int(num) <= int(prev)):
                return None
        elif prev is not None and str(prev).isalpha() and num <= prev:
            return None
        top[0] = num
        label = f"Appendix {num}. {rest}" if num.isalpha() else f"{num}. {rest}"
        return 1, label
    if top[0] is None or parts[0] != str(top[0]):  # "5.4" only inside section 5
        return None
    return len(parts), f"{num} {rest}"


def split_sections(text: str, markdown: bool = True) -> list[tuple[str, str]]:
    """[(section path, body)] in document order; text before the first heading has an empty path."""
    out, stack, buf, top = [], [], [], [None]

    def flush():
        body = "\n".join(buf).strip()
        if body:
            out.append((" › ".join(h for _, h in stack), body))
        buf.clear()

    for line in text.replace("\r", "").split("\n"):
        h = _heading(line, top, markdown)
        if h is None:
            buf.append(line)
            continue
        flush()
        level, label = h
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, label))
    flush()
    return out


def fit_chars(text: str, max_chars: int, overlap: int = 40) -> list[str]:
    """Split text at word boundaries into pieces of at most max_chars characters (overlapping by `overlap` words)."""
    if not max_chars or len(text) <= max_chars:
        return [text]
    words, out, i = text.split(), [], 0
    while i < len(words):
        j, n = i, 0
        while j < len(words) and (n + len(words[j]) + (j > i)) <= max_chars:
            n += len(words[j]) + (j > i)
            j += 1
        j = max(j, i + 1)  # a single over-long token still makes progress
        out.append(" ".join(words[i:j]))
        if j >= len(words):
            break
        i = max(i + 1, j - min(overlap, (j - i) // 2))
    return out


def _docling_chunks(path: Path) -> list[dict]:
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    doc = DocumentConverter().convert(str(path)).document
    out = []
    for ch in HybridChunker().chunk(dl_doc=doc):
        meta = getattr(ch, "meta", None)
        heads = getattr(meta, "headings", None) or []
        page = None
        try:
            page = meta.doc_items[0].prov[0].page_no
        except Exception:
            pass
        text = ch.text.strip()
        if text:
            out.append({"title": path.stem, "section": " › ".join(heads), "text": text, "page": page})
    return out


def basic_extract(path: Path, page_markers: bool = False) -> tuple[str, str]:
    """(title, text). With page_markers, every PDF page starts with a "⟦p:N⟧" line (see with_pages)."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        r = PdfReader(str(path))
        title = r.metadata.title if r.metadata and r.metadata.title else path.stem
        pages = [(p.extract_text() or "") for p in r.pages]
        if page_markers:
            pages = [f"\u27e6p:{i}\u27e7\n{t}" for i, t in enumerate(pages, 1)]
        return title, "\n\n".join(pages)
    if ext in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            t.decompose()
        title = soup.title.get_text(strip=True) if soup.title else path.stem
        blocks = [
            ("#" * int(el.name[1]) + " " if el.name in {"h1", "h2", "h3", "h4"} else "") + el.get_text(" ", strip=True)
            for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "pre"])
        ]
        return title, "\n\n".join(b for b in blocks if b)
    if ext == ".docx":
        import docx

        d = docx.Document(str(path))
        title = d.core_properties.title or path.stem

        def md(p):
            st = (p.style.name or "") if p.style is not None else ""
            lvl = st.split()[-1] if st.startswith("Heading") else ""
            return ("#" * int(lvl) + " " if lvl.isdigit() else "") + p.text

        return title, "\n\n".join(md(p) for p in d.paragraphs if p.text.strip())
    return path.stem, path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, words: int = 220, overlap: int = 40) -> list[str]:
    """Paragraph-aware windows of about `words` words with `overlap` words of overlap."""
    text = re.sub(r"[ \t]+", " ", text.replace("\r", ""))
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], []
    for p in paras:
        w = p.split()
        while len(w) > words:  # split very long paragraphs
            head, w = w[:words], w[words - overlap :]
            if cur:
                chunks.append(" ".join(cur))
                cur = []
            chunks.append(" ".join(head))
        if len(cur) + len(w) > words and cur:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur += w
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) >= 5]
