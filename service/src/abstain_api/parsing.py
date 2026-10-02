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
    """Return [{"title", "text"}] chunks."""
    path = Path(path)
    use_docling = s.parser == "docling" or (
        s.parser == "auto" and _docling_available() and path.suffix.lower() in {".pdf", ".docx", ".html", ".htm"}
    )
    if use_docling:
        try:
            return _docling_chunks(path)
        except Exception:
            if s.parser == "docling":
                raise
            log.exception("Docling failed on %s; falling back to the basic parser", path.name)
    title, text = basic_extract(path)
    return [{"title": title, "text": c} for c in chunk_text(text, s.chunk_words, s.chunk_overlap)]


def _docling_chunks(path: Path) -> list[dict]:
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    doc = DocumentConverter().convert(str(path)).document
    out = []
    for ch in HybridChunker().chunk(dl_doc=doc):
        heads = getattr(getattr(ch, "meta", None), "headings", None) or []
        text = ch.text.strip()
        if text:
            out.append({"title": " / ".join(heads) or path.stem, "text": text})
    return out


def basic_extract(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        r = PdfReader(str(path))
        title = r.metadata.title if r.metadata and r.metadata.title else path.stem
        return title, "\n\n".join((p.extract_text() or "") for p in r.pages)
    if ext in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            t.decompose()
        title = soup.title.get_text(strip=True) if soup.title else path.stem
        blocks = [
            el.get_text(" ", strip=True) for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "pre"])
        ]
        return title, "\n\n".join(b for b in blocks if b)
    if ext == ".docx":
        import docx

        d = docx.Document(str(path))
        title = d.core_properties.title or path.stem
        return title, "\n\n".join(p.text for p in d.paragraphs if p.text.strip())
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
