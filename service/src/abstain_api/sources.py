"""Non-file sources: web pages, YouTube transcripts and pasted text, saved as files and then indexed like uploads.

Web pages are fetched by the worker, never from private network addresses (no SSRF: every redirect target is
checked too). YouTube sources use the video's transcript (captions), grouped into two-minute sections whose
headings are timestamps, so citations point to a moment in the video.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

from .settings import Settings

_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
MAX_REDIRECTS = 5


class SourceError(ValueError):
    """A source that cannot be added (shown to the user)."""


# ------------------------------------------------------------------------------------------- URLs
def youtube_id(url: str) -> str | None:
    u = urlparse(url.strip())
    host = (u.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    vid = None
    if host == "youtu.be":
        vid = u.path.lstrip("/").split("/")[0]
    elif host in {"youtube.com", "music.youtube.com", "youtube-nocookie.com"}:
        if u.path == "/watch":
            vid = (parse_qs(u.query).get("v") or [None])[0]
        else:
            m = re.match(r"^/(?:shorts|embed|live|v)/([^/?#]+)", u.path)
            vid = m.group(1) if m else None
    return vid if vid and _YT_ID.match(vid) else None


def check_public_url(url: str, s: Settings) -> str:
    """Only http(s) URLs whose host resolves to public addresses (unless ALLOW_PRIVATE_URLS)."""
    u = urlparse(url.strip())
    if u.scheme not in {"http", "https"} or not u.hostname:
        raise SourceError("enter a web address starting with http:// or https://")
    if s.allow_private_urls:
        return url.strip()
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except socket.gaierror as e:
        raise SourceError(f"could not find the website {u.hostname}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast:
            raise SourceError("this address points to a private network and cannot be added")
    return url.strip()


def safe_get(url: str, s: Settings, max_bytes: int, accept: str = "*/*"):
    """GET with every redirect re-checked against check_public_url. Returns (final_url, content_type, bytes)."""
    import httpx

    headers = {"User-Agent": "GroundedNotebook/1.0 (+source import)", "Accept": accept}
    with httpx.Client(timeout=s.fetch_timeout_s, follow_redirects=False, headers=headers) as c:
        for _ in range(MAX_REDIRECTS + 1):
            check_public_url(url, s)
            with c.stream("GET", url) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers.get("location", ""))
                    continue
                if r.status_code >= 400:
                    raise SourceError(f"the website answered with error {r.status_code}")
                data = b""
                for chunk in r.iter_bytes():
                    data += chunk
                    if len(data) > max_bytes:
                        raise SourceError(f"the page is larger than {max_bytes // (1024 * 1024)} MB")
                return str(r.url), r.headers.get("content-type", "").split(";")[0].strip().lower(), data
    raise SourceError("too many redirects")


def fetch_web_page(url: str, dest_stem: Path, s: Settings) -> tuple[Path, str]:
    """Download a page (HTML, PDF or plain text) next to the uploads. Returns (path, title)."""
    final, ctype, data = safe_get(url, s, s.max_upload_mb * 1024 * 1024, "text/html,application/pdf,text/plain")
    if ctype == "application/pdf" or data[:5] == b"%PDF-":
        path = dest_stem.with_suffix(".pdf")
        path.write_bytes(data)
        return path, Path(urlparse(final).path).name or urlparse(final).hostname or final
    if ctype in {"text/plain", "text/markdown"}:
        path = dest_stem.with_suffix(".txt")
        path.write_bytes(data)
        return path, Path(urlparse(final).path).name or final
    if ctype and "html" not in ctype and "xml" not in ctype:
        raise SourceError(f"unsupported content type {ctype}")
    from bs4 import BeautifulSoup

    html = data.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "") or urlparse(final).hostname or final
    text_len = len(soup.get_text(" ", strip=True))
    if text_len < 200:
        raise SourceError("the page has almost no text (it may need JavaScript or a login)")
    path = dest_stem.with_suffix(".html")
    path.write_text(html, encoding="utf-8")
    return path, title


# ------------------------------------------------------------------------------------------- YouTube
def _fetch_transcript(video_id: str) -> list[dict]:
    """[{text, start}] in English if available, else the first available language."""
    from youtube_transcript_api import YouTubeTranscriptApi

    langs = ["en", "en-US", "en-GB"]
    if hasattr(YouTubeTranscriptApi, "get_transcript"):  # youtube-transcript-api < 1.0
        try:
            return YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
        except Exception:
            tl = YouTubeTranscriptApi.list_transcripts(video_id)
            return next(iter(tl)).fetch()
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=langs)
    except Exception:
        fetched = next(iter(api.list(video_id))).fetch()
    return [{"text": sn.text, "start": sn.start} for sn in fetched]


def _clock(sec: float) -> str:
    sec = int(sec)
    h, m, s_ = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s_:02d}" if h else f"{m}:{s_:02d}"


def transcript_markdown(title: str, snippets: list[dict], section_s: int = 120) -> str:
    """Two-minute sections with timestamp headings ("## 4:00"), so citations point to a moment in the video."""
    out, cur, start = [f"# {title}"], [], None
    for sn in snippets:
        t = float(sn.get("start", 0.0))
        text = re.sub(r"\s+", " ", str(sn.get("text", ""))).strip()
        if not text or text.startswith("[") and text.endswith("]"):  # [Music], [Applause]
            continue
        if start is None:
            start = t
        if t - start >= section_s and cur:
            out.append(f"## {_clock(start)}\n\n" + " ".join(cur))
            cur, start = [], t
        cur.append(text)
    if cur:
        out.append(f"## {_clock(start or 0)}\n\n" + " ".join(cur))
    return "\n\n".join(out)


def youtube_title(video_id: str, s: Settings) -> str:
    import json

    try:
        _, _, data = safe_get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            s,
            100_000,
            "application/json",
        )
        return json.loads(data).get("title") or f"YouTube video {video_id}"
    except Exception:
        return f"YouTube video {video_id}"


def fetch_youtube(url: str, dest_stem: Path, s: Settings) -> tuple[Path, str]:
    vid = youtube_id(url)
    if not vid:
        raise SourceError("not a YouTube video link")
    try:
        snippets = _fetch_transcript(vid)
    except Exception as e:
        raise SourceError("this video has no transcript (captions) that can be read") from e
    title = youtube_title(vid, s)
    md = transcript_markdown(title, snippets)
    if len(md) < len(title) + 50:
        raise SourceError("the transcript of this video is empty")
    path = dest_stem.with_suffix(".md")
    path.write_text(md, encoding="utf-8")
    return path, title


def seconds_of(section: str | None) -> int | None:
    """ "4:00" / "1:02:03" (a transcript section heading, possibly "Title › 4:00") -> seconds."""
    section = (section or "").split("\u203a")[-1]
    if not section or not re.fullmatch(r"\d+(:\d{2}){1,2}", section.strip()):
        return None
    sec = 0
    for part in section.strip().split(":"):
        sec = sec * 60 + int(part)
    return sec
