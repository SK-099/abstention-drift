"""Discover sources: web search for a topic; the user picks results to add as web / YouTube sources."""

from __future__ import annotations

import httpx

from .settings import Settings
from .sources import youtube_id


class SearchUnavailable(RuntimeError):
    pass


def _ddg(q: str, n: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        try:
            from duckduckgo_search import DDGS
        except ImportError as e:
            raise SearchUnavailable("web search needs the 'ddgs' package") from e
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in DDGS().text(q, max_results=n)
    ]


def _tavily(q: str, n: int, s: Settings) -> list[dict]:
    r = httpx.post(
        "https://api.tavily.com/search", json={"api_key": s.search_api_key, "query": q, "max_results": n}, timeout=20
    )
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
        for x in r.json().get("results", [])
    ]


def _brave(q: str, n: int, s: Settings) -> list[dict]:
    r = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": q, "count": n},
        headers={"X-Subscription-Token": s.search_api_key or "", "Accept": "application/json"},
        timeout=20,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("description", "")}
        for x in r.json().get("web", {}).get("results", [])
    ]


def _searxng(q: str, n: int, s: Settings) -> list[dict]:
    r = httpx.get(f"{(s.searxng_url or '').rstrip('/')}/search", params={"q": q, "format": "json"}, timeout=20)
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
        for x in r.json().get("results", [])[:n]
    ]


def web_search(q: str, n: int, s: Settings) -> list[dict]:
    p = s.search_provider
    if p == "off":
        raise SearchUnavailable("web search is turned off (SEARCH_PROVIDER=off)")
    if p in ("tavily", "brave") and not s.search_api_key:
        raise SearchUnavailable(f"SEARCH_PROVIDER={p} needs SEARCH_API_KEY")
    if p == "searxng" and not s.searxng_url:
        raise SearchUnavailable("SEARCH_PROVIDER=searxng needs SEARXNG_URL")
    try:
        rows = {
            "duckduckgo": lambda: _ddg(q, n),
            "tavily": lambda: _tavily(q, n, s),
            "brave": lambda: _brave(q, n, s),
            "searxng": lambda: _searxng(q, n, s),
        }[p]()
    except SearchUnavailable:
        raise
    except Exception as e:
        raise SearchUnavailable(f"the web search failed ({e.__class__.__name__}); try again in a moment") from e
    out, seen = [], set()
    for r in rows:
        url = (r.get("url") or "").strip()
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "title": (r.get("title") or url)[:200],
                "url": url,
                "snippet": (r.get("snippet") or "")[:400],
                "kind": "youtube" if youtube_id(url) else "url",
            }
        )
    return out[:n]
