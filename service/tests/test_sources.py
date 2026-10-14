"""Phase 2: web pages, YouTube transcripts and pasted text as sources; source guides; notebook-wide questions."""

import pytest
from conftest import auth
from sse import parse_sse

from abstain_api import sources
from abstain_api.settings import get_settings

FERRY = (
    "The ferry to Helgoland leaves Cuxhaven at 10:30 every morning in summer. "
    "The crossing takes about two hours and fifteen minutes. Tickets cost 55 euros for adults."
)
TRAM = (
    "Tram line 9 runs every eight minutes from the main station to the zoo. "
    "On Sundays it runs every fifteen minutes. A single ticket costs 3 euros."
)
PAGE = """<html><head><title>Harbour guide</title><script>alert(1)</script></head><body>
<h1>Harbour guide</h1><p>The lighthouse on the north pier was built in 1891 and is open to visitors on
Saturdays from April to October. Guided tours start at the harbour office every hour.</p>
<h2>Parking</h2><p>The car park next to the ferry terminal has 400 spaces and costs 8 euros per day.</p>
</body></html>"""


def _nb(client, cid, who="nora"):
    assert client.post("/collections", json={"id": cid, "name": cid.title()}, headers=auth(who)).status_code == 201


def _src(client, cid, body, who="nora"):
    return client.post(f"/collections/{cid}/sources", json=body, headers=auth(who))


def test_pasted_text_gets_indexed_with_a_guide(client):
    _nb(client, "paste")
    r = _src(client, "paste", {"type": "text", "title": "Ferry notes", "text": FERRY})
    assert r.status_code == 202, r.text
    d = r.json()
    assert d["status"] == "indexed" and d["source_type"] == "text" and d["filename"] == "Ferry notes"
    docs = client.get("/collections/paste/documents", headers=auth("nora")).json()
    g = docs[0]["guide"]
    assert docs[0]["guide_status"] == "ready"
    assert g["summary"].startswith("The document says") and g["topics"][:2] == ["travel", "timetables"]
    assert len(g["questions"]) == 3 and all(q.endswith("?") for q in g["questions"])
    # regenerate on request
    r = client.post(f"/collections/paste/documents/{d['id']}/guide", headers=auth("nora"))
    assert r.status_code == 202 and r.json()["guide_status"] == "ready"
    # validation: too short, other user's notebook
    assert _src(client, "paste", {"type": "text", "text": "short"}).status_code == 422
    assert _src(client, "paste", {"type": "text", "text": FERRY}, who="eve").status_code in (403, 404)


def test_web_page_source(client, monkeypatch):
    monkeypatch.setattr(sources, "check_public_url", lambda url, s: url)
    seen = []

    def fake_get(url, s, max_bytes, accept="*/*"):
        seen.append(url)
        return "https://example.org/harbour", "text/html", PAGE.encode()

    monkeypatch.setattr(sources, "safe_get", fake_get)
    _nb(client, "web")
    r = _src(client, "web", {"type": "url", "url": "https://example.org/harbour"})
    assert r.status_code == 202, r.text
    d = r.json()
    assert d["status"] == "indexed", d["error"]
    assert d["filename"] == "Harbour guide" and d["source_url"] == "https://example.org/harbour"
    ps = client.get(f"/collections/web/documents/{d['id']}/passages", headers=auth("nora")).json()
    assert any("lighthouse" in p["text"] for p in ps) and not any("alert(1)" in p["text"] for p in ps)
    ev = parse_sse(
        client.post(
            "/chat/stream",
            json={"message": "How much does the car park cost per day?", "collection": "web"},
            headers=auth("nora"),
        ).text
    )
    assert ev[-1][1]["decision"] == "answer" and ev[-1][1]["citations"][0]["source_document"] == d["id"]
    # the stored page is downloaded, never rendered on the API origin
    f = client.get(f"/collections/web/documents/{d['id']}/file", headers=auth("nora"))
    assert f.headers["content-security-policy"] == "sandbox" and "attachment" in f.headers["content-disposition"]


def test_private_addresses_are_refused(client):
    _nb(client, "ssrf")
    for url in ["http://127.0.0.1:8000/admin", "http://localhost/", "http://169.254.169.254/latest", "ftp://x.org/a"]:
        r = _src(client, "ssrf", {"type": "url", "url": url})
        assert r.status_code == 422, (url, r.text)
    s = get_settings()
    with pytest.raises(sources.SourceError):
        sources.check_public_url("http://10.0.0.5/", s)


def test_youtube_source_has_timestamped_sections(client, monkeypatch):
    snippets = [
        {"text": f"Part {i}: the speaker explains topic number {i} in detail.", "start": i * 40.0} for i in range(9)
    ]
    snippets.insert(1, {"text": "[Music]", "start": 20.0})
    monkeypatch.setattr(sources, "_fetch_transcript", lambda vid: snippets)
    monkeypatch.setattr(sources, "youtube_title", lambda vid, s: "How ferries work")
    _nb(client, "yt")
    r = _src(client, "yt", {"type": "url", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10"})
    d = r.json()
    assert r.status_code == 202 and d["status"] == "indexed", d
    assert d["source_type"] == "youtube" and d["filename"] == "How ferries work"
    ps = client.get(f"/collections/yt/documents/{d['id']}/passages", headers=auth("nora")).json()
    secs = [sources.seconds_of(p["section"]) for p in ps]
    assert {0, 120, 240} <= set(secs)
    assert not any("[Music]" in p["text"] for p in ps)
    bad = _src(client, "yt", {"type": "youtube", "url": "https://example.org/watch?v=nope"})
    assert bad.status_code == 422


def test_youtube_links_and_timestamps():
    ok = {
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=42": "dQw4w9WgXcQ",
        "https://m.youtube.com/shorts/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ": "dQw4w9WgXcQ",
    }
    for url, vid in ok.items():
        assert sources.youtube_id(url) == vid
    assert sources.youtube_id("https://vimeo.com/123") is None
    assert sources.seconds_of("4:00") == 240 and sources.seconds_of("1:02:03") == 3723
    assert sources.seconds_of("Introduction") is None and sources.seconds_of("How ferries work \u203a 2:00") == 120
    md = sources.transcript_markdown("T", [{"text": "a b", "start": 0}, {"text": "c d", "start": 130}])
    assert md == "# T\n\n## 0:00\n\na b\n\n## 2:10\n\nc d"


def test_notebook_wide_question_uses_the_source_guides(client):
    _nb(client, "overview")
    a = _src(client, "overview", {"type": "text", "title": "Ferries", "text": FERRY}).json()["id"]
    b = _src(client, "overview", {"type": "text", "title": "Trams", "text": TRAM}).json()["id"]
    ev = parse_sse(
        client.post(
            "/chat/stream", json={"message": "Summarise my sources", "collection": "overview"}, headers=auth("nora")
        ).text
    )
    steps = [e[1]["node"] for e in ev if e[0] == "step"]
    final = ev[-1][1]
    assert steps == ["overview"] and final["decision"] == "answer"
    assert [c["source_document"] for c in final["citations"]] == [a, b]
    assert final["citations"][0]["section"] == "Source guide" and final["trace"][0].startswith("overview")
    # only the ticked sources
    ev = parse_sse(
        client.post(
            "/chat/stream",
            json={
                "message": "What are the main topics of these documents?",
                "collection": "overview",
                "document_ids": [b],
            },
            headers=auth("nora"),
        ).text
    )
    assert [c["source_document"] for c in ev[-1][1]["citations"]] == [b]
    # a specific question still goes through retrieval and the abstention agent
    ev = parse_sse(
        client.post(
            "/chat/stream",
            json={"message": "How often does tram line 9 run?", "collection": "overview"},
            headers=auth("nora"),
        ).text
    )
    assert "retrieve" in [e[1]["node"] for e in ev if e[0] == "step"]


def test_overview_detection():
    from abstain_api.generation import is_overview_question, parse_guide

    for q in [
        "Summarize the documents",
        "Give me an overview",
        "What are the key takeaways?",
        "What are these sources about?",
        "main themes please",
    ]:
        assert is_overview_question(q), q
    for q in ["When does the ferry leave?", "How much is a tram ticket on Sundays?"]:
        assert not is_overview_question(q), q
    g = parse_guide(
        "**SUMMARY:** A text.\nIt is short.\nTOPICS: one; two, Three\nQUESTIONS:\n1. Why is it short\n2) How long is it?"
    )
    assert g["summary"] == "A text.\nIt is short." and g["topics"] == ["one", "two", "Three"]
    assert g["questions"] == ["Why is it short?", "How long is it?"]
    assert parse_guide("no structure at all")["summary"] == "no structure at all"
