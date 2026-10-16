"""Notebook features: model picker, ticked sources, page numbers, source viewer, notes."""

import io

from conftest import auth
from sse import parse_sse

DOC_A = """# Ferries

The ferry to Helgoland leaves Cuxhaven at 10:30 every morning in summer.
"""
DOC_B = """# Trams

Tram line 9 runs every eight minutes from the main station to the zoo.
"""


def _nb(client, cid, who="nora"):
    r = client.post("/collections", json={"id": cid, "name": cid.title()}, headers=auth(who))
    assert r.status_code == 201, r.text


def _upload(client, cid, name, data, who="nora", ctype="text/markdown"):
    r = client.post(f"/collections/{cid}/documents", files={"file": (name, data, ctype)}, headers=auth(who))
    assert r.status_code == 202 and r.json()["status"] == "indexed", r.text
    return r.json()["id"]


def _chat(client, body, who="nora"):
    r = client.post("/chat/stream", json=body, headers=auth(who))
    assert r.status_code == 200, r.text
    return parse_sse(r.text)


def test_models_endpoint_lists_local_default(client):
    r = client.get("/models", headers=auth("nora"))
    assert r.status_code == 200
    body = r.json()
    assert body["default"].startswith("local:")
    ids = [m["id"] for m in body["models"]]
    assert body["default"] in ids  # discovery fails in tests, the calibrated model is still listed
    assert next(m for m in body["models"] if m["id"] == body["default"])["calibrated"]
    prov = {p["key"]: p for p in body["providers"]}
    assert prov["local"]["configured"] and not prov["openrouter"]["configured"]


def test_unconfigured_provider_is_rejected(client):
    _nb(client, "pick")
    _upload(client, "pick", "a.md", DOC_A.encode())
    r = client.post(
        "/chat/stream",
        json={"message": "When does the ferry leave?", "collection": "pick", "model": "openrouter:openai/gpt-4.1"},
        headers=auth("nora"),
    )
    assert r.status_code == 422 and "not configured" in r.text


def test_model_and_ticked_sources_reach_the_agent(client):
    _nb(client, "travel")
    a = _upload(client, "travel", "ferries.md", DOC_A.encode())
    b = _upload(client, "travel", "trams.md", DOC_B.encode())
    model = "local:llama3.2:3b"
    ev = _chat(
        client,
        {
            "message": "Tram line 9 runs from the main station to the zoo how often?",
            "collection": "travel",
            "model": model,
            "document_ids": [b],
        },
    )
    meta = ev[0][1]
    final = ev[-1][1]
    assert meta["model"] == model and final["model"] == model
    assert final["decision"] == "answer"
    assert {c["source_document"] for c in final["citations"]} == {b}
    # with only the ferry document ticked, the tram passage cannot be found
    ev = _chat(
        client,
        {
            "message": "Tram line 9 runs from the main station to the zoo how often?",
            "collection": "travel",
            "document_ids": [a],
        },
    )
    assert all(c["source_document"] == a for c in ev[-1][1]["citations"])
    # a document of another notebook is refused
    r = client.post(
        "/chat/stream",
        json={"message": "tram?", "collection": "travel", "document_ids": ["nope"]},
        headers=auth("nora"),
    )
    assert r.status_code == 422
    convs = client.get("/conversations?collection=travel", headers=auth("nora")).json()
    assert len(convs) == 2
    msgs = client.get(f"/conversations/{convs[-1]['id']}", headers=auth("nora")).json()["messages"]
    assert msgs[-1]["meta"]["model"] == model


def test_pdf_pages_viewer_and_file(client):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.drawString(72, 760, "1. Opening hours")
    c.drawString(72, 740, "The museum opens at nine in the morning on weekdays.")
    c.showPage()
    c.drawString(72, 760, "2. Tickets")
    c.drawString(72, 740, "Students pay four euros for a museum ticket with a valid card.")
    c.showPage()
    c.save()
    _nb(client, "museum")
    doc = _upload(client, "museum", "museum.pdf", buf.getvalue(), ctype="application/pdf")
    ps = client.get(f"/collections/museum/documents/{doc}/passages", headers=auth("nora")).json()
    assert [p["page"] for p in ps] == [1, 2]
    assert ps[1]["section"] == "2. Tickets" and ps[1]["text"].startswith("2. Tickets")
    ev = _chat(client, {"message": "How much do students pay for a museum ticket?", "collection": "museum"})
    cite = ev[-1][1]["citations"][0]
    assert cite["page"] == 2 and cite["section"] == "2. Tickets"
    f = client.get(f"/collections/museum/documents/{doc}/file", headers=auth("nora"))
    assert f.status_code == 200 and f.content[:4] == b"%PDF"
    assert client.get(f"/collections/museum/documents/{doc}/passages", headers=auth("other")).status_code == 404
    assert client.delete(f"/collections/museum/documents/{doc}", headers=auth("nora")).status_code == 204
    assert client.get(f"/collections/museum/documents/{doc}/passages", headers=auth("nora")).status_code == 404


def test_notes_crud(client):
    _nb(client, "notes")
    h = auth("nora")
    r = client.post(
        "/collections/notes/notes",
        json={"title": "Saved answer", "content": "Ferry at 10:30 [1]", "kind": "answer", "citations": [{"index": 1}]},
        headers=h,
    )
    assert r.status_code == 201
    nid = r.json()["id"]
    client.post("/collections/notes/notes", json={"title": "Second"}, headers=h)
    assert client.patch(f"/collections/notes/notes/{nid}", json={"pinned": True}, headers=h).json()["pinned"]
    notes = client.get("/collections/notes/notes", headers=h).json()
    assert [n["title"] for n in notes] == ["Saved answer", "Second"]  # pinned first
    assert client.get("/collections/notes/notes", headers=auth("eve")).status_code == 404
    assert client.delete(f"/collections/notes/notes/{nid}", headers=auth("eve")).status_code == 404
    assert client.delete(f"/collections/notes/notes/{nid}", headers=h).status_code == 204
    assert len(client.get("/collections/notes/notes", headers=h).json()) == 1


def test_rename_and_delete_notebook(client):
    _nb(client, "tmpnb")
    _upload(client, "tmpnb", "a.md", DOC_A.encode())
    _chat(client, {"message": "When does the ferry leave?", "collection": "tmpnb"})
    r = client.patch("/collections/tmpnb", json={"name": "Renamed"}, headers=auth("nora"))
    assert r.json()["name"] == "Renamed"
    assert client.delete("/collections/cloud", headers=auth("admin")).status_code == 403
    assert client.delete("/collections/tmpnb", headers=auth("nora")).status_code == 204
    assert all(c["id"] != "tmpnb" for c in client.get("/collections", headers=auth("nora")).json())


def test_routing_chat_builds_provider_clients():
    from abstain_api.llm import RoutingChat
    from abstain_api.model_registry import REQUEST_MODEL, get_registry
    from abstain_api.settings import Settings

    s = Settings(openrouter_api_key="k1", openai_api_key="k2")
    rc = RoutingChat(s)
    c = rc.client("openrouter:anthropic/claude-sonnet-4.5")
    assert c.base_url == "https://openrouter.ai/api/v1" and c.api_key == "k1" and c.max_tokens_field == "max_tokens"
    o = rc.client("openai:o4-mini")
    assert o.reasoning and o.max_tokens_field == "max_completion_tokens"
    assert not rc.client("openai:gpt-4.1-mini").reasoning
    tok = REQUEST_MODEL.set("openai:gpt-4.1-mini")
    try:
        assert rc.client().model == "gpt-4.1-mini"
    finally:
        REQUEST_MODEL.reset(tok)
    assert rc.client().provider == "local"
    reg = get_registry(s)
    assert reg.resolve("openrouter:x/y") == "openrouter:x/y"
    s2 = Settings(openrouter_api_key="k", model_allowlist=["openrouter:anthropic/*"])
    import pytest

    with pytest.raises(ValueError):
        get_registry(s2).resolve("openrouter:openai/gpt-4.1")
    with pytest.raises(ValueError):
        get_registry(s2).resolve("gemini:gemini-2.5-flash")


def test_registry_parses_provider_lists(monkeypatch):
    from abstain_api import model_registry as mr
    from abstain_api.settings import Settings

    replies = {
        "http://localhost:11434/api/tags": {
            "models": [{"name": "qwen2.5:3b"}, {"name": "llama3.2:3b"}, {"name": "nomic-embed-text"}]
        },
        "https://openrouter.ai/api/v1/models": {
            "data": [
                {"id": "anthropic/claude-sonnet-4.5", "name": "Anthropic: Claude Sonnet 4.5", "context_length": 200000},
                {"id": "openai/gpt-4.1-mini", "name": "OpenAI: GPT-4.1 Mini"},
                {"id": "black-forest/flux", "architecture": {"output_modalities": ["image"]}},
            ]
        },
        "https://api.anthropic.com/v1/models?limit=100": {"data": [{"id": "claude-x", "display_name": "Claude X"}]},
        "https://generativelanguage.googleapis.com/v1beta/openai/models": {
            "data": [{"id": "models/gemini-2.5-flash"}, {"id": "models/text-embedding-004"}]
        },
    }
    monkeypatch.setattr(mr, "_get", lambda url, headers, timeout=15.0: replies[url])
    s = Settings(
        openrouter_api_key="a", anthropic_api_key="b", gemini_api_key="c", llm_base_url="http://localhost:11434/v1"
    )
    ms = {m.id: m for m in mr.ModelRegistry(s).list()}
    assert list(ms)[0] == "local:qwen2.5:3b" and ms["local:qwen2.5:3b"].calibrated
    assert "local:llama3.2:3b" in ms and "local:nomic-embed-text" not in ms
    assert ms["openrouter:anthropic/claude-sonnet-4.5"].featured
    assert ms["openrouter:anthropic/claude-sonnet-4.5"].context_length == 200000
    assert "openrouter:black-forest/flux" not in ms
    assert ms["anthropic:claude-x"].name == "Claude X"
    assert "gemini:gemini-2.5-flash" in ms and "gemini:text-embedding-004" not in ms


def test_reasoning_models_think_blocks_are_hidden():
    from abstain_api.llm import strip_thinking, visible_text
    from abstain_api.model_registry import is_reasoning

    assert strip_thinking("<think>step 1\nstep 2</think>\n\nParis is the capital.") == "Paris is the capital."
    assert strip_thinking("<think>ran out of tokens") == ""
    assert visible_text("<thi") == "" and visible_text("<think>so") == ""
    assert visible_text("<think>so</think> Paris") == "Paris"
    assert visible_text("Paris") == "Paris"
    assert is_reasoning("deepseek-r1:1.5b") and is_reasoning("qwen3:1.7b") and is_reasoning("openai/o4-mini")
    assert not is_reasoning("qwen2.5:3b") and not is_reasoning("llama3.2:3b") and not is_reasoning("gemma3:4b")


def test_warm_model_and_guide_with_the_chosen_model(client, monkeypatch):
    from abstain_api import indexing

    r = client.post("/models/warm", json={"model": "openrouter:openai/gpt-4.1"}, headers=auth("nora"))
    assert r.status_code == 422
    r = client.post("/models/warm", json={"model": "local:qwen2.5:3b"}, headers=auth("nora"))
    assert r.status_code == 200 and r.json()["model"] == "local:qwen2.5:3b"  # no Ollama in tests: loaded False
    seen = []
    real = indexing.make_guide

    def spy(session, doc_id, s, store=None, model=None):
        seen.append(model)
        return real(session, doc_id, s, store, model)

    monkeypatch.setattr(indexing, "make_guide", spy)
    _nb(client, "guidemodel")
    r = client.post(
        "/collections/guidemodel/documents",
        files={"file": ("a.md", DOC_A.encode(), "text/markdown")},
        data={"model": "local:gemma3:4b"},
        headers=auth("nora"),
    )
    assert r.json()["status"] == "indexed"
    r = client.post(
        "/collections/guidemodel/sources", json={"type": "text", "text": DOC_B, "model": "bogus"}, headers=auth("nora")
    )
    doc = r.json()["id"]
    client.post(f"/collections/guidemodel/documents/{doc}/guide?model=local:llama3.2:3b", headers=auth("nora"))
    assert seen == ["local:gemma3:4b", None, "local:llama3.2:3b"]  # an invalid model falls back to the default
