"""Phase 5: sharing notebooks (viewer / editor / link) and discovering sources by web search."""

from conftest import auth
from sse import parse_sse

from abstain_api import search

TEXT = "The museum opens at nine in the morning on weekdays. Students pay four euros for a ticket with a card."


def _h(who):
    return auth(who, email=f"{who}@uni.example")


def test_sharing_roles(client):
    o, v, e, x = _h("olga"), _h("vik"), _h("eda"), _h("xen")
    assert client.post("/collections", json={"id": "shared1", "name": "Museum"}, headers=o).status_code == 201
    client.post("/collections/shared1/sources", json={"type": "text", "title": "Museum", "text": TEXT}, headers=o)
    # nobody else can see it yet
    assert client.get("/collections/shared1", headers=v).status_code == 404
    r = client.put("/collections/shared1/sharing/members", json={"principal": "VIK@uni.example"}, headers=o)
    assert r.status_code == 200 and r.json()["members"] == [{"principal": "vik@uni.example", "role": "viewer"}]
    client.put("/collections/shared1/sharing/members", json={"principal": "eda", "role": "editor"}, headers=o)
    # the viewer sees it in their list, reads and chats, but cannot change it
    mine = {c["id"]: c for c in client.get("/collections", headers=v).json()}
    assert mine["shared1"]["role"] == "viewer"
    assert client.get("/collections/shared1/documents", headers=v).status_code == 200
    ev = parse_sse(
        client.post(
            "/chat/stream",
            json={"message": "How much do students pay for a ticket?", "collection": "shared1"},
            headers=v,
        ).text
    )
    assert ev[-1][1]["decision"] == "answer"
    assert (
        client.post("/collections/shared1/sources", json={"type": "text", "text": TEXT}, headers=v).status_code == 403
    )
    assert client.post("/collections/shared1/studio", json={"kind": "faq"}, headers=v).status_code == 403
    assert client.put("/collections/shared1/sharing/members", json={"principal": "xen"}, headers=v).status_code == 403
    # the editor (shared by user id) can add sources and generate, but not share or delete
    assert client.get("/collections/shared1", headers=e).json()["role"] == "editor"
    r = client.post("/collections/shared1/sources", json={"type": "text", "title": "More", "text": TEXT}, headers=e)
    assert r.status_code == 202
    assert client.post("/collections/shared1/studio", json={"kind": "faq"}, headers=e).status_code == 202
    assert client.patch("/collections/shared1/sharing", json={"link_access": "viewer"}, headers=e).status_code == 403
    assert client.delete("/collections/shared1", headers=e).status_code == 403
    # notes and chats stay private to each person
    client.post("/collections/shared1/notes", json={"title": "Olga's note"}, headers=o)
    assert client.get("/collections/shared1/notes", headers=v).json() == []
    # anyone signed in with the link
    assert client.get("/collections/shared1", headers=x).status_code == 404
    r = client.patch("/collections/shared1/sharing", json={"link_access": "viewer"}, headers=o)
    assert r.json()["link_access"] == "viewer"
    got = client.get("/collections/shared1", headers=x).json()
    assert got["role"] == "viewer" and got["shared"]
    assert "shared1" not in {c["id"] for c in client.get("/collections", headers=x).json()}  # not listed
    # revoke
    client.patch("/collections/shared1/sharing", json={"link_access": "none"}, headers=o)
    client.delete("/collections/shared1/sharing/members/vik@uni.example", headers=o)
    assert client.get("/collections/shared1", headers=v).status_code == 404
    assert client.get("/collections/shared1", headers=x).status_code == 404
    assert client.get("/collections/shared1/sharing", headers=o).json()["members"] == [
        {"principal": "eda", "role": "editor"}
    ]
    assert (
        client.put(
            "/collections/shared1/sharing/members", json={"principal": "olga@uni.example"}, headers=o
        ).status_code
        == 422
    )
    assert client.delete("/collections/shared1", headers=o).status_code == 204


def test_discover_sources(client, monkeypatch):
    o = _h("olga")
    client.post("/collections", json={"id": "disc", "name": "Discover"}, headers=o)

    def fake_ddg(q, n):
        return [
            {"title": "Helgoland - Wikipedia", "url": "https://en.wikipedia.org/wiki/Helgoland", "snippet": "Island"},
            {"title": "dup", "url": "https://en.wikipedia.org/wiki/Helgoland", "snippet": ""},
            {"title": "Ferry video", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "snippet": "A trip"},
            {"title": "bad", "url": "javascript:alert(1)", "snippet": ""},
        ]

    monkeypatch.setattr(search, "_ddg", fake_ddg)
    r = client.post("/collections/disc/discover", json={"query": "Helgoland ferry"}, headers=o)
    assert r.status_code == 200, r.text
    assert [(h["url"], h["kind"]) for h in r.json()] == [
        ("https://en.wikipedia.org/wiki/Helgoland", "url"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
    ]
    assert client.post("/collections/disc/discover", json={"query": "x"}, headers=o).status_code == 422
    assert client.post("/collections/disc/discover", json={"query": "ferry"}, headers=_h("vik")).status_code == 404

    def boom(q, n):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(search, "_ddg", boom)
    r = client.post("/collections/disc/discover", json={"query": "Helgoland ferry"}, headers=o)
    assert r.status_code == 503 and "try again" in r.text
