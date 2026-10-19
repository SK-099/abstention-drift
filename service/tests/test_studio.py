"""Phase 3 + 4: Studio outputs (briefing, study guide, FAQ, timeline, mind map, flashcards, quiz, audio)."""

import io
import wave

from conftest import auth

from abstain_api import audio, generation
from abstain_api.settings import get_settings

FERRY = "The ferry to Helgoland leaves Cuxhaven at 10:30 every morning. Tickets cost 55 euros for adults."
TRAM = "Tram line 9 runs every eight minutes from the main station to the zoo. Tickets cost 3 euros."


def _setup(client, cid):
    h = auth("nora")
    assert client.post("/collections", json={"id": cid, "name": "Travel"}, headers=h).status_code == 201
    a = client.post(f"/collections/{cid}/sources", json={"type": "text", "title": "Ferry", "text": FERRY}, headers=h)
    b = client.post(f"/collections/{cid}/sources", json={"type": "text", "title": "Tram", "text": TRAM}, headers=h)
    return a.json()["id"], b.json()["id"]


def _make(client, cid, kind, **kw):
    r = client.post(f"/collections/{cid}/studio", json={"kind": kind, **kw}, headers=auth("nora"))
    assert r.status_code == 202, r.text
    return client.get(f"/collections/{cid}/studio/{r.json()['id']}", headers=auth("nora")).json()


def test_every_studio_kind(client):
    a, b = _setup(client, "studio1")
    out = {
        k: _make(client, "studio1", k)
        for k in ["briefing", "study_guide", "faq", "timeline", "mindmap", "flashcards", "quiz"]
    }
    for k, art in out.items():
        assert art["status"] == "ready", (k, art["error"])
        assert art["title"].endswith("· Travel") and art["data"]["sources"][0]["title"] == "Ferry"
    assert out["briefing"]["content"].startswith("## Executive summary")
    assert [x["q"] for x in out["faq"]["data"]["items"]] == ["When does the ferry leave?", "How long is the trip?"]
    ev = out["timeline"]["data"]["events"]
    assert [(e["date"], e["event"], e["sources"]) for e in ev] == [
        ("1891", "The lighthouse was built", [1]),
        ("2020", "The terminal opened", [2]),
    ]
    tree = out["mindmap"]["data"]["tree"]
    assert tree["label"] == "Travel" and [c["label"] for c in tree["children"]] == ["Ferries", "Trams"]
    assert tree["children"][0]["children"][1]["label"] == "55 euros"
    cards = out["flashcards"]["data"]["cards"]
    assert cards == [
        {"front": "Ferry departure", "back": "10:30", "sources": [1]},
        {"front": "Tram line", "back": "9", "sources": [2]},
    ]
    qs = out["quiz"]["data"]["questions"]
    assert len(qs) == 1 and qs[0]["answer"] == 1 and qs[0]["options"][1] == "10:30" and qs[0]["sources"] == [1]
    listed = client.get("/collections/studio1/studio", headers=auth("nora")).json()
    assert len(listed) == 7 and listed[0]["kind"] == "quiz"  # newest first
    # only the ticked sources, and validation
    one = _make(client, "studio1", "briefing", document_ids=[b])
    assert one["document_ids"] == [b] and [s_["title"] for s_ in one["data"]["sources"]] == ["Tram"]
    r = client.post("/collections/studio1/studio", json={"kind": "quiz", "document_ids": []}, headers=auth("nora"))
    assert r.status_code == 422
    assert client.post("/collections/studio1/studio", json={"kind": "poem"}, headers=auth("nora")).status_code == 422
    assert client.post("/collections/cloud/studio", json={"kind": "faq"}, headers=auth("admin")).status_code == 403
    assert client.get("/collections/studio1/studio", headers=auth("eve")).status_code == 404
    aid = listed[0]["id"]
    assert client.delete(f"/collections/studio1/studio/{aid}", headers=auth("eve")).status_code == 404
    assert client.delete(f"/collections/studio1/studio/{aid}", headers=auth("nora")).status_code == 204


def _wav(seconds: float, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))
    return buf.getvalue()


def test_audio_overview_without_and_with_the_voice_service(client, monkeypatch):
    _setup(client, "studio2")
    art = _make(client, "studio2", "audio")
    assert art["status"] == "ready"
    turns = art["data"]["turns"]
    assert [t["speaker"] for t in turns] == ["ALEX", "SAM", "ALEX"]
    assert turns[1]["text"] == "When do they leave?" and turns[2]["text"] == "At 10:30."  # no citations / stage cues
    assert art["data"]["audio"] is False and "voice service" in art["data"]["audio_error"]
    assert client.get(f"/collections/studio2/studio/{art['id']}/audio", headers=auth("nora")).status_code == 404

    spoken = []

    def fake_speak(text, voice, s):
        spoken.append(voice)
        return b"\x01\x00" * 12000 if len(spoken) % 2 else _wav(0.5)  # raw PCM, or a WAV from older servers

    monkeypatch.setattr(audio, "speak", fake_speak)
    r = client.post(f"/collections/studio2/studio/{art['id']}/speak", headers=auth("nora"))
    assert r.status_code == 202 and r.json()["data"]["audio"] is True
    s = get_settings()
    assert spoken == [s.tts_voice_a, s.tts_voice_b, s.tts_voice_a]
    assert abs(r.json()["data"]["duration_s"] - (1.5 + 2 * 0.35)) < 0.05
    f = client.get(f"/collections/studio2/studio/{art['id']}/audio", headers=auth("nora"))
    assert f.status_code == 200 and f.headers["content-type"] == "audio/wav" and f.content[:4] == b"RIFF"


def test_join_wavs_rejects_mixed_formats():
    import pytest

    data, secs = audio.join_wavs([_wav(1.0), _wav(0.5)], pause_s=0.5)
    assert abs(secs - 2.0) < 0.01 and data[:4] == b"RIFF"
    with pytest.raises(ValueError):
        audio.join_wavs([_wav(0.2, 24000), _wav(0.2, 16000)])


def test_parsers_tolerate_messy_model_output():
    md = "Central: Climate\n- **Causes**\n  - CO2 [1]\n  - Methane\n- Effects\n  1. Heat\n"
    tree = generation.parse_mindmap(md)
    assert tree["label"] == "Climate" and tree["children"][0]["children"][0]["label"] == "CO2"
    assert generation.parse_mindmap("")["children"] == []
    quiz = "**Question 1:** What?\n(a) one\n(b) two\n**Correct answer:** (b)\n**Explanation:** Because [3]."
    q = generation.parse_quiz(quiz)
    assert q[0]["answer"] == 1 and q[0]["why"] == "Because." and q[0]["sources"] == [3]
    assert generation.structure("timeline", "NO DATES")[0] == "The sources contain no dated events."
    m, src = generation.notebook_material(
        [("d1", "A", {"summary": "About A."}, [{"text": "x " * 50}]), ("d2", "B", None, [{"text": "y"}])], 1000
    )
    assert (
        m.startswith("[1] A\nSummary: About A.")
        and "[2] B" in m
        and src[1] == {"index": 2, "document_id": "d2", "title": "B"}
    )


def test_podcast_scripts_in_other_formats():
    """Small models (DeepSeek-R1 1.5B, Llama 1B) rarely keep to "ALEX: ..." exactly."""
    from abstain_api.generation import parse_script
    from abstain_api.llm import strip_thinking

    def who(t):
        return [x["speaker"] for x in parse_script(t)]

    assert who("**Alex (excited):** Welcome!\n**Sam:** What is it?\nAlex - The Transformer.") == ["ALEX", "SAM", "ALEX"]
    assert who("Host 1: Hi there.\nHost 2: Hello all.\nHost 1: Let us start.") == ["ALEX", "SAM", "ALEX"]
    assert who("Title: Ep 1\nPriya: Today, attention.\nRavi: Why?\nPriya: It is fast.") == ["ALEX", "SAM", "ALEX"]
    prose = "Attention is all you need here.\n\nIt drops recurrence for speed.\n\nIt set a new BLEU record then."
    assert who(prose) == ["ALEX", "SAM", "ALEX"]
    assert parse_script("ok") == []
    # Ollama's DeepSeek-R1 template opens <think> in the prompt: the output only closes it
    assert strip_thinking("I should write a script...\n</think>\nALEX: Hi") == "ALEX: Hi"


def test_podcast_scripts_are_tidied_and_skip_reference_lists():
    from abstain_api.generation import is_back_matter, sample_passages, tidy_script

    def t(s, x):
        return {"speaker": s, "text": x}

    loop = [
        t("ALEX", "Today: QPolypNet for polyp segmentation."),
        t("SAM", "What does it add?"),
        t("ALEX", "A quantum-inspired layer between the ResNet-50 encoder and the decoder."),
        t("SAM", "That's a great point. What do you think is the most exciting aspect of this technology?"),
        t("ALEX", "It reaches a Dice score of 0.942 on CVC-ClinicDB."),
        t("SAM", "That's a great point. What do you think is the most exciting thing about this technology?"),
        t("ALEX", "I think the most important takeaway"),
    ]
    got = [x["text"] for x in tidy_script(loop)]
    assert got[-1].startswith("It reaches") and len(got) == 5  # repeated question and cut-off line dropped
    words = "alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega rho tau phi chi psi".split()
    long = [
        t("ALEX" if i % 2 else "SAM", f"{a} {b} {c} statement.")
        for i, (a, b, c) in enumerate(zip(words * 3, words[1:] * 3, words[3:] * 3))
    ]
    assert len(tidy_script(long)) == 16

    refs = {"section": "", "text": "[12] A. Smith et al. In Proceedings of CVPR (2023). [13] B. Lee et al. arXiv 2021."}
    assert is_back_matter(refs) and is_back_matter({"section": "References", "text": "x"})
    assert not is_back_matter({"section": "2. Method", "text": "We use ResNet-50 (He et al. 2016)."})
    body = {"section": "2. Method", "text": "The encoder is ResNet-50."}
    assert "Smith" not in sample_passages([body, refs], 2000)


def test_flashcards_in_the_shapes_small_models_write():
    from abstain_api.generation import parse_flashcards

    assert len(parse_flashcards("Q: What is X? | A: A model [1]\nQ: Dice? | A: 0.94 [1]")) == 2
    split = parse_flashcards(
        "Q: What is QPolypNet?\nA: A quantum-inspired model [1]\n\nQuestion 2: Dataset?\nAnswer: Kvasir"
    )
    assert [c["front"] for c in split] == ["What is QPolypNet?", "Dataset?"] and split[0]["sources"] == [1]
    bullets = parse_flashcards(
        "1. **QPolypNet**: segmentation model\n2. **Dice score**: 0.942 [1]\nPlain prose: not a card."
    )
    assert [c["front"] for c in bullets] == ["QPolypNet", "Dice score"]
    table = parse_flashcards("| Front | Back |\n|---|---|\n| QPolypNet | a model |\n| Dice | 0.94 [2] |")
    assert [(c["front"], c["sources"]) for c in table] == [("QPolypNet", []), ("Dice", [2])]
    assert parse_flashcards("Just a paragraph about the paper.") == []
