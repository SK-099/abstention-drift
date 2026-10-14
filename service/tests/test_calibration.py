"""Per-model refusal thresholds: selection rule, files read by the API, thresholds applied per request."""

import importlib.util
import json
from pathlib import Path

from conftest import auth
from sse import parse_sse

from abstain_api.settings import get_settings

DOC = """# Ferries

The ferry to Helgoland leaves Cuxhaven at 10:30 every morning in summer.
"""


def _script():
    f = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_models.py"
    spec = importlib.util.spec_from_file_location("calibrate_models", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(model_id, **kw):
    from abstain_api.calibration import slug

    d = Path(get_settings().calibration_dir)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"model_id": model_id, "tau": 0.5, "min_p": 1.01, "f1": 0.7, "recall": 0.6, "fa": 0.1, "n_dev": 224} | kw
    (d / f"{slug(model_id)}.json").write_text(json.dumps(rec))


def test_selection_rule_respects_false_abstention_budget():
    m = _script()
    rows = []
    # answerable: gate scores low, a few refusals; unanswerable: gate scores high
    for i in range(100):
        rows.append({"label": 0, "p": 0.1 + 0.004 * i, "refused": i < 10, "supported": i % 7 != 0})
    for i in range(40):
        rows.append({"label": 1, "p": 0.45 + 0.01 * i, "refused": i < 5, "supported": i % 2 == 0})
    best, base = m.choose(rows, fa_budget=0.10)
    assert abs(base["fa"] - 0.10) < 1e-9  # own refusals only
    assert best["fa"] <= base["fa"] + 0.10 + 1e-9
    assert best["f1"] > base["f1"]
    assert 0.30 <= best["tau"] <= 0.95
    # a stricter budget can never give a higher false-abstention rate
    strict, _ = m.choose(rows, fa_budget=0.0)
    assert strict["fa"] <= base["fa"] + 1e-9 and strict["f1"] <= best["f1"]
    assert m.evaluate(rows, 1.01, 1.01) == base


def test_models_show_which_are_calibrated(client):
    _write("local:llama3.2:3b", tau=0.42, min_p=0.3)
    _write("local:broken:1b", complete=False)
    from abstain_api.calibration import get_calibrations
    from abstain_api.model_registry import ModelInfo, get_registry

    s = get_settings()
    reg = get_registry(s)
    reg._cache["local"] = (
        1e18,
        [
            ModelInfo(f"local:{n}", "local", "Local", n, n)
            for n in (s.llm_model, "llama3.2:3b", "gemma3:4b", "broken:1b")
        ],
    )
    try:
        ms = {m["id"]: m for m in client.get("/models", headers=auth("nora")).json()["models"]}
    finally:
        reg._cache.clear()
    assert ms[f"local:{s.llm_model}"]["calibrated"]  # research thresholds
    assert ms[f"local:{s.llm_model}"]["calibration"]["source"] == "research"
    assert ms["local:llama3.2:3b"]["calibrated"] and ms["local:llama3.2:3b"]["calibration"]["tau"] == 0.42
    assert not ms["local:gemma3:4b"]["calibrated"] and ms["local:gemma3:4b"]["calibration"] is None
    assert not ms["local:broken:1b"]["calibrated"]  # partial runs are ignored
    th = get_calibrations(s).get("local:gemma3:4b")
    assert (th.tau, th.min_p, th.source) == (s.resolved_gate_threshold(), s.resolved_min_p(), "default")


def test_each_model_answers_with_its_own_thresholds(client):
    h = auth("nora")
    assert client.post("/collections", json={"id": "cal", "name": "Cal"}, headers=h).status_code == 201
    r = client.post("/collections/cal/documents", files={"file": ("f.md", DOC.encode(), "text/markdown")}, headers=h)
    assert r.json()["status"] == "indexed"
    q = "When does the ferry to Helgoland leave Cuxhaven?"
    # a model calibrated with tau = 0: every gate score counts as weak evidence, so it must abstain
    _write("local:strict:1b", tau=0.0, min_p=1.01)
    _write("local:lenient:1b", tau=0.99, min_p=1.01)
    strict = parse_sse(
        client.post(
            "/chat/stream", json={"message": q, "collection": "cal", "model": "local:strict:1b"}, headers=h
        ).text
    )[-1][1]
    lenient = parse_sse(
        client.post(
            "/chat/stream", json={"message": q, "collection": "cal", "model": "local:lenient:1b"}, headers=h
        ).text
    )[-1][1]
    assert strict["decision"] in ("abstain", "clarify") and lenient["decision"] == "answer"
    conv = client.get("/conversations?collection=cal", headers=h).json()
    metas = [client.get(f"/conversations/{c['id']}", headers=h).json()["messages"][-1]["meta"] for c in conv]
    by_model = {m["model"]: m["versions"] for m in metas}
    assert by_model["local:strict:1b"]["tau"] == 0.0 and by_model["local:strict:1b"]["thresholds"] == "calibrated"
    assert by_model["local:lenient:1b"]["tau"] == 0.99


def test_latest_tag_shares_the_calibration():
    from abstain_api.calibration import get_calibrations

    _write("local:phi4-mini", tau=0.52, min_p=1.01)
    cal = get_calibrations(get_settings())
    assert cal.get("local:phi4-mini:latest").tau == 0.52 and cal.get("local:phi4-mini:latest").calibrated
