"""End-to-end test of the whole notebook app, for every model, on your own PDFs.

Runs against the running stack (docker compose up -d) from the host, with plain Python (no packages needed):

    python scripts/smoke_test.py --pdfs paper1.pdf paper2.pdf paper3.pdf
    python scripts/smoke_test.py --pdfs a.pdf b.pdf --models qwen2.5:3b gemma3:4b --quick
    python scripts/smoke_test.py --resume smoke_results/20261020-1405      # continue after a crash / restart

What it checks
  once:      health, model list, notebook create/rename, PDF upload -> indexing -> source guide, section
             headings, file download, pasted-text / web-page / YouTube sources, Discover (web search),
             notes, sharing settings, feedback
  per model: warm-up; per PDF: a general question, a section question ("experiments and results") and an
             unanswerable one (should be refused); a follow-up in the same chat (multi-turn); two notebook-wide
             questions (summary, comparison); Studio outputs (default: FAQ + flashcards); audio overview
             (default: first model only, needs Kokoro: docker compose --profile tts up -d kokoro)

Results go to smoke_results/<time>/: results.jsonl (one line per check, used by --resume), report.md (tables
per model) and report.csv. "FAIL" = something broke (error, timeout); "WARN" = it ran but the result is not
what a good model should give (e.g. answered a question the papers cannot answer) - that is model quality,
which is what the project measures, not a bug.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------------------------------------ prompts
PER_DOC = [
    ("general", "What problem does this paper address, and what is its main contribution?", "answer"),
    ("section", "Explain the experiments of this paper and their main results.", "answer"),
    ("unanswerable", "What was the weather in the city where the authors live on the day the paper was submitted?",
     "abstain"),
]
FOLLOW_UP = ("follow_up", "What limitations does it mention?", "answer")
NOTEBOOK = [
    ("overview", "Summarize my sources.", "answer"),
    ("compare", "Compare these documents: what does each one propose?", "answer"),
]
SECTION_WORDS = ("experiment", "result", "evaluat", "benchmark", "setup", "ablation", "performance")
STUDIO_ALL = ["briefing", "study_guide", "faq", "timeline", "mindmap", "flashcards", "quiz"]


# ------------------------------------------------------------------------------------------------ HTTP
class Api:
    def __init__(self, base: str, token: str | None):
        self.base, self.token = base.rstrip("/"), token

    def _req(self, method, path, body=None, headers=None, timeout=60):
        h = {"Accept": "application/json", **(headers or {})}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None and not isinstance(body, bytes):
            data, h["Content-Type"] = json.dumps(body).encode(), "application/json"
        elif isinstance(body, bytes):
            data = body
        return urllib.request.Request(self.base + path, data=data, headers=h, method=method), timeout

    def call(self, method, path, body=None, headers=None, timeout=60, raw=False):
        """GETs are retried when the API drops the connection (it restarts, or is busy loading models)."""
        tries = 8 if method == "GET" else 1
        for attempt in range(tries):
            try:
                return self._call(method, path, body, headers, timeout, raw)
            except ApiError:
                raise
            except (OSError, http.client.HTTPException) as e:  # reset, refused, timeout, remote disconnected
                if attempt == tries - 1:
                    raise
                print(f"  ... API not answering ({type(e).__name__}), retrying in 15 s", flush=True)
                time.sleep(15)

    def _call(self, method, path, body=None, headers=None, timeout=60, raw=False):
        req, timeout = self._req(method, path, body, headers, timeout)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                content = r.read()
                if raw:
                    return r.status, content
                return r.status, (json.loads(content) if content else None)
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            try:
                detail = json.loads(text).get("detail", text)
            except Exception:
                detail = text
            raise ApiError(e.code, str(detail)[:400]) from None

    def upload(self, cid, path: Path, model: str | None):
        boundary = uuid.uuid4().hex
        ctype = mimetypes.guess_type(path.name)[0] or "application/pdf"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n".encode() + path.read_bytes() + b"\r\n"
        ]
        if model:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n".encode())
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        return self.call("POST", f"/collections/{cid}/documents", body,
                         {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=300)

    def chat(self, body, timeout):
        """POST /chat/stream and read the server-sent events; returns (final, error, steps, seconds)."""
        req, _ = self._req("POST", "/chat/stream", body, {"Accept": "text/event-stream"})
        t0, final, error, steps, ev, data = time.time(), None, None, [], "message", []
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    if line.startswith("event:"):
                        ev = line[6:].strip()
                    elif line.startswith("data:"):
                        data.append(line[5:].strip())
                    elif line == "" and data:
                        try:
                            d = json.loads("\n".join(data))
                        except Exception:
                            d = {"text": "\n".join(data)}
                        if ev == "final":
                            final = d
                        elif ev == "error":
                            error = d.get("detail") or d.get("message") or str(d)
                        elif ev == "step":
                            steps.append(d.get("node"))
                        ev, data = "message", []
        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
        except Exception as e:  # timeout, connection reset
            error = f"{type(e).__name__}: {e}"
        return final, error, steps, time.time() - t0


class ApiError(Exception):
    def __init__(self, code, detail):
        super().__init__(f"HTTP {code}: {detail}")
        self.code = code


# ------------------------------------------------------------------------------------------------ runner
class Run:
    def __init__(self, out: Path):
        self.out, self.done = out, {}
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / "results.jsonl"
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.done[r["key"]] = r
        self.state_path = out / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def save_state(self):
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def record(self, key, model, feature, status, detail="", seconds=None, **extra):
        r = {"key": key, "model": model, "feature": feature, "status": status, "detail": str(detail)[:600],
             "seconds": None if seconds is None else round(seconds, 1), "at": datetime.now().isoformat(timespec="seconds"),
             **extra}
        self.done[key] = r
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        secs = f" {r['seconds']:.0f}s" if r["seconds"] is not None else ""
        print(f"  [{status:4}] {feature:<28}{secs:>7}  {' '.join(r['detail'].split())[:110]}", flush=True)
        return r

    def check(self, key, model, feature, fn):
        """Run fn() -> (status, detail, extra) once; skipped when already in results.jsonl (resume)."""
        if key in self.done and self.done[key]["status"] != "FAIL":
            r = self.done[key]
            print(f"  [done] {feature:<28} {r['status']}", flush=True)
            return r
        t0 = time.time()
        try:
            status, detail, extra = fn()
        except ApiError as e:
            status, detail, extra = "FAIL", str(e), {}
        except Exception as e:
            status, detail, extra = "FAIL", f"{type(e).__name__}: {e}", {}
        return self.record(key, model, feature, status, detail, time.time() - t0, **extra)


def wait_docs(api, cid, ids, timeout, want_guides=True):
    """Poll until every document is indexed (or failed) and, optionally, its source guide is written."""
    t_end, last = time.time() + timeout, {}
    while time.time() < t_end:
        _, docs = api.call("GET", f"/collections/{cid}/documents")
        last = {d["id"]: d for d in docs if d["id"] in ids}
        idx_done = all(d["status"] in ("indexed", "failed", "rejected") for d in last.values())
        guide_done = all(d.get("guide_status") in ("ready", "failed", "none") or d["status"] != "indexed"
                         for d in last.values())
        if len(last) == len(ids) and idx_done and (guide_done or not want_guides):
            return last
        time.sleep(5)
    return last


def wait_artifact(api, cid, aid, timeout):
    t_end, a = time.time() + timeout, None
    while time.time() < t_end:
        _, a = api.call("GET", f"/collections/{cid}/studio/{aid}")
        stage = (a.get("data") or {}).get("stage")
        if a["status"] in ("ready", "failed") and stage != "speaking":
            return a
        time.sleep(8)
    return a


def short(model_id: str) -> str:
    return model_id.split(":", 1)[1] if model_id.startswith("local:") else model_id


# ------------------------------------------------------------------------------------------------ tests
def setup_tests(api: Api, run: Run, args, pdfs: list[Path], first_model: str | None):
    print("\n== Setup and model-independent features")
    st = run.state

    def health():
        try:
            _, r = api.call("GET", "/readyz", timeout=30)
        except ApiError as e:  # 503 when a dependency is down
            return "WARN", str(e), {}
        bad = {k: v for k, v in (r.get("checks") or {}).items() if str(v).startswith("error")}
        return ("WARN" if bad or not r.get("ready") else "PASS"), json.dumps(bad or r.get("checks"))[:200], {}
    run.check("health", "-", "health / readiness", health)

    def models():
        _, m = api.call("GET", "/models")
        loc = [x for x in m["models"] if x["provider"] == "local"]
        cal = [x["name"] for x in loc if x.get("calibrated")]
        return "PASS", f"{len(m['models'])} models, {len(loc)} local, calibrated: {', '.join(cal)}", {}
    run.check("models", "-", "model list", models)

    if not st.get("cid"):
        cid = "smoke-" + datetime.now().strftime("%m%d-%H%M")
        api.call("POST", "/collections", {"id": cid, "name": f"Smoke test {datetime.now():%Y-%m-%d %H:%M}"})
        st["cid"] = cid
        run.save_state()
    cid = st["cid"]
    print(f"  notebook: {cid}  (open http://localhost:3000/?nb={cid} to watch)")

    def rename():
        _, c = api.call("PATCH", f"/collections/{cid}", {"name": "Smoke test notebook"})
        return ("PASS" if c["name"] == "Smoke test notebook" else "FAIL"), c["name"], {}
    run.check("rename", "-", "rename notebook", rename)

    # ---- PDFs
    st.setdefault("docs", {})
    for p in pdfs:
        if p.name in st["docs"]:
            continue
        t0 = time.time()
        try:
            _, d = api.upload(cid, p, first_model)
            st["docs"][p.name] = d["id"]
            run.save_state()
            run.record(f"upload:{p.name}", "-", f"upload {p.name[:18]}", "PASS", f"{d['size_bytes']} bytes",
                       time.time() - t0)
        except Exception as e:
            run.record(f"upload:{p.name}", "-", f"upload {p.name[:18]}", "FAIL", e, time.time() - t0)
    ids = list(st["docs"].values())
    already = all(run.done.get(f"index:{n}", {}).get("status") == "PASS" and f"guide:{n}" in run.done for n in st["docs"])
    if already:
        print("  PDFs already indexed (resume)")
        return finish_setup(api, run, args, cid)
    print(f"  waiting for indexing and source guides of {len(ids)} PDF(s) (can take several minutes)...", flush=True)
    t0 = time.time()
    docs = wait_docs(api, cid, ids, args.index_timeout)
    for name, did in st["docs"].items():
        d = docs.get(did, {})
        ok = d.get("status") == "indexed"
        run.record(f"index:{name}", "-", f"index {name[:19]}", "PASS" if ok else "FAIL",
                   f"{d.get('status')} - {d.get('n_passages')} passages" + (f" - {d.get('error')}" if d.get("error") else ""),
                   time.time() - t0)
        g = d.get("guide") or {}
        gs = d.get("guide_status")
        run.record(f"guide:{name}", "-", f"source guide {name[:12]}",
                   "PASS" if gs == "ready" and g.get("summary") else ("WARN" if gs in ("pending", "running") else "FAIL"),
                   f"{gs}: " + (g.get("summary") or "")[:160] + f" | {len(g.get('questions') or [])} questions")
        if ok:
            def sections(did=did):
                _, ps = api.call("GET", f"/collections/{cid}/documents/{did}/passages")
                secs = [p.get("section") or "" for p in ps]
                distinct = sorted({s.split(" › ")[0] for s in secs if s})
                abstract = sum(1 for s in secs if s.lower().startswith("abstract")) / max(1, len(secs))
                status = "PASS" if len(distinct) >= 3 and abstract < 0.5 else "WARN"
                return status, f"{len(distinct)} top sections, {abstract:.0%} labelled Abstract: " + "; ".join(distinct[:8]), {}
            run.check(f"sections:{name}", "-", f"section headings {name[:8]}", sections)

            def download(did=did):
                code, content = api.call("GET", f"/collections/{cid}/documents/{did}/file", raw=True)
                return ("PASS" if code == 200 and content[:4] == b"%PDF" else "WARN"), f"{len(content)} bytes", {}
            run.check(f"file:{name}", "-", f"download {name[:17]}", download)

    return finish_setup(api, run, args, cid)


def finish_setup(api: Api, run: Run, args, cid: str):
    st = run.state
    # ---- other source types (kept out of the chat tests: questions use document_ids of the PDFs)
    def text_source():
        _, d = api.call("POST", f"/collections/{cid}/sources", {
            "type": "text", "title": "Pasted note",
            "text": "This pasted note is part of the smoke test. The lighthouse on the north pier was built in 1891 "
                    "and is open on Saturdays from April to October. Guided tours start every hour."})
        st["text_doc"] = d["id"]
        run.save_state()
        got = wait_docs(api, cid, [d["id"]], 300).get(d["id"], {})
        return ("PASS" if got.get("status") == "indexed" else "FAIL"), f"{got.get('status')}, {got.get('n_passages')} passages", {}
    run.check("source:text", "-", "pasted text source", text_source)

    if not args.no_web:
        def web_source():
            _, d = api.call("POST", f"/collections/{cid}/sources",
                            {"type": "url", "url": args.web_url}, timeout=180)
            got = wait_docs(api, cid, [d["id"]], 600, want_guides=False).get(d["id"], {})
            return ("PASS" if got.get("status") == "indexed" else "FAIL"), \
                f"{got.get('filename')}: {got.get('status')}, {got.get('n_passages')} passages {got.get('error') or ''}", {}
        run.check("source:web", "-", "web page source", web_source)

        def discover():
            _, hits = api.call("POST", f"/collections/{cid}/discover", {"query": args.discover_query}, timeout=90)
            return ("PASS" if hits else "WARN"), f"{len(hits)} results: " + "; ".join(h["title"][:40] for h in hits[:3]), {}
        run.check("discover", "-", "Discover (web search)", discover)

    if args.youtube:
        def youtube():
            _, d = api.call("POST", f"/collections/{cid}/sources", {"type": "youtube", "url": args.youtube}, timeout=180)
            got = wait_docs(api, cid, [d["id"]], 600, want_guides=False).get(d["id"], {})
            return ("PASS" if got.get("status") == "indexed" else "FAIL"), \
                f"{got.get('filename')}: {got.get('status')}, {got.get('n_passages')} passages {got.get('error') or ''}", {}
        run.check("source:youtube", "-", "YouTube source", youtube)

    # ---- notes and sharing
    def notes():
        _, n = api.call("POST", f"/collections/{cid}/notes", {"title": "Smoke note", "content": "hello"})
        api.call("PATCH", f"/collections/{cid}/notes/{n['id']}", {"pinned": True, "content": "edited"})
        _, lst = api.call("GET", f"/collections/{cid}/notes")
        found = [x for x in lst if x["id"] == n["id"]]
        api.call("DELETE", f"/collections/{cid}/notes/{n['id']}")
        ok = found and found[0]["content"] == "edited" and found[0]["pinned"]
        return ("PASS" if ok else "FAIL"), "create, edit, pin, list, delete", {}
    run.check("notes", "-", "notes", notes)

    def sharing():
        api.call("PUT", f"/collections/{cid}/sharing/members", {"principal": "friend@example.com", "role": "viewer"})
        _, sh = api.call("PATCH", f"/collections/{cid}/sharing", {"link_access": "viewer"})
        ok = sh["link_access"] == "viewer" and any(m["principal"] == "friend@example.com" for m in sh["members"])
        api.call("DELETE", f"/collections/{cid}/sharing/members/friend@example.com")
        _, sh = api.call("PATCH", f"/collections/{cid}/sharing", {"link_access": "none"})
        ok = ok and not sh["members"] and sh["link_access"] == "none"
        return ("PASS" if ok else "FAIL"), "member add/remove, link access on/off (a second login is needed to test access itself)", {}
    run.check("sharing", "-", "sharing settings", sharing)
    return cid


def model_tests(api: Api, run: Run, args, cid: str, mid: str, audio: bool):
    m = short(mid)
    print(f"\n== Model {m}")
    st = run.state
    pdf_ids = list(st["docs"].values())
    names = list(st["docs"].keys())

    def warm():
        _, r = api.call("POST", "/models/warm", {"model": mid}, timeout=args.chat_timeout)
        return ("PASS" if r.get("loaded") else "WARN"), f"loaded in {r.get('seconds')} s {r.get('detail') or ''}", {}
    run.check(f"{m}|warm", m, "warm-up (load model)", warm)

    def ask(key, label, question, expect, doc_ids, conversation=None):
        def fn():
            body = {"message": question, "collection": cid, "model": mid, "document_ids": doc_ids}
            if conversation:
                body["conversation_id"] = conversation
            final, error, steps, secs = api.chat(body, args.chat_timeout)
            if error or not final:
                return "FAIL", error or "no final answer", {}
            cites = final.get("citations") or []
            secs_ = [c.get("section") or "" for c in cites]
            extra = {"decision": final["decision"], "p": final.get("p_unanswerable"), "citations": len(cites),
                     "abstained_by": final.get("abstained_by"), "conversation_id": final.get("conversation_id"),
                     "message_id": final.get("message_id"), "steps": steps, "answer": (final.get("answer") or "")[:400]}
            got = final["decision"]
            if expect == "abstain":
                status = "PASS" if got == "abstain" else "WARN"
            else:
                status = "PASS" if got == "answer" and (cites or label in ("overview", "compare")) else "WARN"
            if label == "section" and got == "answer" and not any(w in s.lower() for s in secs_ for w in SECTION_WORDS):
                status = "WARN"
            p = final.get("p_unanswerable")
            detail = (f"{got}" + (f" by {final.get('abstained_by')}" if got == "abstain" else "")
                      + f", p={p}, {len(cites)} cites [{'; '.join(s[:30] for s in secs_[:3])}] - "
                      + (final.get("answer") or "").replace("\n", " ")[:160])
            return status, detail, extra
        return run.check(key, m, label, fn)

    for name, did in zip(names, pdf_ids if not args.quick else pdf_ids[:1]):
        tag = Path(name).stem[:10]
        for label, q, expect in PER_DOC:
            r = ask(f"{m}|{name}|{label}", f"{label} [{tag}]", q, expect, [did])
            if label == "general" and name == names[0] and r.get("conversation_id"):
                st.setdefault("conv", {})[m] = r["conversation_id"]
                run.save_state()
        if name == names[0] and st.get("conv", {}).get(m):
            fl, q, expect = FOLLOW_UP
            ask(f"{m}|{name}|{fl}", f"{fl} [{tag}]", q, expect, [did], conversation=st["conv"][m])
    if len(pdf_ids) > 1:
        for label, q, expect in NOTEBOOK:
            ask(f"{m}|nb|{label}", label, q, expect, pdf_ids)

    # feedback on one answer
    msg = next((r.get("message_id") for k, r in run.done.items() if k.startswith(f"{m}|") and r.get("message_id")), None)
    if msg:
        run.check(f"{m}|feedback", m, "feedback (thumbs up)", lambda: (
            "PASS", f"saved {api.call('POST', '/feedback', {'message_id': msg, 'rating': 1})[1].get('id', '')}", {}))

    # Studio
    kinds = [k for k in args.studio if k != "none"]
    if audio:
        kinds.append("audio")
    for kind in kinds:
        def studio(kind=kind):
            _, a = api.call("POST", f"/collections/{cid}/studio",
                            {"kind": kind, "model": mid, "document_ids": pdf_ids}, timeout=60)
            a = wait_artifact(api, cid, a["id"], args.studio_timeout)
            if a is None or a["status"] not in ("ready", "failed"):
                return "FAIL", f"not finished after {args.studio_timeout} s (status {a and a['status']})", {}
            if a["status"] == "failed":
                return "FAIL", a.get("error") or "failed", {}
            d = a.get("data") or {}
            n = {"faq": len(d.get("items") or []), "flashcards": len(d.get("cards") or []),
                 "quiz": len(d.get("questions") or []), "timeline": len(d.get("events") or []),
                 "mindmap": len((d.get("tree") or {}).get("children") or []), "audio": len(d.get("turns") or [])
                 }.get(kind, len(a.get("content") or ""))
            unit = {"faq": "Q&As", "flashcards": "cards", "quiz": "questions", "timeline": "events",
                    "mindmap": "branches", "audio": "turns"}.get(kind, "chars")
            detail = f"{n} {unit}"
            status = "PASS" if n else "WARN"
            if kind == "audio":
                if d.get("audio"):
                    detail += f", audio {d.get('duration_s')} s"
                else:
                    status, detail = "WARN", detail + f", no audio: {d.get('audio_error') or 'not spoken'}"
            return status, detail + " - " + (a.get("content") or "").replace("\n", " ")[:120], {}
        run.check(f"{m}|studio|{kind}", m, f"studio: {kind}", studio)

    if args.guides:
        def guide():
            did = pdf_ids[0]
            api.call("POST", f"/collections/{cid}/documents/{did}/guide?model={urllib.request.quote(mid)}")
            time.sleep(3)
            d = wait_docs(api, cid, [did], args.studio_timeout).get(did, {})
            g = d.get("guide") or {}
            return ("PASS" if d.get("guide_status") == "ready" and g.get("summary") else "FAIL"), \
                f"{d.get('guide_status')}: {(g.get('summary') or '')[:150]}", {}
        run.check(f"{m}|guide", m, "source guide (this model)", guide)


# ------------------------------------------------------------------------------------------------ report
def cell(text: str, n: int) -> str:
    return " ".join(str(text).split()).replace("|", "/")[:n]


def secs(r: dict) -> str:
    return "" if r.get("seconds") is None else f"{r['seconds']:.0f} s"


def write_report(run: Run, models: list[str]):
    rows = list(run.done.values())
    with (run.out / "report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "feature", "status", "seconds", "decision", "p", "citations", "detail"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    lines = [f"# Smoke test report\n\n{datetime.now():%Y-%m-%d %H:%M} - notebook `{run.state.get('cid')}`\n"]
    totals = {s: sum(1 for r in rows if r["status"] == s) for s in ("PASS", "WARN", "FAIL")}
    lines.append(f"**{totals['PASS']} passed, {totals['WARN']} warnings, {totals['FAIL']} failed**\n")
    lines.append("FAIL = something broke. WARN = it worked, but the answer is not what a good model should give "
                 "(e.g. it answered a question the papers cannot answer).\n")
    lines.append("## Model-independent features\n\n| Check | Status | Time | Detail |\n|---|---|---|---|")
    for r in rows:
        if r["model"] == "-":
            lines.append(f"| {r['feature']} | {icon.get(r['status'], r['status'])} | {secs(r)} | "
                         f"{cell(r['detail'], 220)} |")
    # per-model summary
    ms = [short(x) for x in models] or sorted({r["model"] for r in rows if r["model"] != "-"})
    lines.append("\n## Per model\n\n| Model | Pass | Warn | Fail | Refused unanswerable | Answered answerable | Median answer time |"
                 "\n|---|---|---|---|---|---|---|")
    for m in ms:
        rs = [r for r in rows if r["model"] == m]
        if not rs:
            continue
        un = [r for r in rs if "unanswerable" in r["feature"] and r.get("decision")]
        an = [r for r in rs if r.get("decision") and "unanswerable" not in r["feature"]]
        times = sorted(r["seconds"] for r in rs if r.get("decision") and r["seconds"])
        med = times[len(times) // 2] if times else None
        lines.append(f"| {m} | {sum(r['status'] == 'PASS' for r in rs)} | {sum(r['status'] == 'WARN' for r in rs)} | "
                     f"{sum(r['status'] == 'FAIL' for r in rs)} | {sum(r['decision'] == 'abstain' for r in un)}/{len(un)} | "
                     f"{sum(r['decision'] == 'answer' for r in an)}/{len(an)} | {med or '-'} s |")
    for m in ms:
        rs = [r for r in rows if r["model"] == m]
        if not rs:
            continue
        lines.append(f"\n### {m}\n\n| Check | Status | Time | Detail |\n|---|---|---|---|")
        for r in rs:
            lines.append(f"| {r['feature']} | {icon.get(r['status'], r['status'])} | {secs(r)} | "
                         f"{cell(r['detail'], 260)} |")
    (run.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return totals


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdfs", nargs="+", default=[], help="2-3 PDF files (papers work best)")
    ap.add_argument("--models", nargs="*", help="model names (e.g. qwen2.5:3b) or ids; default: all local models")
    ap.add_argument("--api", default=os.environ.get("API_URL", "http://localhost:8000"))
    ap.add_argument("--token", default=os.environ.get("API_TOKEN"), help="bearer token (only if AUTH_MODE is not dev)")
    ap.add_argument("--quick", action="store_true", help="per-PDF questions on the first PDF only")
    ap.add_argument("--studio", nargs="*", default=["faq", "flashcards"],
                    help=f"Studio kinds per model ({' '.join(STUDIO_ALL)}), 'all' or 'none'")
    ap.add_argument("--audio-models", type=int, default=1, help="how many models also make an audio overview (0 = none)")
    ap.add_argument("--guides", action="store_true", help="also rewrite the first PDF's source guide with each model")
    ap.add_argument("--no-web", action="store_true", help="skip the web-page source and Discover (no internet)")
    ap.add_argument("--web-url", default="https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    ap.add_argument("--discover-query", default="retrieval augmented generation hallucination")
    ap.add_argument("--youtube", help="a YouTube link with captions to test the YouTube source")
    ap.add_argument("--chat-timeout", type=int, default=900)
    ap.add_argument("--studio-timeout", type=int, default=1800)
    ap.add_argument("--index-timeout", type=int, default=1800)
    ap.add_argument("--resume", help="results folder of an earlier run to continue")
    ap.add_argument("--out", default="smoke_results")
    args = ap.parse_args()
    if args.studio == ["all"]:
        args.studio = STUDIO_ALL
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    api = Api(args.api, args.token)
    run = Run(Path(args.resume) if args.resume else Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M"))
    pdfs = [Path(p) for p in args.pdfs]
    if not pdfs and not run.state.get("docs"):
        ap.error("give 2-3 PDFs with --pdfs")
    missing = [str(p) for p in pdfs if not p.is_file()]
    if missing:
        ap.error(f"not found: {', '.join(missing)}")

    try:
        _, catalog = api.call("GET", "/models", timeout=30)
    except Exception as e:
        sys.exit(f"The API at {args.api} is not reachable ({e}). Start it with: docker compose up -d")
    local = [x["id"] for x in catalog["models"] if x["provider"] == "local"]
    if args.models:
        want, models = args.models, []
        for w in want:
            hit = [i for i in [x["id"] for x in catalog["models"]] if i == w or short(i) == w or short(i) == f"{w}:latest"
                   or short(i).removesuffix(":latest") == w]
            if hit:
                models.append(hit[0])
            else:
                print(f"! model {w} not in the menu, skipped")
    else:
        models = local
    print(f"Results: {run.out}\nModels ({len(models)}): {', '.join(short(m) for m in models)}")
    t0 = time.time()
    try:
        cid = setup_tests(api, run, args, pdfs, models[0] if models else None)
        if not any(r["status"] == "PASS" for k, r in run.done.items() if k.startswith("index:")):
            print("\nNo PDF was indexed: stopping before the model tests.")
        else:
            for i, mid in enumerate(models):
                model_tests(api, run, args, cid, mid, audio=i < args.audio_models)
                write_report(run, models)
    except KeyboardInterrupt:
        print("\nStopped. Continue later with:  python scripts/smoke_test.py --resume", run.out)
    except Exception as e:  # the API went away for good: keep what was done, say how to continue
        print(f"\nStopped by an error: {type(e).__name__}: {e}\n"
              "Check the stack (docker compose ps; docker compose logs --tail 50 api worker), then continue with:\n"
              f"  python scripts/smoke_test.py --resume {run.out}")
    totals = write_report(run, models)
    print(f"\nDone in {(time.time() - t0) / 60:.0f} min: {totals['PASS']} passed, {totals['WARN']} warnings, "
          f"{totals['FAIL']} failed\nReport: {run.out / 'report.md'}")


if __name__ == "__main__":
    main()
