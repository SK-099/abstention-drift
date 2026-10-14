"""Test environment: SQLite (or $TEST_DATABASE_URL), in-memory Qdrant, in-memory LangGraph checkpointer,
eager Celery, HS256 auth, and scripted fakes for the LLM, embedder, re-ranker and answerability gate."""

import hashlib
import math
import os
import re
import tempfile

import pytest

TMP = tempfile.mkdtemp(prefix="abstain-test-")
os.environ.update(
    {
        "ENV": "test",
        "DATABASE_URL": os.environ.get("TEST_DATABASE_URL", f"sqlite+aiosqlite:///{TMP}/test.db"),
        "CHECKPOINTER": os.environ.get("TEST_CHECKPOINTER", "memory"),
        "CELERY_EAGER": "true",
        "QDRANT_URL": os.environ.get("TEST_QDRANT_URL", ":memory:"),
        "QDRANT_COLLECTION": f"test_{os.getpid()}",
        "AUTH_MODE": "hs256",
        "AUTH_SECRET": "test-secret-test-secret-test-secret!",
        "ADMIN_USER_IDS": '["admin"]',
        "VECTOR_STORE": os.environ.get("TEST_VECTOR_STORE", "qdrant"),
        "UPLOAD_DIR": f"{TMP}/uploads",
        "CALIBRATION_DIR": f"{TMP}/calibration",
        "PARSER": "basic",
        "GATE_THRESHOLD": "0.5",
        "SELF_CHECK_MIN_P": "0.0",
        "RATE_LIMIT_CHAT": "1000/minute",
        "DENSE_DIM": "64",
        "CHUNK_WORDS": "60",
        "CHUNK_OVERLAP": "10",
    }
)

import jwt
from fastapi.testclient import TestClient

from abstain_api.agent_runtime import AgentRuntime, Components, make_checkpointer, set_runtime
from abstain_api.retrieval import HybridQdrantRetriever, PgVectorStore, QdrantStore, set_shared
from abstain_api.settings import get_settings

WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims (deterministic, no model download)."""

    dim = 64

    def _vec(self, t):
        v = [0.0] * self.dim
        for w in WORD.findall(t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_query(self, q):
        return self._vec(q)

    def embed_documents(self, ts):
        return [self._vec(t) for t in ts]


class FakeReranker:
    def scores(self, query, texts):
        q = set(WORD.findall(query.lower()))
        return [len(q & set(WORD.findall(t.lower()))) / (len(q) or 1) for t in texts]


class FakeGate:
    """p(unanswerable) = 1 - best re-ranker score of the retrieved passages."""

    threshold = 0.5

    def __call__(self, query, passages):
        return 1.0 - max((p.get("score", 0.0) for p in passages), default=0.0)


class FakeLLM:
    """Rewrite = copy the question; answer = first passage sentence; self-check always supports."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, max_new_tokens=None):
        sys_, user = messages[0]["content"], messages[-1]["content"]
        self.calls.append(sys_[:30])
        if "short guides to documents" in sys_:
            first = re.sub(r"\s+", " ", user.split("\n\n", 1)[-1])[:160]
            return f"SUMMARY: The document says: {first}\nTOPICS: travel; timetables; ferries\nQUESTIONS:\n- When does it leave?\n- Where does it go?\n- How often does it run?"
        if "standalone search query" in sys_:
            return re.search(r"Latest question: (.*)", user).group(1).strip()
        if "Proposed answer" in user:
            return "yes"
        if "too vague or ambiguous" in user:
            return "no"
        if "clarifying question" in user:
            return "Could you say which one you mean?"
        from langgraph.config import get_stream_writer

        from abstain.agent import STREAM_TOKENS

        m = re.search(r"\[1\][^\n]*\n?((?:[^\n]+\n?){1,3})", user)
        lines = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()] if m else []
        # (passages start with their section header: answer with the longest of the first lines)
        ans = (max(lines, key=len) if lines else "The documents say so.")[:200]
        if STREAM_TOKENS.get():
            w = get_stream_writer()
            for tok in ans.split(" "):
                w({"type": "token", "text": tok + " "})
        return ans


def token(sub="alice", **claims):
    import time

    return jwt.encode({"sub": sub, "exp": int(time.time()) + 600, **claims}, os.environ["AUTH_SECRET"], "HS256")


def auth(sub="alice", **claims):
    return {"Authorization": f"Bearer {token(sub, **claims)}"}


@pytest.fixture(scope="session")
def store():
    s = get_settings()
    st = PgVectorStore(s) if s.vector_store == "pgvector" else QdrantStore(s)
    st.ensure_collection()
    return st


@pytest.fixture(scope="session")
def client(store):
    s = get_settings()
    emb = FakeEmbedder()
    set_shared(store=store, embedder=emb)
    llm = FakeLLM()
    rt = AgentRuntime(
        s, Components(llm, HybridQdrantRetriever(s, store, emb, FakeReranker()), FakeGate(), make_checkpointer(s))
    )
    set_runtime(rt)
    if s.database_url.startswith("postgresql"):  # start from empty tables (schema from `alembic upgrade head`)
        from sqlalchemy import text

        from abstain_api.db import Base, sync_engine

        Base.metadata.create_all(sync_engine())
        with sync_engine().begin() as conn:
            conn.execute(text("TRUNCATE " + ", ".join(t.name for t in Base.metadata.sorted_tables) + " CASCADE"))
    from abstain_api.main import app

    with TestClient(app) as c:
        c.llm = llm
        yield c
    if s.vector_store == "pgvector":
        with store._conn() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {store.table}")
    elif s.qdrant_url != ":memory:":
        store.client.delete_collection(store.name)
