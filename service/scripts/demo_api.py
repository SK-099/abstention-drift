"""Run the API without models, GPU, Docker or API keys: for front-end development and demos of the UI.

    python service/scripts/demo_api.py            # http://localhost:8000, then `cd web && npm run dev`

SQLite, in-memory Qdrant, eager Celery and scripted fakes for the embedder, re-ranker, gate and LLM. The
model picker shows a canned list (local models + an OpenRouter-style catalogue); every "model" is the same
fake that answers with the best passage and cites it. Uploads, ticked sources, citations with page numbers,
notes, chats and guardrails all work as in the real service.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import tempfile
import time

TMP = tempfile.mkdtemp(prefix="abstain-demo-")
os.environ.update(
    {
        "ENV": "dev",
        "DATABASE_URL": f"sqlite+aiosqlite:///{TMP}/demo.db",
        "CHECKPOINTER": "memory",
        "CELERY_EAGER": "true",
        "QDRANT_URL": ":memory:",
        "AUTH_MODE": "dev",
        "UPLOAD_DIR": f"{TMP}/uploads",
        "PARSER": "basic",
        "GATE_THRESHOLD": "0.6",
        "SELF_CHECK_MIN_P": "0.5",
        "DENSE_DIM": "64",
        "RATE_LIMIT_CHAT": "1000/minute",
        "OPENROUTER_API_KEY": "demo",
        "LANGFUSE_PUBLIC_KEY": "",
        "CORS_ORIGINS": '["http://localhost:3000","http://127.0.0.1:3000"]',
    }
)
sys.path[:0] = [
    os.path.join(os.path.dirname(__file__), "..", "src"),
    os.path.join(os.path.dirname(__file__), "../../src"),
]

WORD = re.compile(r"[a-z0-9]+")
STOP = set("a an the is are was were of to in on for and or what how when which who does do did why with by".split())


class FakeEmbedder:
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


def _terms(t):
    return {w for w in WORD.findall(t.lower()) if w not in STOP}


class FakeReranker:
    def scores(self, query, texts):
        q = _terms(query)
        return [len(q & _terms(t)) / (len(q) or 1) for t in texts]


class FakeGate:
    threshold = 0.6

    def __call__(self, query, passages):
        return 1.0 - max((p.get("score", 0.0) for p in passages), default=0.0)


class FakeLLM:
    """Rewrite = the question; answer = the best sentence of passage [1] with a [1] citation."""

    def chat(self, messages, max_new_tokens=None):
        sys_, user = messages[0]["content"], messages[-1]["content"]
        if "answers ONLY from the provided passages" in sys_:
            return self._answer(user)
        if "standalone search query" in sys_:
            return re.search(r"Latest question: (.*)", user).group(1).strip()
        if "Proposed answer" in user:
            return "yes"
        if "too vague or ambiguous" in user:
            return "no"
        if "clarifying question" in user:
            return "Could you say which one you mean?"
        return self._answer(user)

    @staticmethod
    def _answer(user):
        from langgraph.config import get_stream_writer

        from abstain.agent import STREAM_TOKENS

        q = _terms(re.findall(r"(?:Latest question|Question):\s*(.*)", user)[-1] if "uestion:" in user else user)
        m = re.search(r"\[1\][^\n]*\n((?:.|\n)*?)(?:\n\[2\]|\Z)", user)
        text = m.group(1) if m else ""
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 3]
        best = max(sents, key=lambda s: len(q & _terms(s)), default="")
        ans = f"{best} [1]" if best else "I'm sorry, but I don't have the answer to your question."
        if STREAM_TOKENS.get():
            w = get_stream_writer()
            for tok in ans.split(" "):
                w({"type": "token", "text": tok + " "})
                time.sleep(0.03)
        return ans


DEMO_MODELS = {
    "data": [
        {"id": "anthropic/claude-sonnet-4.5", "name": "Anthropic: Claude Sonnet 4.5", "context_length": 200000},
        {"id": "anthropic/claude-haiku-4.5", "name": "Anthropic: Claude Haiku 4.5", "context_length": 200000},
        {"id": "openai/gpt-4.1", "name": "OpenAI: GPT-4.1", "context_length": 1047576},
        {"id": "openai/gpt-4.1-mini", "name": "OpenAI: GPT-4.1 Mini", "context_length": 1047576},
        {"id": "openai/o4-mini", "name": "OpenAI: o4 Mini", "context_length": 200000},
        {"id": "google/gemini-2.5-pro", "name": "Google: Gemini 2.5 Pro", "context_length": 1048576},
        {"id": "google/gemini-2.5-flash", "name": "Google: Gemini 2.5 Flash", "context_length": 1048576},
        {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Meta: Llama 3.3 70B Instruct", "context_length": 131072},
        {"id": "qwen/qwen3-32b", "name": "Qwen: Qwen3 32B", "context_length": 40960},
        {"id": "deepseek/deepseek-r1", "name": "DeepSeek: R1", "context_length": 163840},
        {"id": "mistralai/mistral-large", "name": "Mistral Large", "context_length": 128000},
    ]
}


def main() -> None:
    import uvicorn

    from abstain_api import model_registry as mr
    from abstain_api.agent_runtime import AgentRuntime, Components, make_checkpointer, set_runtime
    from abstain_api.retrieval import HybridQdrantRetriever, QdrantStore, set_shared
    from abstain_api.settings import get_settings

    def fake_get(url, headers, timeout=15.0):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen2.5:3b", "details": {"parameter_size": "3.1B"}}, {"name": "llama3.2:3b"}]}
        return DEMO_MODELS

    mr._get = fake_get
    s = get_settings()
    store = QdrantStore(s)
    store.ensure_collection()
    emb = FakeEmbedder()
    set_shared(store=store, embedder=emb)
    set_runtime(
        AgentRuntime(
            s,
            Components(
                FakeLLM(), HybridQdrantRetriever(s, store, emb, FakeReranker()), FakeGate(), make_checkpointer(s)
            ),
        )
    )
    from abstain_api.main import app

    print(f"demo API on http://localhost:8000 (data in {TMP})")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
