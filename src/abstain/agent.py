r"""S5: a self-correcting abstention agent built with LangGraph (LangChain).

The fixed pipeline (S2/S4) rewrites once, retrieves once and then answers or refuses. The agent adds
two behaviours where our error analysis found most failures:

  1. Retry before refusing: if the evidence looks weak (gate score p >= tau) or the generator refuses,
     rewrite the question with a different strategy (expand abbreviations, keep the user's own words,
     do not carry over an old topic), retrieve again and re-grade (at most `max_retries` times).
     If it still fails, ask a clarifying question when the question is ambiguous, otherwise abstain.
  2. Self-check after answering: an LLM judge checks that the answer is supported by the passages;
     if not, the agent abstains instead of returning an unsupported answer.

    rewrite -> retrieve -> grade --ok--> generate --answered--> self_check --supported--> ANSWER
                             |               \--refused--\           \--unsupported--> ABSTAIN
                             \--weak---------------------> retry? --yes--> (rewrite again)
                                                              \--no--> clarify_or_abstain -> CLARIFY | ABSTAIN

The graph is deliberately rule-based (fixed routing) rather than a free tool-calling agent: a 3B model
follows explicit steps reliably but chooses tools poorly, and fixed routing keeps the agent evaluable.

LangChain pieces: `ChatOllama` for the LLM (wrapped so the existing prompts can be reused), a
`BaseRetriever` subclass around our hybrid retriever, and a LangGraph `StateGraph` for control flow.
"""
from __future__ import annotations

import contextvars
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypedDict

from .data import Task, format_history
from .generate import format_passages, generate_answer
from .idk import IDK_RESPONSE, is_idk
from .rewrite import QueryRewriter

# ----------------------------------------------------------------------------- prompts
RETRY_REWRITE_SYSTEM = (
    "The previous search query did not find passages that answer the user's latest question. "
    "Write a DIFFERENT standalone search query for that question. Expand abbreviations and acronyms, "
    "keep the user's own key words, and do not carry over an earlier topic unless the latest question "
    "clearly refers to it. Output only the new query."
)
SELF_CHECK_PROMPT = (
    "Passages:\n{p}\n\nQuestion: {q}\n\nProposed answer: {a}\n\n"
    "Is the main claim of the proposed answer directly supported by the passages? "
    "Reply with exactly one word: yes or no."
)
AMBIGUOUS_PROMPT = (
    "Conversation so far:\n{h}\n\nLatest question: {q}\n\n"
    "Is the latest question too vague or ambiguous to answer without first asking the user what they "
    "mean (for example, it could refer to several different things)? Reply with exactly one word: yes or no."
)
CLARIFY_PROMPT = (
    "Conversation so far:\n{h}\n\nLatest question: {q}\n\n"
    "The question is ambiguous. Ask the user ONE short clarifying question that would let you answer it. "
    "Output only the question."
)


# Set while the `generate` node runs. An LLM wrapper that supports streaming (the web service's
# OpenAI-compatible client) emits each token through LangGraph's stream writer while this is True;
# other wrappers ignore it, so research runs are unaffected.
STREAM_TOKENS: contextvars.ContextVar[bool] = contextvars.ContextVar("stream_tokens", default=False)


@contextmanager
def streaming_tokens():
    tok = STREAM_TOKENS.set(True)
    try:
        yield
    finally:
        STREAM_TOKENS.reset(tok)


def _yes(text: str, default: bool) -> bool:
    t = (text or "").strip().lower()
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    return default          # unparseable judge output: fall back to the safe default


def self_check_supported(llm, query: str, passages: list, answer: str, passage_chars: int | None = None) -> bool:
    """LLM judge: is the answer's main claim supported by the passages? (unparseable output -> True)"""
    msg = SELF_CHECK_PROMPT.format(p=format_passages(passages, passage_chars), q=query, a=answer[:800])
    return _yes(llm.chat([{"role": "user", "content": msg}], max_new_tokens=3), default=True)


# ----------------------------------------------------------------------------- LangChain wrappers
class LangChainChat:
    """LangChain `ChatOllama` behind the `.chat(messages, max_new_tokens)` interface used by the
    rest of the code base, so the existing rewrite/generation prompts are reused unchanged."""

    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434",
                 max_new_tokens: int = 256, num_ctx: int = 4096, retries: int = 3, wait: float = 15.0):
        import importlib.util

        if importlib.util.find_spec("langchain_ollama") is None:
            raise ImportError("Install the agent dependencies: pip install langgraph langchain-core langchain-ollama")

        self.model, self.base_url, self.max_new_tokens, self.num_ctx = model, base_url, max_new_tokens, num_ctx
        self.retries, self.wait = retries, wait
        self._models: dict[int, Any] = {}

    def _model(self, n: int):
        if n not in self._models:
            from langchain_ollama import ChatOllama

            self._models[n] = ChatOllama(model=self.model, base_url=self.base_url, temperature=0, seed=13,
                                         num_predict=n, num_ctx=self.num_ctx, keep_alive="60m")
        return self._models[n]

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        cls = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        msgs = [cls[m["role"]](content=m["content"]) for m in messages]
        for attempt in range(self.retries):
            try:
                return str(self._model(max_new_tokens or self.max_new_tokens).invoke(msgs).content).strip()
            except Exception:  # Ollama occasionally returns HTTP 500 after long runs
                if attempt == self.retries - 1:
                    raise
                time.sleep(self.wait)
        return ""


def make_retriever(retrievers: dict, k: int = 5):
    """LangChain `BaseRetriever` around our per-collection HybridRetriever objects."""
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class HybridLCRetriever(BaseRetriever):
        retrievers: dict
        k: int = 5

        def _get_relevant_documents(self, query: str, *, run_manager=None, collection: str = "") -> list:
            ps = self.retrievers[collection].search(query, k=self.k)
            return [Document(page_content=p["text"],
                             metadata={"document_id": p["document_id"], "title": p.get("title", ""),
                                       "score": float(p.get("score", 0.0))}) for p in ps]

    return HybridLCRetriever(retrievers=retrievers, k=k)


def docs_to_passages(docs) -> list[dict]:
    return [{"document_id": d.metadata["document_id"], "title": d.metadata.get("title", ""),
             "text": d.page_content, "score": d.metadata.get("score", 0.0)} for d in docs]


# ----------------------------------------------------------------------------- agent
class AgentState(TypedDict, total=False):
    task: Task
    cache: dict                 # optional first-attempt results from a finished S2 run
    attempt: int
    query: str
    queries: list
    passages: list
    p: float | None
    answer: str
    decision: str               # answer | abstain | clarify
    reason: str                 # gate | generator | self-check | clarify | none
    trace: list
    timings: dict


@dataclass
class AgentOutput:
    task_id: str
    query: str
    passages: list
    p_unanswerable: float | None
    abstained: bool
    answer: str
    abstained_by: str
    decision: str
    attempts: int
    queries: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


class AbstentionAgent:
    """llm: object with .chat(); retriever: LangChain retriever (invoke(query, collection=...));
    gate: callable (query, passages) -> P(unanswerable) with a .threshold attribute (or None)."""

    def __init__(self, llm, retriever, gate=None, threshold: float | None = None, max_retries: int = 1,
                 rewrite_kwargs: dict | None = None, gen_kwargs: dict | None = None,
                 self_check: bool = True, clarify: bool = True, self_check_min_p: float = 0.0,
                 checkpointer=None):
        """checkpointer: optional LangGraph checkpointer (e.g. PostgresSaver) that persists the agent
        state per conversation thread; pass thread_id when calling the agent."""
        from langgraph.graph import END, StateGraph

        self.llm, self.retriever, self.gate = llm, retriever, gate
        self.threshold = threshold if threshold is not None else getattr(gate, "threshold", 1.0)
        self.max_retries, self.self_check, self.clarify = max_retries, self_check, clarify
        # selective self-check: only verify answers whose evidence the gate already doubts (p >= min_p);
        # a 3B judge rejects most answers, so checking every answer causes heavy over-refusal
        self.self_check_min_p = self_check_min_p
        self.rw_kw, self.gen_kw = rewrite_kwargs or {}, gen_kwargs or {}
        self.rewriter = QueryRewriter("llm", llm=llm, **self.rw_kw)

        g = StateGraph(AgentState)
        for name, fn in [("rewrite", self._rewrite), ("retrieve", self._retrieve), ("grade", self._grade),
                         ("generate", self._generate), ("self_check", self._self_check),
                         ("retry", self._retry), ("clarify_or_abstain", self._clarify_or_abstain)]:
            g.add_node(name, fn)
        g.set_entry_point("rewrite")
        g.add_conditional_edges("rewrite", self._after_rewrite, {"search": "retrieve", "repeat": "clarify_or_abstain"})
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._after_grade, {"ok": "generate", "weak": "retry"})
        g.add_conditional_edges("generate", self._after_generate,
                                {"answered": "self_check", "refused": "retry", "done": END})
        g.add_edge("self_check", END)
        g.add_conditional_edges("retry", self._after_retry, {"again": "rewrite", "give_up": "clarify_or_abstain"})
        g.add_edge("clarify_or_abstain", END)
        self.graph = g.compile(checkpointer=checkpointer)

    # -- helpers
    def _cached(self, s: AgentState, key: str):
        return (s.get("cache") or {}).get(key) if s.get("attempt", 0) == 0 else None

    @staticmethod
    def _tick(s: AgentState, stage: str, t0: float) -> dict:
        tm = dict(s.get("timings") or {})
        tm[stage] = tm.get(stage, 0.0) + time.perf_counter() - t0
        return tm

    # -- nodes
    def _rewrite(self, s: AgentState) -> dict:
        t0, task, n = time.perf_counter(), s["task"], s.get("attempt", 0)
        q = self._cached(s, "query")
        if q is None and n == 0:
            q = self.rewriter(task)
        elif q is None:
            hist = format_history(task.history, self.rw_kw.get("history_turns", 3), self.rw_kw.get("history_chars"))
            tried = "; ".join(s.get("queries", []))
            q = self.llm.chat([{"role": "system", "content": RETRY_REWRITE_SYSTEM},
                               {"role": "user", "content": f"Conversation:\n{hist}\n\nLatest question: {task.question}\n"
                                                           f"Queries already tried: {tried}\n\nNew query:"}],
                              max_new_tokens=64).split("\n")[0].strip().strip('"') or task.question
        return {"query": q, "queries": list(s.get("queries", [])) + [q],
                "trace": list(s.get("trace", [])) + [f"rewrite[{n}]: {q}"], "timings": self._tick(s, "rewrite", t0)}

    def _retrieve(self, s: AgentState) -> dict:
        t0 = time.perf_counter()
        ps = self._cached(s, "passages")
        if ps is None:
            ps = docs_to_passages(self.retriever.invoke(s["query"], collection=s["task"].collection))
        return {"passages": ps, "trace": list(s["trace"]) + [f"retrieve[{s.get('attempt', 0)}]: {len(ps)} passages"],
                "timings": self._tick(s, "retrieve", t0)}

    def _grade(self, s: AgentState) -> dict:
        t0 = time.perf_counter()
        p = self._cached(s, "p")
        if p is None and self.gate is not None:
            p = float(self.gate(s["query"], s["passages"]))
        tag = "none" if p is None else f"{p:.3f} ({'weak' if p >= self.threshold else 'ok'})"
        return {"p": p, "trace": list(s["trace"]) + [f"grade[{s.get('attempt', 0)}]: p={tag}"],
                "timings": self._tick(s, "grade", t0)}

    def _generate(self, s: AgentState) -> dict:
        t0 = time.perf_counter()
        a = self._cached(s, "answer")
        if a is None:
            with streaming_tokens():
                a = generate_answer(self.llm, s["task"], s["passages"], **self.gen_kw)
        refused = is_idk(a)
        return {"answer": a, "trace": list(s["trace"]) + [f"generate[{s.get('attempt', 0)}]: {'refused' if refused else 'answered'}"],
                "timings": self._tick(s, "generate", t0)}

    def _self_check(self, s: AgentState) -> dict:
        t0 = time.perf_counter()
        p = s.get("p")
        run = self.self_check and (p is None or p >= self.self_check_min_p)
        ok = self_check_supported(self.llm, s["query"], s["passages"], s["answer"],
                                  self.gen_kw.get("passage_chars")) if run else True
        out = {"decision": "answer", "reason": "none"} if ok else \
              {"decision": "abstain", "reason": "self-check", "answer": IDK_RESPONSE}
        msg = "skipped (p below min_p)" if self.self_check and not run else \
              ("supported" if ok else "NOT supported -> abstain")
        out.update(trace=list(s["trace"]) + [f"self_check: {msg}"], timings=self._tick(s, "self_check", t0))
        return out

    def _retry(self, s: AgentState) -> dict:
        why = "gate" if s.get("p") is not None and s["p"] >= self.threshold else "generator"
        return {"attempt": s.get("attempt", 0) + 1, "reason": why,
                "trace": list(s["trace"]) + [f"retry: evidence weak ({why})"]}

    def _clarify_or_abstain(self, s: AgentState) -> dict:
        t0, task = time.perf_counter(), s["task"]
        if self.clarify:
            hist = format_history(task.history, self.rw_kw.get("history_turns", 3), self.rw_kw.get("history_chars"))
            amb = _yes(self.llm.chat([{"role": "user", "content": AMBIGUOUS_PROMPT.format(h=hist or "(none)", q=task.question)}],
                                     max_new_tokens=3), default=False)
            if amb:
                cq = self.llm.chat([{"role": "user", "content": CLARIFY_PROMPT.format(h=hist or "(none)", q=task.question)}],
                                   max_new_tokens=48).strip()
                return {"decision": "clarify", "reason": "clarify", "answer": cq,
                        "trace": list(s["trace"]) + [f"clarify: {cq}"], "timings": self._tick(s, "clarify", t0)}
        return {"decision": "abstain", "answer": IDK_RESPONSE, "reason": s.get("reason", "gate"),
                "trace": list(s["trace"]) + ["abstain"], "timings": self._tick(s, "clarify", t0)}

    # -- routing
    @staticmethod
    def _after_rewrite(s: AgentState) -> str:
        """A retry query identical to one already tried would retrieve the same passages and (greedy decoding)
        produce the same refusal, so go straight to the clarify-or-abstain decision."""
        if s.get("attempt", 0) > 0:
            norm = lambda q: " ".join(q.lower().split())
            if norm(s["query"]) in {norm(q) for q in s.get("queries", [])[:-1]}:
                return "repeat"
        return "search"

    def _after_grade(self, s: AgentState) -> str:
        return "weak" if s.get("p") is not None and s["p"] >= self.threshold else "ok"

    def _after_generate(self, s: AgentState) -> str:
        return "refused" if is_idk(s["answer"]) else "answered"

    def _after_retry(self, s: AgentState) -> str:
        return "again" if s["attempt"] <= self.max_retries else "give_up"

    # -- public
    @staticmethod
    def initial_state(task: Task, cache: dict | None = None) -> dict:
        return {"task": task, "cache": cache or {}, "attempt": 0, "queries": [], "trace": [], "timings": {},
                "decision": "", "reason": "", "answer": "", "p": None}

    @staticmethod
    def output_from_state(task: Task, s: dict) -> AgentOutput:
        dec = s.get("decision") or "abstain"
        return AgentOutput(task.task_id, s.get("query", ""), s.get("passages", []), s.get("p"),
                           dec != "answer", s.get("answer", IDK_RESPONSE),
                           "none" if dec == "answer" else s.get("reason", "gate"), dec,
                           max(len(s.get("queries", [])) - 1, 0), s.get("queries", []), s.get("trace", []),
                           s.get("timings", {}))

    @staticmethod
    def make_config(thread_id: str | None = None, callbacks: list | None = None, metadata: dict | None = None) -> dict:
        cfg: dict = {"recursion_limit": 50}
        if thread_id:
            cfg["configurable"] = {"thread_id": thread_id}
        if callbacks:
            cfg["callbacks"] = callbacks
        if metadata:
            cfg["metadata"] = metadata
        return cfg

    def __call__(self, task: Task, cache: dict | None = None, thread_id: str | None = None,
                 callbacks: list | None = None, metadata: dict | None = None) -> AgentOutput:
        s = self.graph.invoke(self.initial_state(task, cache), config=self.make_config(thread_id, callbacks, metadata))
        return self.output_from_state(task, s)

    def stream(self, task: Task, thread_id: str | None = None, callbacks: list | None = None,
               metadata: dict | None = None, cache: dict | None = None):
        """Yield ("step", node, trace_line), ("token", text) while the graph runs, then ("final", AgentOutput)."""
        cfg = self.make_config(thread_id, callbacks, metadata)
        state = self.initial_state(task, cache)
        for mode, chunk in self.graph.stream(state, config=cfg, stream_mode=["updates", "custom", "values"]):
            if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "token":
                yield ("token", chunk["text"])
            elif mode == "updates":
                for node, upd in (chunk or {}).items():
                    tr = (upd or {}).get("trace") or []
                    yield ("step", node, tr[-1] if tr else node)
            elif mode == "values":
                state = chunk
        yield ("final", self.output_from_state(task, state))


def make_agent_llm(llm_cfg: dict):
    """LangChain ChatOllama for the Ollama backend; otherwise the existing Hugging Face wrapper."""
    if llm_cfg.get("backend") == "ollama":
        return LangChainChat(llm_cfg.get("ollama_model", "qwen2.5:3b"), llm_cfg.get("ollama_url", "http://localhost:11434"),
                             llm_cfg.get("max_new_tokens", 256), llm_cfg.get("num_ctx", 4096))
    from .llm import make_llm

    return make_llm(llm_cfg)


def to_prediction(task_json: dict, out: AgentOutput) -> dict:
    """MTRAG-format row (same as run_pipeline) plus an 'agent' block with the decision trace."""
    from .pipeline import Output, to_mtrag_prediction

    row = to_mtrag_prediction(task_json, Output(out.task_id, out.query, out.passages, out.p_unanswerable,
                                                out.abstained, out.answer, out.abstained_by, out.timings))
    row["agent"] = {"decision": out.decision, "attempts": out.attempts, "queries": out.queries, "trace": out.trace}
    return row


GateFn = Callable[[str, list], float]
