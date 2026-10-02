"""Routing tests for the LangGraph agent with a scripted fake LLM, retriever and gate (no network)."""
import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from abstain.agent import AbstentionAgent, make_retriever
from abstain.data import Task
from abstain.idk import IDK_RESPONSE


class FakeLLM:
    """Answers by prompt type; `answers` and `checks` are consumed in order."""

    def __init__(self, answers=("The answer is 42.",), checks=("yes",), ambiguous="no"):
        self.answers, self.checks, self.ambiguous, self.calls = list(answers), list(checks), ambiguous, []

    def chat(self, messages, max_new_tokens=None):
        sys_ = messages[0]["content"]
        user = messages[-1]["content"]
        self.calls.append(user[:40])
        if "DIFFERENT standalone search query" in sys_:
            return "retry query"
        if "standalone search query" in sys_:
            return "first query"
        if "Proposed answer" in user:
            return self.checks.pop(0)
        if "too vague or ambiguous" in user:
            return self.ambiguous
        if "clarifying question" in user:
            return "Which product do you mean?"
        return self.answers.pop(0)


class FakeRetriever:
    def search(self, query, k=5):
        return [{"document_id": f"{query}-d{i}", "text": f"passage about {query}", "score": 0.9} for i in range(k)]


class Gate:
    """p depends on the query: weak for 'first query', strong for 'retry query'."""

    def __init__(self, weak=("first query",)):
        self.weak, self.threshold = set(weak), 0.5

    def __call__(self, query, passages):
        return 0.9 if query in self.weak else 0.1


def task():
    return Task("t1", "c1", 3, "cloud", [{"speaker": "user", "text": "hi"}, {"speaker": "agent", "text": "hello"}],
                "and the second one?", "ANSWERABLE")


def agent(llm, gate, **kw):
    return AbstentionAgent(llm, make_retriever({"cloud": FakeRetriever()}), gate, **kw)


def test_good_evidence_supported_answer():
    out = agent(FakeLLM(), Gate(weak=()))(task())
    assert out.decision == "answer" and not out.abstained and out.attempts == 0
    assert out.answer == "The answer is 42."


def test_weak_evidence_retry_recovers_answer():
    out = agent(FakeLLM(), Gate())(task())
    assert out.decision == "answer" and out.attempts == 1
    assert out.queries == ["first query", "retry query"]


def test_weak_after_retry_abstains_or_clarifies():
    out = agent(FakeLLM(), Gate(weak=("first query", "retry query")))(task())
    assert out.decision == "abstain" and out.abstained and out.answer == IDK_RESPONSE and out.abstained_by == "gate"
    out = agent(FakeLLM(ambiguous="yes"), Gate(weak=("first query", "retry query")))(task())
    assert out.decision == "clarify" and out.abstained and out.answer.endswith("?")


def test_self_check_rejects_unsupported_answer():
    out = agent(FakeLLM(checks=["no"]), Gate(weak=()))(task())
    assert out.decision == "abstain" and out.abstained_by == "self-check"


def test_generator_refusal_triggers_retry():
    out = agent(FakeLLM(answers=[IDK_RESPONSE, "Found it."]), Gate(weak=()))(task())
    assert out.decision == "answer" and out.attempts == 1 and out.answer == "Found it."


def test_cache_skips_first_attempt_llm_calls():
    llm = FakeLLM()
    cache = {"query": "cached q", "passages": [{"document_id": "x", "text": "t", "score": 0.8}], "p": 0.1,
             "answer": "Cached answer."}
    out = agent(llm, Gate(weak=()))(task(), cache=cache)
    assert out.answer == "Cached answer." and out.query == "cached q"
    assert all("Proposed answer" in c or c.startswith("Passages") for c in llm.calls)   # only the self-check ran


def test_selective_self_check_skips_confident_evidence():
    out = agent(FakeLLM(checks=["no"]), Gate(weak=()), self_check_min_p=0.5)(task())   # p=0.1 < 0.5
    assert out.decision == "answer" and "skipped" in out.trace[-1]
    out = agent(FakeLLM(checks=["no"]), Gate(weak=()), self_check_min_p=0.05)(task())  # p=0.1 >= 0.05
    assert out.decision == "abstain" and out.abstained_by == "self-check"


def test_identical_retry_query_skips_second_search():
    class SameQueryLLM(FakeLLM):
        def chat(self, messages, max_new_tokens=None):
            if "DIFFERENT standalone search query" in messages[0]["content"]:
                return "first query"                      # the "new" query repeats the first one
            return super().chat(messages, max_new_tokens)

    out = agent(SameQueryLLM(), Gate(weak=("first query",)))(task())
    assert out.decision == "abstain" and out.attempts == 1
    assert sum(t.startswith("retrieve") for t in out.trace) == 1     # no second retrieval


def test_clarify_min_turn_abstains_on_early_turns():
    weak = Gate(weak=("first query", "retry query"))
    out = agent(FakeLLM(ambiguous="yes"), weak, clarify_min_turn=4)(task())        # task() is turn 3
    assert out.decision == "abstain" and out.trace[-1] == "abstain"
    out = agent(FakeLLM(ambiguous="yes"), weak, clarify_min_turn=3)(task())
    assert out.decision == "clarify"
