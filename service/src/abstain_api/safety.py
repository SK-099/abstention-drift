"""Chat guardrails: prompt-injection, toxicity and personal-data/secret checks on every question and answer.

Engine: Guardrails AI (custom validators registered below, run by `guardrails.Guard`), or the same checks
called directly (`GUARDRAILS_ENGINE=builtin`, or automatically when the `guardrails-ai` package is absent).

    input  : prompt_injection -> block   toxicity -> block   secrets_pii -> redact before the LLM sees it
    output : toxicity -> block (answer replaced)             secrets_pii -> redact

Each triggered rule is reported as {"stage", "rule", "action", "detail"}; the chat endpoint streams them as
`guardrail` events, stores them with the message, counts them in Prometheus and tags the Langfuse trace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from .guardrails import injection_score, redact_pii, toxicity_score
from .settings import get_settings

log = logging.getLogger(__name__)
BLOCKED_INPUT = "I can't help with that request. Please rephrase your question about the documents."
BLOCKED_OUTPUT = "I'm sorry, but I can't share that answer."


@dataclass
class GuardResult:
    text: str
    blocked: bool = False
    events: list[dict] = field(default_factory=list)

    @property
    def reason(self) -> str | None:
        return next((e["rule"] for e in self.events if e["action"] == "blocked"), None)


# ---------------------------------------------------------------- checks (engine-independent)
def _check_injection(text: str):
    sc = injection_score(text)
    return (sc >= get_settings().chat_injection_threshold, f"score {sc:.2f}")


def _check_toxicity(text: str):
    sc = toxicity_score(text)
    return (sc >= get_settings().toxicity_threshold, f"score {sc:.2f}")


def _check_pii(text: str):
    fixed, kinds = redact_pii(text)
    return (bool(kinds), ", ".join(kinds), fixed)


# ---------------------------------------------------------------- Guardrails AI engine
@lru_cache
def _guards():
    """(input_guard, output_guard) built from custom Guardrails AI validators."""
    from guardrails import Guard
    from guardrails.validators import FailResult, PassResult, Validator, register_validator

    @register_validator(name="abstain/prompt_injection", data_type="string")
    class PromptInjection(Validator):
        def _validate(self, value, metadata):
            bad, detail = _check_injection(value)
            return FailResult(error_message=detail) if bad else PassResult()

    @register_validator(name="abstain/toxicity", data_type="string")
    class Toxicity(Validator):
        def _validate(self, value, metadata):
            bad, detail = _check_toxicity(value)
            return FailResult(error_message=detail) if bad else PassResult()

    @register_validator(name="abstain/secrets_pii", data_type="string")
    class SecretsPII(Validator):
        def _validate(self, value, metadata):
            found, kinds, fixed = _check_pii(value)
            return FailResult(error_message=kinds, fix_value=fixed) if found else PassResult()

    s = get_settings()
    inp = [Toxicity(on_fail="noop"), SecretsPII(on_fail="fix")]
    if s.guard_chat_injection:
        inp.insert(0, PromptInjection(on_fail="noop"))
    return (
        Guard(name="chat-input").use(*inp),
        Guard(name="chat-output").use(Toxicity(on_fail="noop"), SecretsPII(on_fail="fix")),
    )


_RULE = {"PromptInjection": "prompt_injection", "Toxicity": "toxicity", "SecretsPII": "secrets_pii"}
_ACTION = {"prompt_injection": "blocked", "toxicity": "blocked", "secrets_pii": "redacted"}


def _run_guardrails_ai(guard, text: str, stage: str) -> GuardResult:
    out = guard.validate(text)
    events = [
        {
            "stage": stage,
            "rule": _RULE.get(sm.validator_name, sm.validator_name),
            "action": _ACTION.get(_RULE.get(sm.validator_name, ""), "flagged"),
            "detail": sm.failure_reason or "",
        }
        for sm in (out.validation_summaries or [])
        if str(sm.validator_status) == "fail"
    ]
    fixed = out.validated_output if isinstance(out.validated_output, str) else text
    return GuardResult(fixed, any(e["action"] == "blocked" for e in events), events)


# ---------------------------------------------------------------- builtin engine
def _run_builtin(text: str, stage: str) -> GuardResult:
    s, events = get_settings(), []
    if stage == "input" and s.guard_chat_injection:
        bad, detail = _check_injection(text)
        if bad:
            events.append({"stage": stage, "rule": "prompt_injection", "action": "blocked", "detail": detail})
    bad, detail = _check_toxicity(text)
    if bad:
        events.append({"stage": stage, "rule": "toxicity", "action": "blocked", "detail": detail})
    found, kinds, fixed = _check_pii(text)
    if found:
        events.append({"stage": stage, "rule": "secrets_pii", "action": "redacted", "detail": kinds})
        text = fixed
    return GuardResult(text, any(e["action"] == "blocked" for e in events), events)


# ---------------------------------------------------------------- public API
@lru_cache
def _guardrails_ai_available() -> bool:
    try:
        _guards()
        return True
    except ImportError:
        log.warning("guardrails-ai is not installed; using the builtin guardrail engine")
        return False


def engine() -> str:
    s = get_settings()
    if s.guardrails == "off":
        return "off"
    if s.guardrails_engine == "guardrails-ai" and _guardrails_ai_available():
        return "guardrails-ai"
    return "builtin"


def _run(text: str, stage: str) -> GuardResult:
    eng = engine()
    if eng == "off" or not text:
        return GuardResult(text)
    try:
        if eng == "guardrails-ai":
            return _run_guardrails_ai(_guards()[0 if stage == "input" else 1], text, stage)
        return _run_builtin(text, stage)
    except Exception:  # a broken check must not take the chat down
        log.exception("guardrail check failed (%s); passing the text through", stage)
        return GuardResult(text)


def check_input(text: str) -> GuardResult:
    return _run(text, "input")


def check_output(text: str) -> GuardResult:
    res = _run(text, "output")
    if res.blocked:
        res.text = BLOCKED_OUTPUT
    return res


def guard_final(final: dict) -> tuple[dict, list[dict]]:
    """Apply the output guardrails to the agent's final payload (used inside the agent thread)."""
    if final.get("decision") not in ("answer", "clarify"):  # refusals are fixed text
        return final, []
    res = check_output(final["answer"])
    if not res.events:
        return final, []
    final = dict(final, answer=res.text)
    if res.blocked:
        final.update(decision="blocked", abstained_by="guardrail", citations=[])
    return final, res.events
