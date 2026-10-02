"""Guardrails: rate limiting (slowapi), upload validation, and prompt-injection screening of uploaded text."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from .settings import get_settings


def _rate_key(request: Request) -> str:
    """Per user when logged in (set by auth), else per client IP."""
    return getattr(request.state, "user_id", None) or get_remote_address(request)


def make_limiter() -> Limiter:
    s = get_settings()
    storage = "memory://" if s.env == "test" or s.celery_eager else s.redis_url
    return Limiter(key_func=_rate_key, storage_uri=storage, headers_enabled=False)


limiter = make_limiter()


def validate_upload(file: UploadFile, size: int) -> None:
    s = get_settings()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in s.allowed_extensions:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported file type '{ext}'; allowed: {', '.join(s.allowed_extensions)}",
        )
    if size > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"file larger than {s.max_upload_mb} MB")
    if size == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")


# ---------------------------------------------------------------- prompt injection
# Uploaded documents become LLM context, so text that tries to instruct the model is screened out.
_PATTERNS = [
    # Weights: >= 0.5 alone is flagged for uploads (0.7 for chat questions); weaker cues (0.4) only count
    # together with another one (each extra match adds 0.15). Tuned on 24K MTRAG passages + 1,349 questions.
    (
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b.{0,30}\b(instruction|prompt|rule|message|direction)s?",
        0.9,
    ),
    (
        r"\byou are (now|no longer)\b(?!\s+(eligible|able|ready|enrolled|registered|logged|signed|connected|subscribed|covered))",
        0.4,
    ),
    (r"\b(act|behave|respond) as (an?|the) (unrestricted|jailbroken|different|new)\b", 0.7),
    (r"\b(system|developer) prompt\b", 0.4),
    (
        r"\b(reveal|print|show|repeat|leak|output)\b.{0,30}\b(your|the)\s+(system prompt|instructions|hidden prompt|initial prompt|prompt)\b",
        0.8,
    ),
    (r"\bdo not (tell|inform|mention to) the user\b", 0.7),
    (r"\b(jailbreak|DAN mode|developer mode)\b", 0.7),
    (r"<\|?(im_start|im_end|system|endoftext)\|?>|\[/?INST\]|<<SYS>>", 0.9),
    (r"^\s*#{2,}\s*(instruction|system|new task)s?\b", 0.6),
    (
        r"\b(send|exfiltrate|forward|transmit|leak)\b.{0,40}\b(the user'?s?|this conversation|chat history|all (messages|data)|credentials|secrets)\b.{0,60}\bhttps?://",
        0.8,
    ),
    (r"\b(always|instead) (answer|respond|reply|say)\b.{0,40}\b(regardless|no matter)\b", 0.6),
]
_RX = [(re.compile(p, re.IGNORECASE | re.MULTILINE), w) for p, w in _PATTERNS]


@lru_cache
def _classifier():
    name = get_settings().injection_model
    if not name:
        return None
    from transformers import pipeline

    return pipeline("text-classification", model=name, truncation=True, max_length=512)


def injection_score(text: str) -> float:
    """0..1 risk that the text contains instructions aimed at the model.
    Heuristic patterns; if INJECTION_MODEL is set, the max with a classifier score."""
    score = 0.0
    for rx, w in _RX:
        if rx.search(text):
            score = max(score, w) if score == 0 else min(1.0, max(score, w) + 0.15)
    clf = _classifier()
    if clf is not None:
        r = clf(text[:2000])[0]
        if str(r["label"]).upper() in {"INJECTION", "LABEL_1", "JAILBREAK"}:
            score = max(score, float(r["score"]))
    return score


def is_injection(text: str) -> bool:
    return injection_score(text) >= get_settings().injection_threshold


# ---------------------------------------------------------------- personal data and secrets (input and output)
def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n > 4 else n * 2
        total, alt = total + n, not alt
    return total % 10 == 0


_EXAMPLE_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "domain.com", "email.com")
_PII_RULES: list[tuple[str, re.Pattern, object]] = [
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
        None,
    ),
    (
        "api_key",
        re.compile(
            r"\b(?:AKIA[0-9A-Z]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36}|gsk_[A-Za-z0-9]{20,}"
            r"|xox[abprs]-[A-Za-z0-9-]{10,}|hf_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{35})\b"
        ),
        None,
    ),
    (
        "secret",
        re.compile(r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"]?([^\s'\"]{8,})"),
        "group1",
    ),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"), "email"),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), "luhn"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), None),
    ("aadhaar", re.compile(r"(?<![\d-])(?<!\d )[2-9]\d{3}\s\d{4}\s\d{4}(?! ?\d)"), None),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None),
    (
        "phone",
        # (lookarounds keep it from matching digit groups inside card numbers / longer digit runs)
        re.compile(
            r"(?<![\w.])(?<!\d[\s.-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{3,4}[\s.-]\d{3,4}"
            r"(?![\w.])(?![\s.-]?\d)"
        ),
        None,
    ),
]


def find_pii(text: str) -> list[tuple[str, int, int]]:
    """[(kind, start, end)] of personal data / secrets in the text (non-overlapping, first rule wins)."""
    hits: list[tuple[str, int, int]] = []
    taken: list[tuple[int, int]] = []
    for kind, rx, mode in _PII_RULES:
        for m in rx.finditer(text):
            start, end = m.span(1) if mode == "group1" else m.span()
            if mode == "email" and m.group(1).lower().endswith(_EXAMPLE_EMAIL_DOMAINS):
                continue
            if mode == "luhn":
                digits = re.sub(r"\D", "", m.group())
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    continue
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            hits.append((kind, start, end))
    return sorted(hits, key=lambda h: h[1])


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace personal data / secrets with [REDACTED_<KIND>]; returns (text, kinds found)."""
    hits = find_pii(text)
    for kind, start, end in reversed(hits):
        text = text[:start] + f"[REDACTED_{kind.upper()}]" + text[end:]
    return text, sorted({h[0] for h in hits})


# ---------------------------------------------------------------- toxicity / threats (input and output)
_TOX_PATTERNS = [
    (r"\b(i('| a)?m going to|i will|i'll|gonna)\s+(kill|murder|hurt|stab|shoot)\s+(you|u|him|her|them)\b", 0.95),
    (r"\b(kill|hang)\s+yourself\b|\bkys\b|\bgo\s+die\b", 0.95),
    (r"\byou(?:'re| are)?\s+(?:a\s+|an\s+|such\s+an?\s+)?(stupid|idiot|moron|worthless|pathetic|dumb)\b", 0.7),
    (r"\b(f+u+c+k+|f\*+k)\s+(you|off|u)\b", 0.85),
    (r"\bshut\s+up\b", 0.5),
]
_TOX_RX = [(re.compile(p, re.IGNORECASE), w) for p, w in _TOX_PATTERNS]


@lru_cache
def _toxicity_classifier():
    name = get_settings().toxicity_model
    if not name:
        return None
    from transformers import pipeline

    return pipeline("text-classification", model=name, truncation=True, max_length=512, top_k=None)


def toxicity_score(text: str) -> float:
    """0..1 toxicity / threat score: patterns, or the max with TOXICITY_MODEL (e.g. unitary/toxic-bert)."""
    score = max((w for rx, w in _TOX_RX if rx.search(text)), default=0.0)
    clf = _toxicity_classifier()
    if clf is not None:
        res = clf(text[:2000])
        res = res[0] if res and isinstance(res[0], list) else res
        bad = [
            float(r["score"])
            for r in res
            if str(r["label"]).lower() not in {"non-toxic", "neutral", "label_0", "not_toxic"}
        ]
        score = max([score, *bad])
    return score
