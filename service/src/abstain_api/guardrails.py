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
    (
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b.{0,30}\b(instruction|prompt|rule|message|direction)s?",
        0.9,
    ),
    (r"\byou are (now|no longer)\b", 0.6),
    (r"\b(act|behave|respond) as (an?|the) (unrestricted|jailbroken|different|new)\b", 0.7),
    (r"\b(system|developer) prompt\b", 0.5),
    (r"\b(reveal|print|show|repeat|leak)\b.{0,30}\b(system prompt|instructions|hidden|secret|api key|password)", 0.8),
    (r"\bdo not (tell|inform|mention to) the user\b", 0.7),
    (r"\b(jailbreak|DAN mode|developer mode)\b", 0.7),
    (r"<\|?(im_start|im_end|system|endoftext)\|?>|\[/?INST\]|<<SYS>>", 0.9),
    (r"^\s*#{2,}\s*(instruction|system|new task)s?\b", 0.6),
    (r"\b(send|post|exfiltrate|upload)\b.{0,40}\b(https?://|data to)\b", 0.6),
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
