"""Prompt and threshold versions. Every answer records them (and Langfuse traces are tagged with them),
so a regression can be traced to a prompt or threshold change. scripts/push_prompts_to_langfuse.py
registers the same prompts in Langfuse prompt management."""

from __future__ import annotations

import hashlib
from functools import lru_cache

from abstain import agent as A
from abstain.generate import GEN_SYSTEM
from abstain.rewrite import REWRITE_SYSTEM

from .settings import Settings

PROMPTS = {
    "rewrite_system": REWRITE_SYSTEM,
    "generate_system": GEN_SYSTEM,
    "retry_rewrite_system": A.RETRY_REWRITE_SYSTEM,
    "self_check": A.SELF_CHECK_PROMPT,
    "ambiguity_check": A.AMBIGUOUS_PROMPT,
    "clarify": A.CLARIFY_PROMPT,
}


@lru_cache
def prompt_hash() -> str:
    h = hashlib.sha256()
    for k in sorted(PROMPTS):
        h.update(k.encode() + b"\0" + PROMPTS[k].encode() + b"\0")
    return h.hexdigest()[:12]


def versions(s: Settings) -> dict:
    return {
        "prompt_version": s.prompt_version,
        "prompt_hash": prompt_hash(),
        "llm_model": s.llm_model,
        "tau": round(s.resolved_gate_threshold(), 4),
        "self_check_min_p": round(s.resolved_min_p(), 4),
        "max_retries": s.max_retries,
    }
