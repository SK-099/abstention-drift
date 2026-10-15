"""All configuration comes from environment variables (or a .env file); see .env.example."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["dev", "test", "production"] = "dev"
    app_name: str = "Grounded Assistant"
    cors_origins: list[str] = ["http://localhost:3000"]

    # --- database (async URL for the API; the worker derives a sync URL)
    database_url: str = "postgresql+asyncpg://app:app@localhost:5432/app"
    # LangGraph checkpointer: "postgres" (persistent, resumable threads) or "memory"
    checkpointer: Literal["postgres", "memory"] = "postgres"

    # --- LLM: any OpenAI-compatible endpoint (Ollama /v1, vLLM, Groq, Together, OpenRouter)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:3b"
    llm_api_key: str = "ollama"
    llm_max_tokens: int = 320  # answer length cap (the research runs used 128)
    llm_timeout_s: float = 120.0
    llm_seed: int | None = 13
    # --- model picker: the user chooses any model listed by GET /models. Providers appear when configured.
    # Model ids are "<provider>:<model>", e.g. "ollama:qwen2.5:3b", "openrouter:anthropic/claude-sonnet-4.5".
    default_model: str | None = None  # default: "local:" + LLM_MODEL (the calibrated research model)
    ollama_url: str | None = None  # Ollama server for model discovery (default: derived from LLM_BASE_URL)
    openrouter_api_key: str | None = None  # one key: Claude, GPT, Gemini, Llama, Qwen, DeepSeek, Mistral, ...
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    mistral_api_key: str | None = None
    deepseek_api_key: str | None = None
    together_api_key: str | None = None
    model_allowlist: list[str] = []  # optional glob patterns ("openrouter:anthropic/*"); empty = everything
    model_blocklist: list[str] = []
    models_cache_s: int = 600  # provider model lists are re-fetched after this many seconds
    reasoning_max_tokens: int = 4000  # completion budget for reasoning models (o-series, gpt-5, thinking models)
    reasoning_timeout_s: float = 600.0

    # --- retrieval
    vector_store: Literal["qdrant", "pgvector"] = "qdrant"  # pgvector: same hybrid search inside PostgreSQL
    qdrant_url: str = "http://localhost:6333"  # ":memory:" for tests
    qdrant_api_key: str | None = None
    qdrant_collection: str = "passages"
    dense_model: str = "BAAI/bge-small-en-v1.5"
    dense_dim: int = 384
    reranker_model: str = "BAAI/bge-reranker-base"
    tei_embed_url: str | None = None  # e.g. http://tei-embed:80 (else in-process)
    tei_rerank_url: str | None = None  # e.g. http://tei-rerank:80
    prefetch_k: int = 100
    rerank_top_n: int = 15
    final_k: int = 5
    bm25_avgdl: float = 220.0  # average passage length in tokens (sparse BM25 weights)
    device: str | None = None  # "cpu", "cuda" or None (auto)

    # --- answerability gate (the fine-tuned S4 checker) and agent
    checker_path: str = "checkpoints/checker/best"
    tei_checker_url: str | None = None  # serve the fine-tuned checker with TEI instead
    gate_threshold: float | None = None  # default: checker_path/threshold.json
    self_check_min_p: float | None = None  # default: checkpoints/self_check.json
    self_check_file: str = "checkpoints/self_check.json"
    calibration_dir: str = "checkpoints/calibration"  # per-model tau / min_p (service/scripts/calibrate_models.py)
    max_retries: int = 1
    self_check: bool = True
    clarify: bool = True
    clarify_min_turn: int = 1  # 2 = never ask a clarifying question on a conversation's first turn
    history_turns: int = 3
    history_chars: int = 400
    passage_chars: int = 1000
    prompt_version: str = "s5-v1"

    # --- background jobs
    redis_url: str = "redis://localhost:6379/0"
    celery_eager: bool = False  # run tasks inline (tests)
    upload_dir: str = "data/uploads"
    max_upload_mb: int = 20
    allowed_extensions: list[str] = [".pdf", ".html", ".htm", ".docx", ".md", ".txt"]
    parser: Literal["auto", "docling", "basic"] = "auto"
    chunk_words: int = 150  # ~1,000 characters: fits PASSAGE_CHARS, so the LLM reads every retrieved chunk whole
    chunk_overlap: int = 30
    nightly_eval_hour_utc: int = 2

    # --- auth: Clerk (jwks), Auth.js (hs256 shared secret) or dev (no login; refused in production)
    auth_mode: Literal["dev", "hs256", "jwks"] = "dev"
    auth_jwks_url: str | None = None  # Clerk: https://<your-app>.clerk.accounts.dev/.well-known/jwks.json
    auth_issuer: str | None = None
    auth_audience: str | None = None
    auth_secret: str | None = None  # Auth.js: AUTH_SECRET used to sign an HS256 token
    auth_authorized_parties: list[str] = []
    admin_user_ids: list[str] = []

    # --- guardrails
    rate_limit_chat: str = "20/minute"
    rate_limit_upload: str = "30/hour"
    max_message_chars: int = 2000
    injection_threshold: float = 0.5
    # chat guardrails (safety.py): Guardrails AI validators, or the same checks without the library
    guardrails: Literal["on", "off"] = "on"
    guardrails_engine: Literal["guardrails-ai", "builtin"] = "guardrails-ai"
    guard_chat_injection: bool = True
    chat_injection_threshold: float = 0.7  # stricter than uploads: users may ask *about* prompts
    toxicity_model: str | None = None  # e.g. unitary/toxic-bert (else threat/insult patterns)
    toxicity_threshold: float = 0.8
    guardrails_buffer_output: bool = False  # true: don't stream tokens, send only the checked answer
    injection_model: str | None = None  # e.g. protectai/deberta-v3-base-prompt-injection-v2

    # --- notebook sources and generated content (NotebookLM-style features)
    source_guides: bool = True  # summary, topics and suggested questions for every new source
    guide_chars: int = 6000  # text of a source shown to the LLM for its guide (small local models: ~1.5k tokens)
    notebook_context_chars: int = 7000  # material per Studio / overview prompt
    fetch_timeout_s: float = 20.0  # web pages and YouTube transcripts
    allow_private_urls: bool = False  # never fetch localhost / private network addresses (SSRF)
    tts_url: str | None = None  # Kokoro-FastAPI (OpenAI-compatible /v1/audio/speech), e.g. http://kokoro:8880
    tts_voice_a: str = "af_heart"  # host "Alex"
    tts_voice_b: str = "am_michael"  # host "Sam"
    tts_speed: float = 1.0
    tts_sample_rate: int = 24000  # Kokoro's output rate (raw PCM is requested)
    tts_timeout_s: float = 300.0
    # "discover sources": web search for a topic (duckduckgo needs no key)
    search_provider: Literal["duckduckgo", "tavily", "brave", "searxng", "off"] = "duckduckgo"
    search_api_key: str | None = None  # Tavily / Brave
    searxng_url: str | None = None
    generation_model: str | None = None
    generation_timeout_s: float = 900.0  # guides / Studio / scripts: long prompts, Ollama may be busy with a chat
    warmup: bool = True  # load the embedder, re-ranker and gate when the API starts, not on the first question  # model for guides / Studio / audio scripts (default: DEFAULT_MODEL)

    # --- observability
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None
    sentry_dsn: str | None = None
    sentry_traces_sample_rate: float = 0.1

    # --- evaluation (CI quality gate and nightly job)
    mtrag_dir: str = "data/mt-rag-benchmark"
    eval_sample_file: str = "service/eval/sample_ids.json"
    eval_baseline_file: str = "service/eval/baseline.json"

    @field_validator("database_url")
    @classmethod
    def _async_driver(cls, v: str) -> str:
        """Accept plain postgres:// URLs from managed providers (Render, Neon, RDS, Cloud SQL)."""
        for prefix in ("postgres://", "postgresql://"):
            if v.startswith(prefix):
                return "postgresql+asyncpg://" + v[len(prefix) :]
        return v

    @model_validator(mode="after")
    def _safety(self):
        if self.env == "production" and self.auth_mode == "dev":
            raise ValueError("AUTH_MODE=dev is not allowed when ENV=production")
        if self.auth_mode == "jwks" and not self.auth_jwks_url:
            raise ValueError("AUTH_MODE=jwks needs AUTH_JWKS_URL")
        if self.auth_mode == "hs256" and not self.auth_secret:
            raise ValueError("AUTH_MODE=hs256 needs AUTH_SECRET")
        return self

    @property
    def sync_database_url(self) -> str:
        """psycopg/sqlite URL for Alembic and the Celery worker."""
        u = self.database_url
        return u.replace("+asyncpg", "+psycopg").replace("+aiosqlite", "")

    @property
    def checkpoint_conninfo(self) -> str:
        """libpq connection string for the LangGraph Postgres checkpointer."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")

    def resolved_gate_threshold(self) -> float:
        if self.gate_threshold is not None:
            return self.gate_threshold
        f = Path(self.checker_path) / "threshold.json"
        return float(json.loads(f.read_text())["threshold"]) if f.exists() else 0.611

    def resolved_min_p(self) -> float:
        if self.self_check_min_p is not None:
            return self.self_check_min_p
        f = Path(self.self_check_file)
        return float(json.loads(f.read_text())["min_p"]) if f.exists() else 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
