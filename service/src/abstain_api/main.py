"""FastAPI application: abstention-aware multi-turn RAG service around the S5 LangGraph agent.

    uvicorn abstain_api.main:app --host 0.0.0.0 --port 8000      (or: abstain-api)
OpenAPI docs at /docs.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import select

from . import __version__
from .db import async_sessionmaker_
from .guardrails import limiter
from .models import Collection
from .routers import chat, collections, conversations, feedback, health, metrics
from .settings import get_settings
from .telemetry import setup_metrics, setup_sentry

log = logging.getLogger("abstain_api")
BUILTIN_COLLECTIONS = {
    "clapnq": ("ClapNQ (Wikipedia)", "Wikipedia passages from the MTRAG benchmark"),
    "cloud": ("IBM Cloud docs", "IBM Cloud documentation from the MTRAG benchmark"),
    "fiqa": ("FiQA (finance forums)", "Personal-finance Q&A from the MTRAG benchmark"),
    "govt": ("Government sites", "US government web pages from the MTRAG benchmark"),
}


async def seed_collections() -> list[str]:
    async with async_sessionmaker_()() as s:
        have = set((await s.execute(select(Collection.id))).scalars().all())
        for cid, (name, desc) in BUILTIN_COLLECTIONS.items():
            if cid not in have:
                s.add(Collection(id=cid, name=name, description=desc, is_public=True))
        await s.commit()
        return sorted(have | set(BUILTIN_COLLECTIONS))


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    if s.env != "production" and s.database_url.startswith("sqlite"):  # dev/test convenience; prod uses Alembic
        from .db import Base, async_engine

        async with async_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    cols = list(BUILTIN_COLLECTIONS)
    try:
        cols = await seed_collections()
    except Exception:
        log.exception("could not seed collections (did you run `alembic upgrade head`?)")
    from .telemetry import get_metrics

    get_metrics().prime(cols)  # counters start at 0 so the first event after a restart is counted
    yield
    from .tracing import flush

    flush()


def create_app() -> FastAPI:
    s = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    setup_sentry(s, "api")
    setup_metrics()
    app = FastAPI(
        title="Abstention-aware RAG API",
        version=__version__,
        lifespan=lifespan,
        description="Multi-turn RAG that knows when not to answer (CS683 project, S5 agent).",
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware, allow_origins=s.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )
    for r in (health, chat, conversations, feedback, collections, metrics):
        app.include_router(r.router)
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,healthz,readyz")
    except Exception:
        log.warning("OpenTelemetry FastAPI instrumentation unavailable")
    return app


app = create_app()


def run() -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "abstain_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=s.env == "dev",
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
