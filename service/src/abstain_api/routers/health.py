from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .. import __version__
from ..db import get_session
from ..prompts import versions
from ..settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness")
async def healthz():
    return {"status": "ok", "version": __version__}


@router.get("/readyz", summary="Readiness: database, Qdrant, Redis and the LLM endpoint")
async def readyz(session: AsyncSession = Depends(get_session)):
    s = get_settings()
    checks = {}
    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e.__class__.__name__}"

    def qdrant():
        from qdrant_client import QdrantClient

        if s.qdrant_url == ":memory:":
            return "ok"
        QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=3).get_collections()
        return "ok"

    def redis_():
        if s.celery_eager:
            return "skipped (eager)"
        import redis

        redis.Redis.from_url(s.redis_url, socket_timeout=3).ping()
        return "ok"

    def llm():
        import httpx

        r = httpx.get(
            s.llm_base_url.rstrip("/") + "/models",
            timeout=3,
            headers={"Authorization": f"Bearer {s.llm_api_key}"} if s.llm_api_key else {},
        )
        return "ok" if r.status_code < 500 else f"http {r.status_code}"

    for name, fn in (("qdrant", qdrant), ("redis", redis_), ("llm", llm)):
        try:
            checks[name] = await asyncio.to_thread(fn)
        except Exception as e:
            checks[name] = f"error: {e.__class__.__name__}"
    ok = checks["database"] == "ok" and checks["qdrant"] == "ok"
    return JSONResponse({"ready": ok, "checks": checks, "versions": versions(s)}, status_code=200 if ok else 503)
