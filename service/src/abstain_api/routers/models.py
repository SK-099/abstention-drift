"""GET /models: every chat model the user can pick (model menu like ChatGPT / Claude.ai)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from ..auth import current_user
from ..model_registry import PROVIDERS, configured_providers, default_model_id, get_registry
from ..models import User
from ..schemas import ModelsOut
from ..settings import get_settings

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsOut, summary="Models available in the model picker")
async def list_models(refresh: bool = False, user: User = Depends(current_user)):
    s = get_settings()
    reg = get_registry(s)
    if refresh:
        reg._cache.clear()
    models = await asyncio.to_thread(reg.list)
    configured = {p.key for p in configured_providers(s)}
    errors = reg.errors()
    providers = [
        {"key": p.key, "label": p.label, "configured": p.key in configured, "error": errors.get(p.key)}
        for p in PROVIDERS.values()
    ]
    return {"default": default_model_id(s), "models": [m.public() for m in models], "providers": providers}


@router.post("/models/warm", summary="Load a local model into memory now (so the first answer after a switch is fast)")
async def warm_model(body: dict, user: User = Depends(current_user)):
    """With one small GPU only one local model fits at a time: switching models means Ollama must load the new
    one first (up to a few minutes). The menu calls this as soon as a model is picked, while the user types."""
    import time

    from fastapi import HTTPException, status

    from ..model_registry import ollama_base, split_id

    s = get_settings()
    mid = str(body.get("model") or "")
    try:
        mid = get_registry(s).resolve(mid)
        prov, name = split_id(mid)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    if prov != "local":
        return {"model": mid, "loaded": True, "seconds": 0.0}

    def load():
        import httpx

        t0 = time.perf_counter()
        r = httpx.post(
            f"{ollama_base(s)}/api/generate",
            json={
                "model": name,
                "prompt": "",
                "keep_alive": s.ollama_keep_alive,
                "options": {"num_ctx": s.ollama_min_ctx},
            },
            timeout=s.llm_timeout_s,
        )
        r.raise_for_status()
        return round(time.perf_counter() - t0, 1)

    try:
        secs = await asyncio.to_thread(load)
    except Exception as e:  # not Ollama (e.g. vLLM) or still loading: the chat will load it anyway
        return {"model": mid, "loaded": False, "seconds": None, "detail": str(e)[:200]}
    return {"model": mid, "loaded": True, "seconds": secs}
