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
