"""POST /collections/{id}/discover: search the web for a topic to find new sources (the user picks which to add)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user
from ..db import get_session
from ..guardrails import limiter
from ..models import User
from ..schemas import DiscoverHit, DiscoverIn
from ..settings import get_settings
from ._access import writable_collection

router = APIRouter(tags=["discover"])


@router.post(
    "/collections/{collection_id}/discover",
    response_model=list[DiscoverHit],
    summary="Search the web for sources about a topic",
)
@limiter.limit(lambda: get_settings().rate_limit_upload)
async def discover(
    request: Request,
    collection_id: str,
    body: DiscoverIn,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    from .. import search

    c = await writable_collection(session, collection_id, user)
    if c.is_public:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "add sources to your own notebooks")
    try:
        return await asyncio.to_thread(search.web_search, body.query, body.max_results, get_settings())
    except search.SearchUnavailable as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
