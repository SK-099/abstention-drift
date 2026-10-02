"""JWT authentication. The front end logs users in with Clerk or Auth.js; FastAPI only verifies tokens.

AUTH_MODE=jwks   Clerk (or any OIDC provider): RS256 tokens checked against the provider's JWKS.
AUTH_MODE=hs256  Auth.js: the front end signs a short-lived HS256 JWT with AUTH_SECRET.
AUTH_MODE=dev    no login; every request is the local admin user (refused when ENV=production).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_session
from .models import User
from .settings import get_settings


@dataclass
class Principal:
    sub: str
    email: str | None
    is_admin: bool


@lru_cache
def _jwks_client(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(url, cache_keys=True, lifespan=3600)


def _claims_admin(claims: dict) -> bool:
    for c in (claims, claims.get("metadata") or {}, claims.get("public_metadata") or {}):
        role = c.get("role")
        if role == "admin" or (isinstance(role, list) and "admin" in role):
            return True
    return False


def verify_token(token: str) -> Principal:
    s = get_settings()
    opts = {"require": ["exp", "sub"]}
    try:
        if s.auth_mode == "jwks":
            key = _jwks_client(s.auth_jwks_url).get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256", "ES256"],
                audience=s.auth_audience,
                issuer=s.auth_issuer,
                options={**opts, "verify_aud": bool(s.auth_audience)},
            )
            if s.auth_authorized_parties and claims.get("azp") not in s.auth_authorized_parties:
                raise jwt.InvalidTokenError("unauthorized party")
        elif s.auth_mode == "hs256":
            claims = jwt.decode(
                token,
                s.auth_secret,
                algorithms=["HS256"],
                audience=s.auth_audience,
                issuer=s.auth_issuer,
                options={**opts, "verify_aud": bool(s.auth_audience)},
            )
        else:
            raise jwt.InvalidTokenError("token verification is disabled in dev mode")
    except jwt.PyJWTError as e:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, f"invalid token: {e}", headers={"WWW-Authenticate": "Bearer"}
        ) from e
    sub = str(claims["sub"])
    return Principal(sub, claims.get("email"), _claims_admin(claims) or sub in s.admin_user_ids)


def principal_from_request(request: Request) -> Principal:
    s = get_settings()
    if s.auth_mode == "dev":
        return Principal("dev-user", "dev@localhost", True)
    h = request.headers.get("authorization", "")
    if not h.lower().startswith("bearer "):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    return verify_token(h[7:].strip())


async def current_user(request: Request, session: AsyncSession = Depends(get_session)) -> User:
    p = principal_from_request(request)
    user = (await session.execute(select(User).where(User.external_id == p.sub))).scalar_one_or_none()
    if user is None:
        user = User(external_id=p.sub, email=p.email, is_admin=p.is_admin)
        session.add(user)
        await session.commit()
    elif user.is_admin != p.is_admin or (p.email and user.email != p.email):
        user.is_admin, user.email = p.is_admin, p.email or user.email
        await session.commit()
    request.state.user_id = user.id
    return user


async def admin_user(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return user
