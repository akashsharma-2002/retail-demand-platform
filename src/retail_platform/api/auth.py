"""OIDC bearer-token verification (Keycloak) and role checks."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from retail_platform.config import get_settings
from retail_platform.services.planning import Actor

ROLE_ORDER = ["viewer", "planner", "approver", "admin"]
bearer = HTTPBearer(auto_error=False)


@dataclass
class TokenVerifier:
    issuer: str
    audience: str
    key_resolver: Callable[[str], Any]

    def verify(self, token: str) -> dict:
        key = self.key_resolver(token)
        return jwt.decode(
            token,
            key=key,
            algorithms=["RS256"],
            audience=self.audience,
            issuer=self.issuer,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )


@lru_cache
def get_verifier() -> TokenVerifier:
    settings = get_settings()
    client = jwt.PyJWKClient(settings.oidc_jwks_url, cache_keys=True, lifespan=300)
    return TokenVerifier(
        settings.oidc_issuer, settings.oidc_audience, lambda token: client.get_signing_key_from_jwt(token).key
    )


def effective_roles(claims: dict) -> frozenset[str]:
    """Keycloak puts realm roles under realm_access.roles. Higher roles include lower ones."""
    granted = set(claims.get("realm_access", {}).get("roles", [])) & set(ROLE_ORDER)
    if not granted:
        return frozenset()
    top = max(ROLE_ORDER.index(r) for r in granted)
    implied = set(ROLE_ORDER[: top + 1])
    # approval is a separate duty: planners are not approvers unless granted
    if "approver" not in granted and "admin" not in granted:
        implied.discard("approver")
    return frozenset(implied | granted)


def current_actor(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    verifier: TokenVerifier = Depends(get_verifier),
) -> Actor:
    if credentials is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    try:
        claims = verifier.verify(credentials.credentials)
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, f"Invalid token: {exc}", headers={"WWW-Authenticate": "Bearer"}
        ) from exc
    subject = claims.get("preferred_username") or claims["sub"]
    request.state.subject = subject
    return Actor(subject=subject, roles=effective_roles(claims), request_id=getattr(request.state, "request_id", None))


def require(role: str) -> Callable[[Actor], Actor]:
    def dependency(actor: Actor = Depends(current_actor)) -> Actor:
        if role not in actor.roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires role: {role}")
        return actor

    return dependency
