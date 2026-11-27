"""FastAPI plumbing shared by every service: authn dependency, error mapping, metrics, health."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import jwt
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from agentic.common import logging as clog
from agentic.common.errors import AgenticError, AuthError, ForbiddenError
from agentic.common.metrics import HTTP_LATENCY, HTTP_REQUESTS
from agentic.common.schemas import Principal
from agentic.config import Settings
from agentic.trust.identity import IdentityProvider

log = logging.getLogger(__name__)


class Authenticator:
    """Accepts: per-agent identity tokens, OIDC/HS256 user JWTs, and hashed service API keys."""

    def __init__(self, settings: Settings, identity: IdentityProvider) -> None:
        self.s = settings
        self.identity = identity
        self._jwks = jwt.PyJWKClient(settings.jwks_url, cache_keys=True) if settings.jwks_url else None
        self._api_keys = settings.api_keys()

    async def authenticate(self, authorization: str | None, api_key: str | None = None) -> Principal:
        if api_key:
            rec = self._api_keys.get(hashlib.sha256(api_key.encode()).hexdigest())
            if not rec:
                raise AuthError("invalid API key")
            return Principal(
                sub=rec["sub"], tenant=rec.get("tenant", "default"),
                scopes=rec.get("scopes", ["agent:run"]), kind="service",
            )
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AuthError("missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if self.identity.looks_like_identity(token):
            return self.identity.verify(token)
        return await self._verify_user_jwt(token)

    async def _verify_user_jwt(self, token: str) -> Principal:
        import asyncio

        s = self.s
        try:
            if self._jwks:
                key = (await asyncio.to_thread(self._jwks.get_signing_key_from_jwt, token)).key
            else:
                key = s.jwt_secret
            claims = jwt.decode(
                token, key, algorithms=s.jwt_algorithm_list, issuer=s.jwt_issuer,
                audience=s.jwt_audience,
                options={"require": ["exp", "sub"], "verify_aud": bool(s.jwt_audience),
                         "verify_iss": bool(s.jwt_issuer)},
            )
        except jwt.PyJWTError as e:
            raise AuthError(f"invalid token: {e}") from e
        scopes = claims.get("scopes") or (claims.get("scp") or "").split() or claims.get("roles") or []
        return Principal(
            sub=str(claims["sub"]), tenant=str(claims.get("tenant") or claims.get("tid") or "default"),
            scopes=list(scopes), kind=claims.get("kind", "user"),
            on_behalf_of=claims.get("obo"),
        )


def principal_dependency(authn: Authenticator) -> Callable[[Request], Awaitable[Principal]]:
    async def dep(request: Request) -> Principal:
        p = await authn.authenticate(request.headers.get("authorization"), request.headers.get("x-api-key"))
        request.state.principal = p
        clog.bind(tenant=p.tenant, sub=p.sub)
        return p

    return dep


def require_scope(scope: str, dep: Callable[..., Any]) -> Callable[..., Awaitable[Principal]]:
    async def checker(p: Principal = Depends(dep)) -> Principal:
        if not p.has(scope):
            raise ForbiddenError(f"missing scope {scope}")
        return p

    return checker


class ObservabilityMiddleware:
    """Request id, structured-log context, Prometheus RED metrics."""

    def __init__(self, app: Any, service: str) -> None:
        self.app, self.service = app, service

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        rid = headers.get(b"x-request-id", b"").decode() or uuid.uuid4().hex
        clog.clear()
        clog.bind(request_id=rid, service=self.service)
        start, status = time.perf_counter(), 500

        async def send_wrap(msg: dict) -> None:
            nonlocal status
            if msg["type"] == "http.response.start":
                status = msg["status"]
                msg.setdefault("headers", []).append((b"x-request-id", rid.encode()))
            await send(msg)

        try:
            await self.app(scope, receive, send_wrap)
        finally:
            route = scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            HTTP_REQUESTS.labels(self.service, path, str(status)).inc()
            HTTP_LATENCY.labels(self.service, path).observe(time.perf_counter() - start)


def instrument_app(app: FastAPI, service: str, ready: Callable[[], Awaitable[dict[str, bool]]] | None = None) -> None:
    app.add_middleware(ObservabilityMiddleware, service=service)

    @app.exception_handler(AgenticError)
    async def _agentic(_: Request, e: AgenticError) -> JSONResponse:
        hdrs = {"Retry-After": str(e.details.get("retry_after", 30))} if e.status_code == 429 else None
        return JSONResponse(
            {"error": {"code": e.code, "message": e.message, "details": e.details}},
            status_code=e.status_code, headers=hdrs,
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse({"error": {"code": "internal_error", "message": "internal error"}}, status_code=500)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": service}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        checks = await ready() if ready else {}
        ok = all(checks.values())
        return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def new_client(base_url: str = "", timeout: float = 60.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url, timeout=httpx.Timeout(timeout, connect=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
