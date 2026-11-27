"""Per-agent identity (trust layer: "Identity — per agent").

Every agent node runs under its own short-lived, narrowly scoped identity token. The gateway
and the MCP tool server authenticate these tokens and enforce scopes, so a compromised or
prompt-injected research agent physically cannot call a write tool.

In production, map `issue()` to your workload identity system (Entra Workload ID / SPIFFE) —
the verifying side only needs the claims contract below.
"""

from __future__ import annotations

import time
import uuid

import jwt

from agentic.common.errors import AuthError
from agentic.common.schemas import Principal

IDENTITY_ISSUER = "agentic-identity"

# Least-privilege scope matrix. Supervisor can only talk to the LLM; only `action` may write.
AGENT_SCOPES: dict[str, list[str]] = {
    "supervisor": ["llm:invoke"],
    "research": ["llm:invoke", "knowledge:read", "tools:read"],
    "action": ["llm:invoke", "tools:read", "tools:write"],
    "review": ["llm:invoke", "knowledge:read"],
    "guard": ["llm:invoke"],
    "memory": ["llm:invoke", "knowledge:read"],
    "gateway": ["agent:run", "approvals:read", "approvals:decide", "memory:delete"],
    "executor": ["tools:read", "tools:write"],  # executes human-approved actions only
    "channel": ["agent:run", "approvals:read", "approvals:decide", "memory:delete"],
    "toolserver": ["internal:read", "internal:write"],
    "ingest": ["llm:invoke", "knowledge:write"],
}


class IdentityProvider:
    def __init__(self, secret: str, ttl_seconds: int = 300) -> None:
        self._secret = secret
        self._ttl = ttl_seconds

    def issue(
        self, agent_id: str, tenant: str, on_behalf_of: str | None = None,
        extra_scopes: list[str] | None = None, ttl: int | None = None,
    ) -> str:
        if agent_id not in AGENT_SCOPES:
            raise AuthError(f"unknown agent identity {agent_id!r}")
        now = int(time.time())
        claims = {
            "iss": IDENTITY_ISSUER,
            "sub": agent_id,
            "tenant": tenant,
            "scopes": sorted(set(AGENT_SCOPES[agent_id]) | set(extra_scopes or [])),
            "obo": on_behalf_of,
            "iat": now,
            "exp": now + (ttl or self._ttl),
            "jti": uuid.uuid4().hex,
        }
        return jwt.encode(claims, self._secret, algorithm="HS256")

    def verify(self, token: str) -> Principal:
        try:
            c = jwt.decode(
                token, self._secret, algorithms=["HS256"], issuer=IDENTITY_ISSUER,
                options={"require": ["exp", "iss", "sub"]},
            )
        except jwt.PyJWTError as e:
            raise AuthError(f"invalid agent identity: {e}") from e
        return Principal(
            sub=c["sub"], tenant=c.get("tenant", "default"), scopes=c.get("scopes", []),
            kind="agent", on_behalf_of=c.get("obo"),
        )

    @staticmethod
    def looks_like_identity(token: str) -> bool:
        try:
            return jwt.decode(token, options={"verify_signature": False}).get("iss") == IDENTITY_ISSUER
        except jwt.PyJWTError:
            return False
