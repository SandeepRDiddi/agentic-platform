"""Microsoft Teams (Bot Framework) channel helpers."""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import jwt

from agentic.common.errors import AuthError

JWKS_URL = "https://login.botframework.com/v1/.well-known/keys"
ISSUER = "https://api.botframework.com"
TOKEN_URL = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"  # noqa: S105
TRUSTED_SERVICE_SUFFIXES = (".botframework.com", ".botframework.us", ".trafficmanager.net")


def trusted_service_url(url: str) -> bool:
    """Never send our bot token to a host an attacker-controlled activity names (SSRF/token theft)."""
    u = urlparse(url)
    return u.scheme == "https" and bool(u.hostname) and any(u.hostname.endswith(s) for s in TRUSTED_SERVICE_SUFFIXES)


class TeamsJwtVerifier:
    def __init__(self, app_id: str | None, jwks: jwt.PyJWKClient | None = None) -> None:
        self.app_id = app_id
        self._jwks = jwks or jwt.PyJWKClient(JWKS_URL, cache_keys=True)

    def verify(self, authorization: str | None) -> dict[str, Any]:
        if not self.app_id:
            raise AuthError("teams app id not configured")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AuthError("missing bearer token")
        token = authorization.split(" ", 1)[1]
        try:
            key = self._jwks.get_signing_key_from_jwt(token).key
            return jwt.decode(token, key, algorithms=["RS256"], audience=self.app_id, issuer=ISSUER,
                              options={"require": ["exp", "aud", "iss"]})
        except jwt.PyJWTError as e:
            raise AuthError(f"invalid teams token: {e}") from e


def clean_text(text: str) -> str:
    return re.sub(r"<at>.*?</at>", "", text or "").strip()


def approval_card(a: dict[str, Any]) -> dict[str, Any]:
    return {
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": {
            "type": "AdaptiveCard", "version": "1.4", "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "body": [{"type": "TextBlock", "weight": "Bolder", "text": f"Approval needed: {a['tool']}"},
                     {"type": "TextBlock", "wrap": True, "text": str(a["args"])}],
            "actions": [
                {"type": "Action.Submit", "title": "Approve", "data": {"approval_id": a["id"], "approve": True}},
                {"type": "Action.Submit", "title": "Deny", "data": {"approval_id": a["id"], "approve": False}}],
        },
    }


class BotTokenProvider:
    def __init__(self, app_id: str | None, secret: str | None) -> None:
        self.app_id, self.secret = app_id, secret
        self._tok: str | None = None
        self._exp = 0.0

    async def get(self, http: httpx.AsyncClient) -> str:
        if self._tok and time.time() < self._exp - 60:
            return self._tok
        r = await http.post(TOKEN_URL, data={
            "grant_type": "client_credentials", "client_id": self.app_id, "client_secret": self.secret,
            "scope": "https://api.botframework.com/.default"})
        r.raise_for_status()
        d = r.json()
        self._tok, self._exp = d["access_token"], time.time() + int(d.get("expires_in", 3000))
        return self._tok
