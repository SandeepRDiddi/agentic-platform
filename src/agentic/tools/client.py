"""MCP client used by the orchestrator. Stateless: one short session per call."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agentic.common.errors import AgenticError, ForbiddenError, UpstreamError


class ToolsClient(Protocol):
    async def call(self, name: str, args: dict[str, Any], *, token: str) -> Any: ...


class McpToolsClient:
    def __init__(self, url: str, httpx_client_factory: Callable[..., httpx.AsyncClient] | None = None) -> None:
        self.url = url
        self._factory = httpx_client_factory

    async def call(self, name: str, args: dict[str, Any], *, token: str) -> Any:
        kw: dict[str, Any] = {"headers": {"authorization": f"Bearer {token}"}}
        if self._factory:
            kw["httpx_client_factory"] = self._factory
        try:
            async with streamablehttp_client(self.url, **kw) as (r, w, _), ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool(name, args)
        except* httpx.HTTPStatusError as eg:
            code = eg.exceptions[0].response.status_code
            raise (ForbiddenError if code == 403 else UpstreamError)(f"tool server HTTP {code}") from eg
        except* (httpx.HTTPError, OSError) as eg:
            raise UpstreamError(f"tool server unreachable: {eg.exceptions[0]}") from eg
        text = "".join(getattr(c, "text", "") for c in res.content)
        if res.isError:
            low = text.lower()
            if "kill switch" in low:
                from agentic.common.errors import KillSwitchEngaged

                raise KillSwitchEngaged(text)
            if "lacks scope" in low or "requires a human" in low or "approval" in low:
                raise ForbiddenError(text)
            raise AgenticError(text or "tool error")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"result": text}
