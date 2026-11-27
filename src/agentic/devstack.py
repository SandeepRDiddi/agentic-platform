"""In-process full stack (all services wired over ASGI, in-memory infra, mock LLM).

Used by the test-suite and the CI eval gate so the *real* code path runs end to end —
gateway → orchestrator → MCP tool server → internal API — with no network and no API keys.
"""

from __future__ import annotations

import asyncio
import pathlib
import time
from contextlib import AsyncExitStack
from typing import Any

import httpx
import jwt

from agentic.backbone.worker import Worker
from agentic.channels.app import create_app as create_channels
from agentic.config import Settings
from agentic.container import Container
from agentic.gateway.app import create_app as create_gateway
from agentic.orchestration.app import create_app as create_orchestrator
from agentic.orchestration.runner import Orchestrator
from agentic.tools.client import McpToolsClient
from agentic.tools.internal_api import create_app as create_internal
from agentic.tools.server import create_app as create_tools

SAMPLE_DOCS = pathlib.Path(__file__).resolve().parents[2] / "data" / "sample_docs"


def make_settings(**over: Any) -> Settings:
    base = dict(
        env="test", llm_mode="mock", worker_max_attempts=3, approver_users="*", rate_limit_per_minute=10_000,
        gateway_url="http://gateway", orchestrator_url="http://orchestrator", mcp_url="http://tools/mcp",
        internal_api_url="http://internal", slack_signing_secret="test-signing-secret", slack_bot_token="xoxb-test",
        dev_token_endpoint=True, worker_concurrency=1,
    )
    base.update(over)
    return Settings(**base)


class Stack:
    def __init__(self, settings: Settings | None = None, seed_docs: bool = True) -> None:
        self.settings = settings or make_settings()
        self.seed_docs = seed_docs
        self.c = Container.build(self.settings)
        self._exit = AsyncExitStack()

    # token helper ---------------------------------------------------------------------------
    def token(self, sub: str = "alice", tenant: str = "default", scopes: list[str] | None = None) -> str:
        now = int(time.time())
        return jwt.encode({"sub": sub, "tenant": tenant, "iat": now, "exp": now + 3600,
                           "scopes": scopes if scopes is not None else ["agent:run", "approvals:read", "approvals:decide"]},
                          self.settings.jwt_secret, algorithm="HS256")

    def auth(self, sub: str = "alice", tenant: str = "default", scopes: list[str] | None = None) -> dict[str, str]:
        return {"authorization": f"Bearer {self.token(sub, tenant, scopes)}"}

    async def __aenter__(self) -> Stack:
        c, s = self.c, self.settings
        self.gateway = create_gateway(c)
        self.internal = create_internal(c)
        self.tools_app = create_tools(c)
        asgi = {s.gateway_url: self.gateway, s.internal_api_url: self.internal}

        def mcp_factory(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:  # noqa: ANN001
            return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.tools_app), base_url="http://tools",
                                     headers=headers, timeout=timeout or 30, auth=auth, follow_redirects=True)

        self.orchestrator_core = Orchestrator(c, tools=McpToolsClient(s.mcp_url, httpx_client_factory=mcp_factory))
        self.orch_app = create_orchestrator(c, self.orchestrator_core)
        asgi[s.orchestrator_url] = self.orch_app
        self.worker = Worker(c, self.orchestrator_core)
        self.ext_calls: list[httpx.Request] = []

        def ext_handler(req: httpx.Request) -> httpx.Response:
            self.ext_calls.append(req)
            return httpx.Response(200, json={"ok": True, "access_token": "t", "expires_in": 3600})

        self.channels = create_channels(c, external=httpx.AsyncClient(transport=httpx.MockTransport(ext_handler)))
        asgi[s.gateway_url] = self.gateway
        c.http = lambda base: httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi[base]), base_url=base,
                                                timeout=60)
        await c.start()
        # The MCP session manager's anyio task group must be entered and exited from ONE task,
        # so the tool app's lifespan lives in its own background task.
        self._lifespan_stop, ready = asyncio.Event(), asyncio.Event()

        async def hold_lifespan() -> None:
            async with self.tools_app.router.lifespan_context(self.tools_app):
                ready.set()
                await self._lifespan_stop.wait()

        self._lifespan_task = asyncio.create_task(hold_lifespan())
        await asyncio.wait_for(ready.wait(), 10)
        if self.seed_docs:
            for f in sorted(SAMPLE_DOCS.glob("*.md")):
                await self.orchestrator_core.knowledge.ingest(s.channel_tenant, f.name, f.stem, f.read_text())
        self.api = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.gateway), base_url="http://gateway", timeout=60)
        self.channels_api = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.channels), base_url="http://channels", timeout=60)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.api.aclose()
        await self.channels_api.aclose()
        self._lifespan_stop.set()
        await self._lifespan_task
        await self._exit.aclose()
        await self.c.close()

    async def run(self, query: str, *, sub: str = "alice", tenant: str = "default", **extra: Any) -> dict:
        r = await self.api.post("/v1/agent/runs", json={"query": query, **extra}, headers=self.auth(sub, tenant))
        return {"http_status": r.status_code, **r.json()}
