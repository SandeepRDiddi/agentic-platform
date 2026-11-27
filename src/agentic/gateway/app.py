"""AI Gateway service (box 2). Port 8001.

Public surface for API clients/channels:  /v1/agent/*  (proxied to the orchestrator)
Internal surface for agents:              /v1/chat/completions, /v1/embeddings
Admin surface:                            /v1/admin/*  (kill switch, audit verification, spend)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agentic.common.errors import UpstreamError
from agentic.common.http import instrument_app, principal_dependency, require_scope
from agentic.common.logging import setup_logging
from agentic.common.schemas import Principal, RunRequest
from agentic.config import Settings, get_settings
from agentic.container import Container
from agentic.gateway.service import GatewayService
from agentic.trust.tracing import inject_headers, instrument_fastapi, setup_tracing


class KillRequest(BaseModel):
    scope: str
    reason: str


class DecisionRequest(BaseModel):
    approve: bool
    reason: str | None = None


class EmbedRequest(BaseModel):
    input: list[str]


def create_app(container: Container | None = None) -> FastAPI:
    settings: Settings = container.settings if container else get_settings()
    c = container or Container.build(settings)
    gw = GatewayService(c)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.log_level)
        setup_tracing(settings, "ai-gateway")
        await c.start()
        yield
        await c.close()

    app = FastAPI(title="AI Gateway", version=settings.service_version, lifespan=lifespan)
    instrument_app(app, "gateway", ready=c.readiness)
    instrument_fastapi(app)
    app.state.container = c
    app.state.service = gw

    who = principal_dependency(c.authn)
    _orch: dict[str, httpx.AsyncClient] = {}

    def orch() -> httpx.AsyncClient:
        if "c" not in _orch:
            _orch["c"] = c.client(settings.orchestrator_url)
        return _orch["c"]

    # ── internal LLM surface ─────────────────────────────────────────────────────────────
    @app.post("/v1/chat/completions")
    async def chat(body: dict[str, Any], p: Principal = Depends(require_scope("llm:invoke", who))) -> dict:
        return await gw.chat(p, body)

    @app.post("/v1/embeddings")
    async def embeddings(body: EmbedRequest, p: Principal = Depends(require_scope("llm:invoke", who))) -> dict:
        return await gw.embed(p, body.input)

    # ── agent runs: authenticated, rate limited, then proxied to the orchestrator ────────
    async def _forward(
        request: Request, p: Principal, method: str, path: str, *, json_body: Any = None,
        params: dict | None = None,
    ) -> JSONResponse:
        await gw.admit(p, cost_check=False)
        obo = p.user_id
        if p.kind == "service" and p.has("channel:impersonate"):
            obo = request.headers.get("x-on-behalf-of") or p.sub
        token = c.identity.issue("gateway", p.tenant, on_behalf_of=obo)
        client = orch()
        try:
            r = await client.request(
                method, path, json=json_body, params=params,
                headers={"authorization": f"Bearer {token}", **inject_headers()},
            )
        except httpx.HTTPError as e:
            raise UpstreamError(f"orchestrator unreachable: {e}") from e
        return JSONResponse(r.json() if r.content else {}, status_code=r.status_code)

    @app.post("/v1/agent/runs")
    async def run(request: Request, body: RunRequest, p: Principal = Depends(require_scope("agent:run", who))):
        return await _forward(request, p, "POST", "/v1/runs", json_body=body.model_dump(by_alias=True))

    @app.get("/v1/agent/tasks/{task_id}")
    async def task(request: Request, task_id: str, p: Principal = Depends(require_scope("agent:run", who))):
        return await _forward(request, p, "GET", f"/v1/tasks/{task_id}")

    @app.get("/v1/agent/approvals")
    async def approvals(request: Request, status: str | None = "pending",
                        p: Principal = Depends(require_scope("approvals:read", who))):
        return await _forward(request, p, "GET", "/v1/approvals", params={"status": status} if status else None)

    @app.post("/v1/agent/approvals/{approval_id}/decision")
    async def decide(request: Request, approval_id: str, body: DecisionRequest,
                     p: Principal = Depends(require_scope("approvals:decide", who))):
        return await _forward(request, p, "POST", f"/v1/approvals/{approval_id}/decision", json_body=body.model_dump())

    @app.delete("/v1/agent/memory")
    async def forget(request: Request, p: Principal = Depends(require_scope("agent:run", who))):
        return await _forward(request, p, "DELETE", "/v1/memory")

    # ── admin ────────────────────────────────────────────────────────────────────────────
    admin = require_scope("admin:killswitch", who)

    @app.get("/v1/admin/killswitch")
    async def ks_status(_: Principal = Depends(admin)) -> list[dict]:
        return await c.killswitch.status()

    @app.put("/v1/admin/killswitch")
    async def ks_engage(body: KillRequest, p: Principal = Depends(admin)) -> dict:
        rec = await c.killswitch.engage(body.scope, body.reason, p.sub)
        await c.audit.record(actor=p.sub, actor_type=p.kind, action="killswitch.engage",
                             resource=body.scope, decision="ok", tenant=p.tenant,
                             details={"reason": body.reason})
        return rec

    @app.delete("/v1/admin/killswitch/{scope:path}")
    async def ks_release(scope: str, p: Principal = Depends(admin)) -> dict:
        await c.killswitch.release(scope)
        await c.audit.record(actor=p.sub, actor_type=p.kind, action="killswitch.release",
                             resource=scope, decision="ok", tenant=p.tenant)
        return {"released": scope}

    @app.get("/v1/admin/audit/verify")
    async def audit_verify(_: Principal = Depends(require_scope("admin:audit", who))) -> dict:
        ok, idx = await c.audit.verify()
        return {"chain_intact": ok, "first_broken_index": idx}

    @app.get("/v1/admin/spend")
    async def spend(p: Principal = Depends(require_scope("admin:spend", who))) -> dict:
        return {"tenant": p.tenant, "spent_usd_today": await gw.spend(p.tenant),
                "budget_usd": settings.tenant_daily_budget_usd}

    return app


def app_factory() -> FastAPI:  # uvicorn --factory agentic.gateway.app:app_factory
    return create_app()
