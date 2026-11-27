"""MCP tool server (box 5). Port 8003, streamable-HTTP, stateless (scales behind any L4/L7 LB).

Defence in depth — a tool call must pass ALL of these before touching an internal API:
  1. valid per-agent identity token (HS256, short-lived)          → who is calling
  2. caller holds the tool's scope (tools:read / tools:write)      → least privilege
  3. kill switch (global / tenant / agent)                        → emergency stop
  4. guardrails on arguments (secrets, injection payloads)
  5. WRITE tools only: a human-approved, unexpired, single-use approval whose
     tool + argument hash match exactly what is being executed    → human in the loop
  6. every decision (allow AND deny) is written to the audit log
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from agentic.common.errors import (
    AgenticError,
    AuthError,
    ForbiddenError,
    GuardrailViolation,
    UpstreamError,
)
from agentic.common.metrics import TOOL_CALLS
from agentic.common.repos import args_hash
from agentic.common.schemas import Principal
from agentic.config import get_settings
from agentic.container import Container
from agentic.tools.catalog import CATALOG
from agentic.trust.tracing import span

log = logging.getLogger(__name__)
_principal: ContextVar[Principal | None] = ContextVar("tool_principal", default=None)


class ToolRuntime:
    def __init__(self, c: Container) -> None:
        self.c = c
        self._http: httpx.AsyncClient | None = None
        self._knowledge: Any = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = self.c.client(self.c.settings.internal_api_url)
        return self._http

    @property
    def knowledge(self):
        if self._knowledge is None:
            from agentic.knowledge.service import Knowledge
            from agentic.models.client import LLMClient
            from agentic.models.router import ModelRouter

            llm = LLMClient(self.c)
            self._knowledge = Knowledge(self.c, llm, ModelRouter(llm, self.c.settings))
        return self._knowledge

    async def api(self, method: str, path: str, p: Principal, obo: str, **kw: Any) -> Any:
        token = self.c.identity.issue("toolserver", p.tenant, on_behalf_of=obo)
        try:
            r = await self.http.request(method, path, headers={"authorization": f"Bearer {token}"}, **kw)
        except httpx.HTTPError as e:
            raise UpstreamError(f"internal API unreachable: {e}") from e
        if r.status_code >= 400:
            raise UpstreamError(f"internal API {r.status_code}: {r.text[:300]}")
        return r.json()

    async def guarded(
        self, name: str, args: dict[str, Any], approval_id: str,
        fn: Callable[[Principal, str], Awaitable[Any]],
    ) -> str:
        c, spec = self.c, CATALOG[name]
        p = _principal.get()
        if p is None:
            raise AuthError("no caller identity")

        async def audit(decision: str, **details: Any) -> None:
            await c.audit.record(actor=p.sub, actor_type="agent", action=f"tool.{name}", resource=name,
                                 decision=decision, tenant=p.tenant, details={"args": args, **details})

        with span(f"tool.{name}", **{"agentic.tool": name, "agentic.agent": p.sub, "agentic.write": spec.write}):
            try:
                if not p.has(spec.scope):
                    raise ForbiddenError(f"{p.sub} lacks scope {spec.scope} for {name}")
                await c.killswitch.check(tenant=p.tenant, agent=p.sub)
                g = c.guardrails.check_tool_call(name, args)
                if not g.allowed:
                    raise GuardrailViolation(f"tool arguments rejected: {g.reason}")
                obo = p.user_id
                approval = None
                if spec.write:
                    if not approval_id:
                        raise ForbiddenError(f"{name} requires a human approval_id")
                    cur = await c.approvals.get(approval_id)
                    if cur.tenant != p.tenant or cur.tool != name or cur.args_hash != args_hash(name, args):
                        raise ForbiddenError("approval does not match this tool call")
                    approval = await c.approvals.claim_execution(approval_id)
                    if approval is None:
                        raise ForbiddenError(f"approval is {cur.status}, not approved (single-use)")
                    obo = approval.requested_by
                try:
                    out = await fn(p, obo)
                except Exception:
                    if approval:
                        await c.approvals.finish(approval_id, "failed", {"error": "tool execution failed"})
                    raise
                if approval:
                    await c.approvals.finish(approval_id, "executed", out if isinstance(out, dict) else {"result": out})
            except AgenticError as e:
                TOOL_CALLS.labels(name, "denied" if isinstance(e, (ForbiddenError, GuardrailViolation)) else "error").inc()
                await audit("deny" if isinstance(e, (ForbiddenError, GuardrailViolation)) else "error", reason=e.message)
                raise
            TOOL_CALLS.labels(name, "ok").inc()
            await audit("allow", approval_id=approval_id or None)
            return json.dumps(out, default=str)


def build_mcp(c: Container) -> FastMCP:
    rt = ToolRuntime(c)
    mcp = FastMCP("agentic-tools", stateless_http=True, json_response=True, host="0.0.0.0")  # noqa: S104

    @mcp.tool()
    async def search_tickets(query: str = "", status: str = "") -> str:
        """Search tickets by text and/or status."""
        return await rt.guarded("search_tickets", {"query": query, "status": status}, "",
                                lambda p, o: rt.api("GET", "/tickets", p, o, params={"q": query, "status": status}))

    @mcp.tool()
    async def get_ticket(ticket_id: int) -> str:
        """Fetch one ticket."""
        return await rt.guarded("get_ticket", {"ticket_id": ticket_id}, "",
                                lambda p, o: rt.api("GET", f"/tickets/{ticket_id}", p, o))

    @mcp.tool()
    async def search_knowledge(query: str, k: int = 4) -> str:
        """Hybrid RAG+GraphRAG search over the knowledge base."""
        async def go(p: Principal, o: str) -> Any:
            ctx = await rt.knowledge.retrieve(p.tenant, query, k=min(k, 10), user=o)
            return {"chunks": [{"id": h.id, "source": h.source, "text": h.content} for h in ctx.chunks],
                    "graph_facts": ctx.facts}
        return await rt.guarded("search_knowledge", {"query": query, "k": k}, "", go)

    @mcp.tool()
    async def create_ticket(title: str, description: str = "", priority: str = "medium", approval_id: str = "") -> str:
        """Create a ticket. Requires a human-approved approval_id."""
        a = {"title": title, "description": description, "priority": priority}
        return await rt.guarded("create_ticket", a, approval_id,
                                lambda p, o: rt.api("POST", "/tickets", p, o, json=a))

    @mcp.tool()
    async def update_ticket_status(ticket_id: int, status: str, approval_id: str = "") -> str:
        """Change a ticket's status. Requires a human-approved approval_id."""
        a = {"ticket_id": ticket_id, "status": status}
        return await rt.guarded("update_ticket_status", a, approval_id,
                                lambda p, o: rt.api("PATCH", f"/tickets/{ticket_id}", p, o, json={"status": status}))

    @mcp.tool()
    async def send_notification(channel: str, message: str, approval_id: str = "") -> str:
        """Post a notification. Requires a human-approved approval_id."""
        a = {"channel": channel, "message": message}
        return await rt.guarded("send_notification", a, approval_id,
                                lambda p, o: rt.api("POST", "/notifications", p, o, json=a))

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "mcp-tools"})

    @mcp.custom_route("/metrics", methods=["GET"])
    async def metrics(_: Request) -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @mcp.custom_route("/readyz", methods=["GET"])
    async def readyz(_: Request) -> JSONResponse:
        checks = await c.readiness()
        return JSONResponse({"ready": all(checks.values()), "checks": checks}, status_code=200 if all(checks.values()) else 503)

    return mcp


class IdentityMiddleware:
    """Pure-ASGI: authenticate the per-agent identity token and expose it via a contextvar."""

    OPEN = {"/healthz", "/readyz", "/metrics"}

    def __init__(self, app: Any, c: Container) -> None:
        self.app, self.c = app, c

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["path"] in self.OPEN:
            await self.app(scope, receive, send)
            return
        auth = dict(scope["headers"]).get(b"authorization", b"").decode()
        try:
            if not auth.lower().startswith("bearer "):
                raise AuthError("missing bearer token")
            p = self.c.identity.verify(auth.split(" ", 1)[1].strip())
        except AuthError as e:
            await JSONResponse({"error": {"code": e.code, "message": e.message}}, status_code=401)(scope, receive, send)
            return
        tok = _principal.set(p)
        try:
            await self.app(scope, receive, send)
        finally:
            _principal.reset(tok)


def create_app(container: Container | None = None) -> Starlette:
    c = container or Container.build(get_settings())
    mcp = build_mcp(c)
    app = mcp.streamable_http_app()
    app.add_middleware(IdentityMiddleware, c=c)
    app.state.container = c
    app.state.mcp = mcp
    return app


def app_factory() -> Starlette:
    c = Container.build(get_settings())
    app = create_app(c)
    inner = app.router.lifespan_context

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(a: Starlette):
        from agentic.common.logging import setup_logging
        from agentic.trust.tracing import setup_tracing

        setup_logging(c.settings.log_level)
        setup_tracing(c.settings, "mcp-tools")
        await c.start()
        async with inner(a):
            yield
        await c.close()

    app.router.lifespan_context = lifespan
    return app
