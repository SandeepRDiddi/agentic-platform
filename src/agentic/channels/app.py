"""Channels service (box 1): Web (+mobile via the same REST/SSE API), Slack, Teams.  Port 8000.

Channels hold NO business logic and NO model access. They authenticate the human (OIDC JWT on
web/mobile; signed requests on Slack/Teams), then submit work to the AI Gateway and stream the
result back. API clients skip this service and call the gateway's /v1/agent/* directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import parse_qs

import httpx
import jwt
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from agentic.channels import slack, teams
from agentic.channels.dispatcher import ResultDispatcher
from agentic.common.errors import AuthError, ForbiddenError, UpstreamError
from agentic.common.http import instrument_app
from agentic.common.logging import setup_logging
from agentic.config import get_settings
from agentic.container import Container
from agentic.trust.tracing import inject_headers, instrument_fastapi, setup_tracing

log = logging.getLogger(__name__)
STATIC = pathlib.Path(__file__).parent / "static"
STREAM_DONE = {"succeeded", "failed", "blocked", "killed", "awaiting_approval"}


class ChatIn(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None


class DecisionIn(BaseModel):
    approve: bool
    reason: str | None = None


class DevTokenIn(BaseModel):
    sub: str
    tenant: str = "default"
    scopes: list[str] = Field(default_factory=lambda: ["agent:run", "approvals:read", "approvals:decide"])


def create_app(container: Container | None = None, external: httpx.AsyncClient | None = None,
               teams_verifier: teams.TeamsJwtVerifier | None = None) -> FastAPI:
    settings = container.settings if container else get_settings()
    c = container or Container.build(settings)
    ext = external or httpx.AsyncClient(timeout=10)
    verifier = teams_verifier or teams.TeamsJwtVerifier(settings.teams_app_id)
    bot_tokens = teams.BotTokenProvider(settings.teams_app_id, settings.teams_app_password)
    dispatcher = ResultDispatcher(c, ext, bot_tokens)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.log_level)
        setup_tracing(settings, "channels")
        await c.start()
        task = asyncio.create_task(dispatcher.run())
        yield
        dispatcher.stop()
        task.cancel()
        await ext.aclose()
        await c.close()

    app = FastAPI(title="Channels", version=settings.service_version, lifespan=lifespan)
    if settings.cors_origin_list:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                           allow_methods=["GET", "POST"], allow_headers=["authorization", "content-type"])
    instrument_app(app, "channels", ready=c.readiness)
    instrument_fastapi(app)
    app.state.container, app.state.dispatcher = c, dispatcher

    _gwc: dict[str, httpx.AsyncClient] = {}

    def _gw() -> httpx.AsyncClient:
        if "c" not in _gwc:
            _gwc["c"] = c.client(settings.gateway_url)
        return _gwc["c"]

    async def gateway(request: Request, method: str, path: str, *, auth: str | None, body: Any = None,
                      params: dict | None = None) -> tuple[int, Any]:
        gw = _gw()
        try:
            r = await gw.request(method, path, json=body, params=params,
                                 headers={"authorization": auth or "", **inject_headers()})
        except httpx.HTTPError as e:
            raise UpstreamError(f"gateway unreachable: {e}") from e
        return r.status_code, (r.json() if r.content else {})

    def channel_auth(user: str) -> str:
        return "Bearer " + c.identity.issue("channel", settings.channel_tenant, on_behalf_of=user)

    # ── Web / mobile ─────────────────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    if settings.dev_token_endpoint and not settings.is_prod:
        @app.post("/dev/token", include_in_schema=False)
        async def dev_token(body: DevTokenIn) -> dict:
            now = int(time.time())
            tok = jwt.encode({"sub": body.sub, "tenant": body.tenant, "scopes": body.scopes, "iat": now,
                              "exp": now + 3600}, settings.jwt_secret, algorithm="HS256")
            return {"access_token": tok}

    @app.post("/api/chat")
    async def chat(request: Request, body: ChatIn) -> JSONResponse:
        st, data = await gateway(request, "POST", "/v1/agent/runs", auth=request.headers.get("authorization"),
                                 body={"query": body.query, "conversation_id": body.conversation_id, "async": True})
        return JSONResponse(data, status_code=st)

    @app.get("/api/chat/{task_id}/events")
    async def events(request: Request, task_id: str) -> StreamingResponse:
        auth = request.headers.get("authorization")

        async def gen() -> AsyncIterator[str]:
            seen, deadline = 0, time.time() + 600
            while time.time() < deadline:
                st, t = await gateway(request, "GET", f"/v1/agent/tasks/{task_id}", auth=auth)
                if st != 200:
                    yield f"event: error\ndata: {json.dumps(t)}\n\n"
                    return
                for ev in t.get("events", [])[seen:]:
                    yield f"event: progress\ndata: {json.dumps(ev)}\n\n"
                seen = len(t.get("events", []))
                if t["status"] in STREAM_DONE:
                    yield f"event: result\ndata: {json.dumps(t)}\n\n"
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/approvals")
    async def list_approvals(request: Request) -> JSONResponse:
        st, d = await gateway(request, "GET", "/v1/agent/approvals", auth=request.headers.get("authorization"))
        return JSONResponse(d, status_code=st)

    @app.post("/api/approvals/{approval_id}")
    async def decide(request: Request, approval_id: str, body: DecisionIn) -> JSONResponse:
        st, d = await gateway(request, "POST", f"/v1/agent/approvals/{approval_id}/decision",
                              auth=request.headers.get("authorization"), body=body.model_dump())
        return JSONResponse(d, status_code=st)

    # ── Slack ────────────────────────────────────────────────────────────────────────────
    async def _dedupe(key: str) -> bool:
        if await c.kv.get(key):
            return True
        await c.kv.set(key, "1", ttl=3600)
        return False

    @app.post("/slack/events", include_in_schema=False)
    async def slack_events(request: Request) -> Response:
        raw = await request.body()
        slack.verify_signature(settings.slack_signing_secret, raw, request.headers.get("x-slack-request-timestamp"),
                               request.headers.get("x-slack-signature"))
        payload = json.loads(raw)
        if payload.get("type") == "url_verification":
            return JSONResponse({"challenge": payload["challenge"]})
        ev = payload.get("event", {})
        if payload.get("type") != "event_callback" or ev.get("bot_id") or ev.get("subtype"):
            return Response(status_code=200)
        if ev.get("type") not in ("app_mention", "message") or (ev["type"] == "message" and ev.get("channel_type") != "im"):
            return Response(status_code=200)
        if await _dedupe(f"slack:event:{payload.get('event_id')}"):
            return Response(status_code=200)  # Slack retry
        text = slack.strip_mention(ev.get("text", ""))
        if text:
            user = f"slack:{ev['user']}"
            reply_to = {"channel": "slack", "slack_channel": ev["channel"], "thread_ts": ev.get("thread_ts") or ev["ts"]}
            conv = f"slack-{ev['channel']}-{reply_to['thread_ts']}"
            st, d = await gateway(request, "POST", "/v1/agent/runs", auth=channel_auth(user),
                                  body={"query": text, "conversation_id": conv, "async": True, "reply_to": reply_to})
            if st >= 400:
                log.warning("slack submit failed: %s %s", st, d)
        return Response(status_code=200)

    @app.post("/slack/interactive", include_in_schema=False)
    async def slack_interactive(request: Request) -> JSONResponse:
        raw = await request.body()
        slack.verify_signature(settings.slack_signing_secret, raw, request.headers.get("x-slack-request-timestamp"),
                               request.headers.get("x-slack-signature"))
        payload = json.loads(parse_qs(raw.decode())["payload"][0])
        act = payload["actions"][0]
        verb, _, aid = act["action_id"].partition(":")
        st, d = await gateway(request, "POST", f"/v1/agent/approvals/{aid}/decision",
                              auth=channel_auth(f"slack:{payload['user']['id']}"), body={"approve": verb == "approve"})
        msg = (f"`{d.get('tool')}` {d.get('status')}" if st == 200 else f"Could not record decision: {d.get('error', {}).get('message', st)}")
        return JSONResponse({"replace_original": True, "text": msg})

    # ── Teams ────────────────────────────────────────────────────────────────────────────
    @app.post("/teams/messages", include_in_schema=False)
    async def teams_messages(request: Request) -> Response:
        verifier.verify(request.headers.get("authorization"))
        act = await request.json()
        if act.get("type") != "message":
            return Response(status_code=200)
        if not teams.trusted_service_url(act.get("serviceUrl", "")):
            raise ForbiddenError("untrusted serviceUrl")
        frm = act.get("from", {})
        user = f"teams:{frm.get('aadObjectId') or frm.get('id')}"
        reply_to = {"channel": "teams", "service_url": act["serviceUrl"], "conversation_id": act["conversation"]["id"],
                    "activity_id": act.get("id")}
        val = act.get("value") or {}
        if "approval_id" in val:
            st, d = await gateway(request, "POST", f"/v1/agent/approvals/{val['approval_id']}/decision",
                                  auth=channel_auth(user), body={"approve": bool(val.get("approve"))})
            await dispatcher.send_teams(reply_to, f"`{d.get('tool')}` {d.get('status')}" if st == 200
                                        else f"Could not record decision: {d.get('error', {}).get('message', st)}", [])
            return Response(status_code=200)
        text = teams.clean_text(act.get("text", ""))
        if text:
            await gateway(request, "POST", "/v1/agent/runs", auth=channel_auth(user),
                          body={"query": text, "conversation_id": f"teams-{act['conversation']['id']}", "async": True,
                                "reply_to": reply_to})
        return Response(status_code=200)

    return app


def app_factory() -> FastAPI:
    return create_app()


__all__ = ["AuthError", "app_factory", "create_app"]
