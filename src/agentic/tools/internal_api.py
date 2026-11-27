"""Reference *internal API* (tickets + notifications) that the MCP server wraps. Port 8004.

Stands in for the enterprise systems your agents act on (ServiceNow/Jira/etc). Swap the
InternalApiClient base URL in prod; the MCP layer and its approval/scope policy don't change.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from agentic.common.db import Database
from agentic.common.errors import NotFound
from agentic.common.http import instrument_app, principal_dependency, require_scope
from agentic.common.logging import setup_logging
from agentic.common.schemas import Principal, now
from agentic.config import get_settings
from agentic.container import Container
from agentic.trust.tracing import instrument_fastapi, setup_tracing

log = logging.getLogger(__name__)


class TicketIn(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    description: str = Field("", max_length=4000)
    priority: str = Field("medium", pattern="^(low|medium|high|critical)$")


class TicketPatch(BaseModel):
    status: str = Field(pattern="^(open|in_progress|resolved|closed)$")


class NotificationIn(BaseModel):
    channel: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=2000)


class MemTickets:
    def __init__(self) -> None:
        self.rows: dict[int, dict[str, Any]] = {}
        self.seq = 0

    async def create(self, tenant: str, t: TicketIn, by: str) -> dict[str, Any]:
        self.seq += 1
        row = {"id": self.seq, "tenant": tenant, **t.model_dump(), "status": "open", "created_by": by,
               "created_at": now().isoformat()}
        self.rows[self.seq] = row
        return row

    async def get(self, tenant: str, tid: int) -> dict[str, Any]:
        r = self.rows.get(tid)
        if not r or r["tenant"] != tenant:
            raise NotFound(f"ticket {tid} not found")
        return r

    async def search(self, tenant: str, q: str, status: str) -> list[dict[str, Any]]:
        ql = q.lower()
        return [r for r in self.rows.values() if r["tenant"] == tenant
                and (not status or r["status"] == status)
                and (not ql or ql in r["title"].lower() or ql in r["description"].lower())][:25]

    async def patch(self, tenant: str, tid: int, status: str) -> dict[str, Any]:
        r = await self.get(tenant, tid)
        r["status"] = status
        return r


class PgTickets:
    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def _d(r: Any) -> dict[str, Any]:
        d = dict(r)
        d["created_at"] = d["created_at"].isoformat()
        d.pop("updated_at", None)
        return d

    async def create(self, tenant, t, by):
        r = await self.db.fetchrow(
            "INSERT INTO tickets (tenant, title, description, priority, created_by) VALUES ($1,$2,$3,$4,$5) RETURNING *",
            tenant, t.title, t.description, t.priority, by)
        return self._d(r)

    async def get(self, tenant, tid):
        r = await self.db.fetchrow("SELECT * FROM tickets WHERE id=$1 AND tenant=$2", tid, tenant)
        if not r:
            raise NotFound(f"ticket {tid} not found")
        return self._d(r)

    async def search(self, tenant, q, status):
        rows = await self.db.fetch(
            "SELECT * FROM tickets WHERE tenant=$1 AND ($2='' OR status=$2)"
            " AND ($3='' OR title ILIKE '%'||$3||'%' OR description ILIKE '%'||$3||'%') ORDER BY id DESC LIMIT 25",
            tenant, status, q)
        return [self._d(r) for r in rows]

    async def patch(self, tenant, tid, status):
        r = await self.db.fetchrow(
            "UPDATE tickets SET status=$3, updated_at=now() WHERE id=$1 AND tenant=$2 RETURNING *", tid, tenant, status)
        if not r:
            raise NotFound(f"ticket {tid} not found")
        return self._d(r)


def create_app(container: Container | None = None) -> FastAPI:
    settings = container.settings if container else get_settings()
    c = container or Container.build(settings)
    store: Any = PgTickets(c.db) if c.db else MemTickets()
    sent: list[dict[str, Any]] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.log_level)
        setup_tracing(settings, "internal-api")
        await c.start()
        yield
        await c.close()

    app = FastAPI(title="Internal API (tickets)", lifespan=lifespan)
    instrument_app(app, "internal-api", ready=c.readiness)
    instrument_fastapi(app)
    who = principal_dependency(c.authn)
    rd, wr = require_scope("internal:read", who), require_scope("internal:write", who)

    @app.get("/tickets")
    async def search(q: str = "", status: str = "", p: Principal = Depends(rd)) -> list[dict]:
        return await store.search(p.tenant, q, status)

    @app.get("/tickets/{tid}")
    async def get(tid: int, p: Principal = Depends(rd)) -> dict:
        return await store.get(p.tenant, tid)

    @app.post("/tickets", status_code=201)
    async def create(body: TicketIn, p: Principal = Depends(wr)) -> dict:
        return await store.create(p.tenant, body, p.user_id)

    @app.patch("/tickets/{tid}")
    async def patch(tid: int, body: TicketPatch, p: Principal = Depends(wr)) -> dict:
        return await store.patch(p.tenant, tid, body.status)

    @app.post("/notifications", status_code=202)
    async def notify(body: NotificationIn, p: Principal = Depends(wr)) -> dict:
        rec = {"tenant": p.tenant, "by": p.user_id, **body.model_dump()}
        sent.append(rec)  # real deployments: post to Teams/Slack webhook or notification service
        log.info("notification", extra={"extra_fields": rec})
        return {"queued": True, **rec}

    app.state.tickets = store
    app.state.sent = sent
    app.state.state = app.state  # tests: app.state.state.tickets
    return app


def app_factory() -> FastAPI:
    return create_app()
