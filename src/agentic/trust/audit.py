"""Audit log (trust layer: "every action").

Append-only, hash-chained: hash_n = SHA256(prev_hash || canonical(event_n)). Any edit or deletion
of a historical row breaks the chain and `verify()` pinpoints where. In Postgres the table also
has triggers that reject UPDATE/DELETE/TRUNCATE. Every event is additionally emitted as a JSON
log line so it can be shipped to WORM/immutable storage (e.g. Azure immutable blob).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agentic.common.db import Database
from agentic.common.schemas import now

log = logging.getLogger("audit")
GENESIS = "0" * 64


class AuditEvent(BaseModel):
    ts: datetime = Field(default_factory=now)
    actor: str
    actor_type: str  # user | agent | service | system
    action: str
    resource: str | None = None
    decision: str | None = None  # allow | deny | blocked | ok | error ...
    tenant: str | None = None
    task_id: str | None = None
    trace_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str = GENESIS
    hash: str = ""

    def compute_hash(self) -> str:
        body = self.model_dump(mode="json", exclude={"hash"})
        body["ts"] = self.ts.isoformat()
        canon = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canon.encode()).hexdigest()


class AuditSink(Protocol):
    async def append(self, ev: AuditEvent) -> AuditEvent: ...
    async def all(self, limit: int | None = None) -> list[AuditEvent]: ...


class MemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append(self, ev: AuditEvent) -> AuditEvent:
        ev.prev_hash = self.events[-1].hash if self.events else GENESIS
        ev.hash = ev.compute_hash()
        self.events.append(ev)
        return ev

    async def all(self, limit: int | None = None) -> list[AuditEvent]:
        return list(self.events if limit is None else self.events[-limit:])


class PgAuditSink:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def append(self, ev: AuditEvent) -> AuditEvent:
        async with self.db.transaction() as c:
            await c.execute("SELECT pg_advisory_xact_lock(727275)")  # serialize chain writers
            prev = await c.fetchval("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1")
            ev.prev_hash = prev or GENESIS
            ev.hash = ev.compute_hash()
            await c.execute(
                "INSERT INTO audit_log (ts, actor, actor_type, action, resource, decision, tenant,"
                " task_id, trace_id, details, prev_hash, hash) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)",
                ev.ts, ev.actor, ev.actor_type, ev.action, ev.resource, ev.decision, ev.tenant,
                ev.task_id, ev.trace_id, ev.details, ev.prev_hash, ev.hash,
            )
        return ev

    async def all(self, limit: int | None = None) -> list[AuditEvent]:
        sql = "SELECT * FROM audit_log ORDER BY id" + (f" DESC LIMIT {int(limit)}" if limit else "")
        rows = await self.db.fetch(sql)
        evs = [AuditEvent(**{k: r[k] for k in r.keys() if k != "id"}) for r in rows]
        return list(reversed(evs)) if limit else evs


class AuditLog:
    def __init__(self, sink: AuditSink) -> None:
        self.sink = sink

    async def record(
        self, *, actor: str, actor_type: str, action: str, resource: str | None = None,
        decision: str | None = None, tenant: str | None = None, task_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        trace_id = None
        try:
            from opentelemetry import trace

            sc = trace.get_current_span().get_span_context()
            if sc.is_valid:
                trace_id = format(sc.trace_id, "032x")
        except Exception:  # noqa: S110
            pass
        ev = AuditEvent(
            actor=actor, actor_type=actor_type, action=action, resource=resource, decision=decision,
            tenant=tenant, task_id=task_id, trace_id=trace_id, details=_clip(details or {}),
        )
        ev = await self.sink.append(ev)
        log.info("audit", extra={"extra_fields": {"audit": ev.model_dump(mode="json")}})
        return ev

    async def verify(self) -> tuple[bool, int | None]:
        """Returns (ok, index_of_first_broken_event)."""
        prev = GENESIS
        for i, ev in enumerate(await self.sink.all()):
            if ev.prev_hash != prev or ev.hash != ev.compute_hash():
                return False, i
            prev = ev.hash
        return True, None


def _clip(d: dict[str, Any], limit: int = 2000) -> dict[str, Any]:
    """Keep audit rows bounded; never store large payloads verbatim."""
    out = {}
    for k, v in d.items():
        s = json.dumps(v, default=str)
        out[k] = v if len(s) <= limit else {"truncated": True, "sha256": hashlib.sha256(s.encode()).hexdigest()}
    return out
