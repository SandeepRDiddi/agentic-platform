"""Dependency container: wires infra (Redis/Postgres/Kafka or in-memory) and the trust layer.

Every service builds one Container. Tests build a single in-memory Container and hand it to
several apps so they all share state, exactly like they would share Redis/Postgres in prod.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from agentic.backbone.bus import MessageBus, build_bus
from agentic.common.db import Database
from agentic.common.http import Authenticator, new_client
from agentic.common.kv import KV, build_kv
from agentic.common.repos import (
    ApprovalRepo,
    MemoryApprovalRepo,
    MemoryTaskRepo,
    PgApprovalRepo,
    PgTaskRepo,
    TaskRepo,
)
from agentic.config import Settings
from agentic.trust.audit import AuditLog, MemoryAuditSink, PgAuditSink
from agentic.trust.guardrails import Guardrails
from agentic.trust.identity import IdentityProvider
from agentic.trust.killswitch import KillSwitch


@dataclass
class Container:
    settings: Settings
    kv: KV
    db: Database | None
    tasks: TaskRepo
    approvals: ApprovalRepo
    audit: AuditLog
    killswitch: KillSwitch
    guardrails: Guardrails
    identity: IdentityProvider
    authn: Authenticator
    bus: MessageBus
    # Factory for HTTP clients to sibling services. Tests swap in ASGI transports.
    http: Callable[[str], httpx.AsyncClient] = field(default=lambda base: new_client(base))
    stores: dict = field(default_factory=dict)  # in-memory knowledge stores shared by all users of this container
    _started: bool = False
    _clients: list[httpx.AsyncClient] = field(default_factory=list)

    @classmethod
    def build(cls, settings: Settings) -> Container:
        kv = build_kv(settings)
        db = Database(settings.database_url) if settings.database_url else None
        if db:
            tasks, approvals, sink = PgTaskRepo(db), PgApprovalRepo(db), PgAuditSink(db)
        else:
            tasks, approvals, sink = MemoryTaskRepo(), MemoryApprovalRepo(), MemoryAuditSink()
        identity = IdentityProvider(settings.identity_secret, settings.identity_ttl_seconds)
        return cls(
            settings=settings, kv=kv, db=db, tasks=tasks, approvals=approvals,
            audit=AuditLog(sink),
            killswitch=KillSwitch(kv, fail_closed=settings.killswitch_fail_closed),
            guardrails=Guardrails(
                pii_mode=settings.pii_mode, max_input_chars=settings.max_request_chars
            ),
            identity=identity, authn=Authenticator(settings, identity), bus=build_bus(settings),
        )

    def client(self, base_url: str) -> httpx.AsyncClient:
        c = self.http(base_url)
        self._clients.append(c)
        return c

    async def start(self) -> None:
        if self._started:
            return
        self.settings.validate_for_prod()
        if self.db:
            await self.db.connect()
        await self.bus.start()
        self._started = True

    async def close(self) -> None:
        for c in self._clients:
            await c.aclose()
        await self.bus.stop()
        await self.kv.close()
        if self.db:
            await self.db.close()
        self._started = False

    async def readiness(self) -> dict[str, bool]:
        out: dict[str, bool] = {}
        # Hard dependencies only. Kafka is deliberately excluded: a broker outage must degrade async
        # submission, not take synchronous traffic out of the load balancer.
        for name, probe in (("redis", self.kv.ping),) + ((("postgres", self.db.ping),) if self.db else ()):
            try:
                out[name] = bool(await probe())
            except Exception:
                out[name] = False
        return out
