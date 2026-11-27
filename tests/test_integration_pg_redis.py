"""Integration tests against REAL Postgres+pgvector and Redis (skipped unless env vars are set).

    TEST_DATABASE_URL=postgresql://postgres@localhost:55432/agentic TEST_REDIS_URL=redis://localhost:56379/0 pytest
CI provides both as service containers.
"""

import asyncio
import os

import asyncpg
import pytest
import pytest_asyncio

from agentic.common.db import Database
from agentic.common.kv import RedisKV
from agentic.common.migrate import migrate
from agentic.devstack import Stack, make_settings
from agentic.trust.audit import AuditLog, PgAuditSink

DSN, REDIS = os.environ.get("TEST_DATABASE_URL"), os.environ.get("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not (DSN and REDIS), reason="TEST_DATABASE_URL / TEST_REDIS_URL not set")


@pytest_asyncio.fixture
async def fresh_db():
    conn = await asyncpg.connect(DSN)
    await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    await conn.close()
    await migrate(DSN)
    kv = RedisKV(REDIS)
    await kv._r.flushdb()  # noqa: SLF001
    await kv.close()
    yield DSN


async def test_migrations_are_idempotent_and_checksummed(fresh_db):
    assert await migrate(DSN) == []
    conn = await asyncpg.connect(DSN)
    await conn.execute("UPDATE schema_migrations SET checksum='tampered'")
    await conn.close()
    with pytest.raises(RuntimeError, match="modified after being applied"):
        await migrate(DSN)


async def test_redis_kv_semantics(fresh_db):
    kv = RedisKV(REDIS)
    await kv.set("a", "1", ttl=1)
    assert await kv.get("a") == "1" and await kv.incr("n", ttl=5) == 1 and await kv.incr("n", ttl=5) == 2
    assert abs(await kv.incrbyfloat("f", 0.5, ttl=5) - 0.5) < 1e-9
    assert "n" in await kv.keys("n")
    await asyncio.sleep(1.1)
    assert await kv.get("a") is None
    await kv.close()


async def test_audit_log_is_append_only_in_postgres(fresh_db):
    db = Database(DSN)
    await db.connect()
    log = AuditLog(PgAuditSink(db))
    for i in range(5):
        await log.record(actor="a", actor_type="agent", action=f"x{i}", decision="allow", details={"i": i, "f": 1.5})
    assert await log.verify() == (True, None)
    with pytest.raises(asyncpg.PostgresError, match="append-only"):
        await db.execute("UPDATE audit_log SET decision='deny' WHERE id=1")
    with pytest.raises(asyncpg.PostgresError, match="append-only"):
        await db.execute("DELETE FROM audit_log")
    with pytest.raises(asyncpg.PostgresError, match="append-only"):
        await db.execute("TRUNCATE audit_log")
    await asyncio.gather(*(log.record(actor="p", actor_type="agent", action=f"c{i}") for i in range(20)))  # concurrent writers
    assert await log.verify() == (True, None)
    await db.close()


async def test_full_platform_on_postgres_and_redis(fresh_db):
    async with Stack(make_settings(database_url=DSN, redis_url=REDIS)) as s:
        r = await s.run("How quickly must the on-call engineer acknowledge a Severity 1 incident?")
        assert r["status"] == "succeeded" and "5 minutes" in r["answer"] and r["citations"]
        # GraphRAG + communities on pgvector
        k = s.orchestrator_core.knowledge
        assert any("nndm" in f for f in (await k.retrieve("default", "What is NNDM?")).facts)
        assert await k.build_communities("default") > 0
        # human-approval flow on Postgres
        r = await s.run("Create a ticket for the failing nightly ingestion job", sub="alice")
        ap = r["pending_approvals"][0]
        d = await s.api.post(f"/v1/agent/approvals/{ap['id']}/decision", json={"approve": True}, headers=s.auth("bob"))
        assert d.json()["status"] == "executed"
        rows = await s.c.db.fetch("SELECT * FROM tickets")
        assert len(rows) == 1 and rows[0]["created_by"] == "alice"
        replay = await s.api.post(f"/v1/agent/approvals/{ap['id']}/decision", json={"approve": True}, headers=s.auth("bob"))
        assert replay.status_code == 409
        # tenant isolation in SQL
        assert (await k.retrieve("other", "Severity 1")).chunks == []
        # kill switch via Redis
        await s.c.killswitch.engage("global", "drill", "root")
        assert (await s.api.post("/v1/agent/runs", json={"query": "hi"}, headers=s.auth())).status_code == 503
        await s.c.killswitch.release("global")
        # memory on pgvector + forget
        assert await k.remember("default", "alice", None, "my team is the Data Platform squad.", "ok") == 1
        assert await k.long_term("default", "alice", "my team") and await k.forget("default", "alice") == 1
        # hash-chained audit verified on Postgres
        v = await s.api.get("/v1/admin/audit/verify", headers=s.auth("root", scopes=["admin:audit"]))
        assert v.json()["chain_intact"] is True
        # per-task persistence round trip
        t = await s.c.tasks.get(r["task_id"])
        assert t.status.value == "awaiting_approval" or t.status.value == "succeeded"
        # rate limit counters live in Redis
        assert await s.c.kv.keys("rl:")
