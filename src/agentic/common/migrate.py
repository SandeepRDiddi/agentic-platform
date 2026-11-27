"""Minimal forward-only SQL migrator. Run as a pre-deploy Job: `agentic-migrate`."""

from __future__ import annotations

import asyncio
import hashlib
import sys
from importlib import resources

from agentic.common.db import Database
from agentic.config import get_settings


async def migrate(dsn: str) -> list[str]:
    db = Database(dsn, min_size=1, max_size=2)
    await db.connect()
    applied: list[str] = []
    try:
        async with db.transaction() as c:
            await c.execute("SELECT pg_advisory_xact_lock(727274)")  # one migrator at a time
            await c.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(name text PRIMARY KEY, checksum text NOT NULL, applied_at timestamptz DEFAULT now())"
            )
            done = {r["name"]: r["checksum"] for r in await c.fetch("SELECT * FROM schema_migrations")}
            files = sorted(
                (f for f in resources.files("agentic.migrations").iterdir() if f.name.endswith(".sql")),
                key=lambda f: f.name,
            )
            for f in files:
                sql = f.read_text()
                checksum = hashlib.sha256(sql.encode()).hexdigest()
                if f.name in done:
                    if done[f.name] != checksum:
                        raise RuntimeError(f"migration {f.name} was modified after being applied")
                    continue
                await c.execute(sql)
                await c.execute(
                    "INSERT INTO schema_migrations (name, checksum) VALUES ($1, $2)", f.name, checksum
                )
                applied.append(f.name)
    finally:
        await db.close()
    return applied


def main() -> None:
    dsn = get_settings().database_url
    if not dsn:
        print("DATABASE_URL is not set", file=sys.stderr)
        raise SystemExit(2)
    applied = asyncio.run(migrate(dsn))
    print("applied:", ", ".join(applied) if applied else "nothing (up to date)")


if __name__ == "__main__":
    main()
