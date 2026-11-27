"""Thin asyncpg wrapper. pgvector values travel as text literals cast with ::vector."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


def vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class Database:
    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10) -> None:
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Any = None

    async def connect(self) -> None:
        import asyncpg

        async def init(conn: Any) -> None:
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

        self.pool = await asyncpg.create_pool(
            self.dsn, min_size=self.min_size, max_size=self.max_size, init=init, command_timeout=30
        )

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        async with self.pool.acquire() as c:
            return await c.fetch(sql, *args)

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        async with self.pool.acquire() as c:
            return await c.fetchrow(sql, *args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        async with self.pool.acquire() as c:
            return await c.fetchval(sql, *args)

    async def execute(self, sql: str, *args: Any) -> str:
        async with self.pool.acquire() as c:
            return await c.execute(sql, *args)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Any]:
        async with self.pool.acquire() as c, c.transaction():
            yield c

    async def ping(self) -> bool:
        return (await self.fetchval("SELECT 1")) == 1
