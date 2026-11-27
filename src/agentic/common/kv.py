"""Tiny key-value abstraction: Redis in prod, in-process dict for dev/tests.

Used for the response cache, rate limits, spend counters, kill-switch flags and
short-term conversation memory.
"""

from __future__ import annotations

import time
from typing import Protocol

from agentic.config import Settings


class KV(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def incr(self, key: str, amount: int = 1, ttl: int | None = None) -> int: ...
    async def incrbyfloat(self, key: str, amount: float, ttl: int | None = None) -> float: ...
    async def keys(self, prefix: str) -> list[str]: ...
    async def ping(self) -> bool: ...
    async def close(self) -> None: ...


class MemoryKV:
    def __init__(self) -> None:
        self._d: dict[str, tuple[str, float | None]] = {}

    def _live(self, key: str) -> str | None:
        item = self._d.get(key)
        if not item:
            return None
        value, exp = item
        if exp is not None and exp < time.time():
            self._d.pop(key, None)
            return None
        return value

    async def get(self, key: str) -> str | None:
        return self._live(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        self._d[key] = (value, time.time() + ttl if ttl else None)

    async def delete(self, key: str) -> None:
        self._d.pop(key, None)

    async def incr(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        cur = self._live(key)
        exp = self._d[key][1] if cur is not None else (time.time() + ttl if ttl else None)
        new = int(cur or 0) + amount
        self._d[key] = (str(new), exp)
        return new

    async def incrbyfloat(self, key: str, amount: float, ttl: int | None = None) -> float:
        cur = self._live(key)
        exp = self._d[key][1] if cur is not None else (time.time() + ttl if ttl else None)
        new = float(cur or 0) + amount
        self._d[key] = (repr(new), exp)
        return new

    async def keys(self, prefix: str) -> list[str]:
        return [k for k in list(self._d) if k.startswith(prefix) and self._live(k) is not None]

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class RedisKV:
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis

        self._r = redis.from_url(url, decode_responses=True, socket_timeout=2, health_check_interval=30)

    async def get(self, key: str) -> str | None:
        return await self._r.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        await self._r.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        await self._r.delete(key)

    async def incr(self, key: str, amount: int = 1, ttl: int | None = None) -> int:
        pipe = self._r.pipeline()
        pipe.incrby(key, amount)
        if ttl:
            pipe.expire(key, ttl, nx=True)
        res = await pipe.execute()
        return int(res[0])

    async def incrbyfloat(self, key: str, amount: float, ttl: int | None = None) -> float:
        pipe = self._r.pipeline()
        pipe.incrbyfloat(key, amount)
        if ttl:
            pipe.expire(key, ttl, nx=True)
        res = await pipe.execute()
        return float(res[0])

    async def keys(self, prefix: str) -> list[str]:
        return [k async for k in self._r.scan_iter(match=f"{prefix}*", count=200)]

    async def ping(self) -> bool:
        return bool(await self._r.ping())

    async def close(self) -> None:
        await self._r.aclose()


def build_kv(settings: Settings) -> KV:
    if settings.redis_url:
        return RedisKV(settings.redis_url)
    return MemoryKV()
