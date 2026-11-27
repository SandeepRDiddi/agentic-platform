"""Async backbone transport (box 7): Kafka in prod, in-process log for dev/tests.

Contract: at-least-once delivery, offsets committed only AFTER the handler finishes,
message key = task_id (per-task ordering), W3C trace context carried in headers.
Consumers must be idempotent (the worker uses TaskRepo.claim for that).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentic.config import Settings

log = logging.getLogger(__name__)


@dataclass
class Message:
    topic: str
    key: str
    value: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    offset: int = 0
    partition: int = 0


class Consumer(Protocol):
    def __aiter__(self) -> AsyncIterator[Message]: ...
    async def commit(self, msg: Message) -> None: ...
    async def stop(self) -> None: ...


class MessageBus(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def publish(
        self, topic: str, key: str, value: dict[str, Any], headers: dict[str, str] | None = None
    ) -> None: ...
    async def consumer(self, topics: list[str], group: str) -> Consumer: ...
    async def ping(self) -> bool: ...


# ── in-memory ─────────────────────────────────────────────────────────────────────────────
class _MemConsumer:
    def __init__(self, bus: MemoryBus, topics: list[str], group: str) -> None:
        self.bus, self.topics, self.group = bus, topics, group
        self._pos = {t: bus._committed.get((group, t), 0) for t in topics}
        self._stopped = False

    def __aiter__(self) -> AsyncIterator[Message]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Message]:
        while not self._stopped:
            for t in self.topics:
                log_ = self.bus._log.setdefault(t, [])
                if self._pos[t] < len(log_):
                    m = log_[self._pos[t]]
                    self._pos[t] += 1
                    yield m
                    break
            else:
                async with self.bus._cond:
                    try:
                        await asyncio.wait_for(self.bus._cond.wait(), timeout=0.2)
                    except TimeoutError:
                        pass

    async def commit(self, msg: Message) -> None:
        self.bus._committed[(self.group, msg.topic)] = max(
            self.bus._committed.get((self.group, msg.topic), 0), msg.offset + 1
        )

    async def stop(self) -> None:
        self._stopped = True


class MemoryBus:
    def __init__(self) -> None:
        self._log: dict[str, list[Message]] = {}
        self._committed: dict[tuple[str, str], int] = {}
        self._cond = asyncio.Condition()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def publish(self, topic, key, value, headers=None) -> None:
        log_ = self._log.setdefault(topic, [])
        log_.append(Message(topic, key, json.loads(json.dumps(value, default=str)), dict(headers or {}), len(log_)))
        async with self._cond:
            self._cond.notify_all()

    async def consumer(self, topics: list[str], group: str) -> Consumer:
        return _MemConsumer(self, topics, group)

    async def ping(self) -> bool:
        return True

    def messages(self, topic: str) -> list[Message]:  # test helper
        return list(self._log.get(topic, []))


# ── kafka ─────────────────────────────────────────────────────────────────────────────────
class _KafkaConsumer:
    def __init__(self, c: Any) -> None:
        self._c = c

    def __aiter__(self) -> AsyncIterator[Message]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Message]:
        async for r in self._c:
            try:
                value = json.loads(r.value)
            except Exception:
                log.error("dropping undecodable message %s[%s]@%s", r.topic, r.partition, r.offset)
                continue
            headers = {k: v.decode() for k, v in (r.headers or [])}
            yield Message(r.topic, (r.key or b"").decode(), value, headers, r.offset, r.partition)

    async def commit(self, msg: Message) -> None:
        from aiokafka import TopicPartition

        await self._c.commit({TopicPartition(msg.topic, msg.partition): msg.offset + 1})

    async def stop(self) -> None:
        await self._c.stop()


class KafkaBus:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._producer: Any = None

    def _security(self) -> dict[str, Any]:
        s = self.s
        kw: dict[str, Any] = {"security_protocol": s.kafka_security_protocol}
        if s.kafka_sasl_mechanism:
            kw.update(
                sasl_mechanism=s.kafka_sasl_mechanism,
                sasl_plain_username=s.kafka_sasl_username,
                sasl_plain_password=s.kafka_sasl_password,
            )
            if "SSL" in s.kafka_security_protocol:
                import ssl

                kw["ssl_context"] = ssl.create_default_context()
        return kw

    async def start(self) -> None:
        from aiokafka import AIOKafkaProducer

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.s.kafka_bootstrap, acks="all", enable_idempotence=True,
            compression_type="gzip", linger_ms=5, **self._security(),
        )
        await self._producer.start()

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()

    async def publish(self, topic, key, value, headers=None) -> None:
        h = [(k, v.encode()) for k, v in (headers or {}).items()]
        await self._producer.send_and_wait(
            topic, key=key.encode(), value=json.dumps(value, default=str).encode(), headers=h
        )

    async def consumer(self, topics: list[str], group: str) -> Consumer:
        from aiokafka import AIOKafkaConsumer

        c = AIOKafkaConsumer(
            *topics, bootstrap_servers=self.s.kafka_bootstrap, group_id=group,
            enable_auto_commit=False, auto_offset_reset="earliest",
            max_poll_interval_ms=(self.s.task_timeout_seconds + 120) * 1000,
            session_timeout_ms=30000, **self._security(),
        )
        await c.start()
        return _KafkaConsumer(c)

    async def ping(self) -> bool:
        return self._producer is not None and not self._producer._closed  # noqa: SLF001


def build_bus(settings: Settings) -> MessageBus:
    return KafkaBus(settings) if settings.kafka_bootstrap else MemoryBus()
