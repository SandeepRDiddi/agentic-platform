"""Kafka integration (skipped unless TEST_KAFKA_BOOTSTRAP is set; CI runs it against a broker service)."""

import asyncio
import os
import uuid

import pytest

from agentic.backbone.bus import KafkaBus
from agentic.config import Settings

BOOT = os.environ.get("TEST_KAFKA_BOOTSTRAP")
pytestmark = pytest.mark.skipif(not BOOT, reason="TEST_KAFKA_BOOTSTRAP not set")


async def test_kafka_roundtrip_commit_and_redelivery():
    topic, group = f"t-{uuid.uuid4().hex[:8]}", f"g-{uuid.uuid4().hex[:8]}"
    bus = KafkaBus(Settings(kafka_bootstrap=BOOT, task_timeout_seconds=30))
    await bus.start()
    try:
        for i in range(3):
            await bus.publish(topic, f"k{i}", {"i": i}, headers={"traceparent": "00-abc", "attempt": "0"})
        c1 = await bus.consumer([topic], group)
        got = []
        async for m in c1:
            got.append(m.value["i"])
            assert m.headers["attempt"] == "0"
            if m.value["i"] == 0:
                await c1.commit(m)
            if len(got) == 3:
                break
        await c1.stop()
        c2 = await bus.consumer([topic], group)  # uncommitted offsets are redelivered
        again = []
        async for m in c2:
            again.append(m.value["i"])
            if len(again) == 2:
                break
        await c2.stop()
        assert sorted(got) == [0, 1, 2] and sorted(again) == [1, 2]
    finally:
        await asyncio.wait_for(bus.stop(), 10)
