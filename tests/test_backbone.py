import asyncio

from agentic.backbone.bus import MemoryBus
from agentic.common.schemas import TaskStatus
from agentic.orchestration.runner import RetryableError


async def test_memory_bus_commit_semantics():
    bus = MemoryBus()
    for i in range(3):
        await bus.publish("t", f"k{i}", {"i": i})
    c1 = await bus.consumer(["t"], "g")
    seen = []
    async for m in c1:
        seen.append(m.value["i"])
        if m.value["i"] < 1:
            await c1.commit(m)  # only first is committed
        if len(seen) == 3:
            break
    await c1.stop()
    c2 = await bus.consumer(["t"], "g")  # restart: uncommitted messages are redelivered
    redelivered = []
    async for m in c2:
        redelivered.append(m.value["i"])
        if len(redelivered) == 2:
            break
    assert seen == [0, 1, 2] and redelivered == [1, 2]
    c3 = await bus.consumer(["t"], "other-group")  # groups are independent
    async for m in c3:
        assert m.value["i"] == 0
        break


async def submit_async(stack, query="What is NNDM?", sub="alice", **extra):
    r = await stack.api.post("/v1/agent/runs", json={"query": query, "async": True, **extra}, headers=stack.auth(sub))
    assert r.status_code == 202 and r.json()["status"] == "queued"
    return r.json()["task_id"]


async def wait_status(stack, task_id, want, timeout=10):
    for _ in range(int(timeout / 0.05)):
        t = await stack.c.tasks.get(task_id)
        if t.status in want:
            return t
        await asyncio.sleep(0.05)
    raise AssertionError(f"task stuck in {t.status}")


async def test_async_task_end_to_end_via_worker(stack):
    tid = await submit_async(stack)
    assert (await stack.c.tasks.get(tid)).status == TaskStatus.QUEUED
    runner = asyncio.create_task(stack.worker.run())
    try:
        t = await wait_status(stack, tid, {TaskStatus.SUCCEEDED})
        assert "Data Marketplace" in t.result.answer and t.attempts == 1
        # result event published for channel delivery
        for _ in range(100):
            if stack.c.bus.messages(stack.settings.topic_results):
                break
            await asyncio.sleep(0.02)
        (res,) = stack.c.bus.messages(stack.settings.topic_results)
        assert res.value["task_id"] == tid and res.value["status"] == "succeeded"
    finally:
        stack.worker.stop()
        await runner


async def test_duplicate_delivery_is_idempotent(stack):
    tid = await submit_async(stack)
    (msg,) = stack.c.bus.messages(stack.settings.topic_tasks)
    await stack.worker.handle(msg)
    t1 = await stack.c.tasks.get(tid)
    await stack.worker.handle(msg)  # at-least-once → redelivery must be a no-op
    t2 = await stack.c.tasks.get(tid)
    assert t1.status == TaskStatus.SUCCEEDED and t2.attempts == t1.attempts == 1
    assert len(stack.c.bus.messages(stack.settings.topic_results)) == 1


async def test_transient_failure_is_retried_then_succeeds(stack, monkeypatch):
    tid = await submit_async(stack)
    real = stack.orchestrator_core.execute
    calls = {"n": 0}

    async def flaky(task):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("model outage")
        return await real(task)

    monkeypatch.setattr(stack.orchestrator_core, "execute", flaky)
    runner = asyncio.create_task(stack.worker.run())
    try:
        t = await wait_status(stack, tid, {TaskStatus.SUCCEEDED, TaskStatus.FAILED})
        assert t.status == TaskStatus.SUCCEEDED and calls["n"] == 3 and t.attempts == 3
        assert stack.c.bus.messages(stack.settings.topic_dlq) == []
    finally:
        stack.worker.stop()
        await runner


async def test_poison_task_is_dead_lettered_and_marked_failed(stack, monkeypatch):
    tid = await submit_async(stack)

    async def always_fail(task):
        raise RetryableError("permanent outage")

    monkeypatch.setattr(stack.orchestrator_core, "execute", always_fail)
    runner = asyncio.create_task(stack.worker.run())
    try:
        t = await wait_status(stack, tid, {TaskStatus.FAILED})
        assert t.attempts == stack.settings.worker_max_attempts
        (dlq,) = stack.c.bus.messages(stack.settings.topic_dlq)
        assert dlq.value["task_id"] == tid and "permanent outage" in dlq.value["error"]
        assert "Nothing was changed" in t.result.answer
    finally:
        stack.worker.stop()
        await runner


async def test_kill_switch_stops_queued_tasks_without_retry(stack):
    tid = await submit_async(stack)
    await stack.c.killswitch.engage("tenant:default", "freeze", "root")
    (msg,) = stack.c.bus.messages(stack.settings.topic_tasks)
    await stack.worker.handle(msg)
    t = await stack.c.tasks.get(tid)
    assert t.status == TaskStatus.KILLED and stack.c.bus.messages(stack.settings.topic_dlq) == []


async def test_async_approval_flow_over_worker(stack):
    tid = await submit_async(stack, "Create a ticket for the failing nightly ingestion job")
    (msg,) = stack.c.bus.messages(stack.settings.topic_tasks)
    await stack.worker.handle(msg)
    t = await stack.c.tasks.get(tid)
    assert t.status == TaskStatus.AWAITING_APPROVAL and t.result.pending_approvals
    assert not stack.internal.state.tickets.rows  # nothing executed until a human approves
