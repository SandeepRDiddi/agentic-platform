"""Async worker (box 7): long tasks = events, not HTTP calls.  Deployed as K8s Deployment + KEDA.

Semantics
  * at-least-once: offset committed only after the task reaches a durable state
  * idempotent: TaskRepo.claim() only succeeds for queued/running tasks → duplicate deliveries are no-ops
  * retries: transient failures re-published with attempt+1 and exponential backoff
  * poison messages: after WORKER_MAX_ATTEMPTS → DLQ topic + task marked failed
  * kill switch: checked before work starts; engaged → task KILLED, never retried
  * SIGTERM: stop consuming, finish in-flight task, exit (K8s terminationGracePeriodSeconds)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from prometheus_client import start_http_server

from agentic.backbone.bus import Message
from agentic.common.errors import KillSwitchEngaged
from agentic.common.logging import setup_logging
from agentic.common.metrics import TASKS_IN_FLIGHT, WORKER_RESULTS
from agentic.common.schemas import TaskStatus
from agentic.config import get_settings
from agentic.container import Container
from agentic.orchestration.runner import Orchestrator, RetryableError
from agentic.trust.tracing import extract_context, setup_tracing, span

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, c: Container, orch: Orchestrator) -> None:
        self.c, self.orch, self.s = c, orch, c.settings
        self._consumers: list = []
        self._stop = asyncio.Event()

    async def _publish_result(self, task_id: str) -> None:
        t = await self.c.tasks.get(task_id)
        await self.c.bus.publish(self.s.topic_results, task_id, {
            "task_id": task_id, "tenant": t.tenant, "user_id": t.user_id, "status": t.status.value,
            "answer": t.result.answer if t.result else None, "reply_to": t.reply_to,
            "pending_approvals": [a.model_dump(mode="json") for a in (t.result.pending_approvals if t.result else [])],
        })

    async def handle(self, msg: Message) -> None:
        task_id, tenant = msg.value["task_id"], msg.value.get("tenant")
        attempt = int(msg.headers.get("attempt", "0"))
        with extract_context(msg.headers), span("worker.handle", **{"agentic.task_id": task_id, "agentic.attempt": attempt}):
            task = await self.c.tasks.claim(task_id)
            if task is None:
                WORKER_RESULTS.labels("duplicate").inc()
                return
            try:
                await self.c.killswitch.check(tenant=tenant, task_id=task_id)
                await self.orch.execute(task)
                WORKER_RESULTS.labels("ok").inc()
            except KillSwitchEngaged as e:
                await self.orch._finish(task, TaskStatus.KILLED, time.perf_counter(), error=e.message,  # noqa: SLF001
                                        answer="This task was stopped by an administrator.")
                WORKER_RESULTS.labels("killed").inc()
            except (RetryableError, Exception) as e:
                if attempt + 1 < self.s.worker_max_attempts:
                    delay = min(2 ** attempt, 30) if self.s.env != "test" else 0
                    log.warning("task %s attempt %d failed (%s); retrying in %ss", task_id, attempt + 1, e, delay)
                    await asyncio.sleep(delay)
                    await self.c.bus.publish(self.s.topic_tasks, task_id, msg.value,
                                             headers={**msg.headers, "attempt": str(attempt + 1)})
                    WORKER_RESULTS.labels("retry").inc()
                    return
                log.error("task %s exhausted retries; dead-lettering: %s", task_id, e)
                await self.c.bus.publish(self.s.topic_dlq, task_id, {**msg.value, "error": str(e)[:500]}, headers=msg.headers)
                await self.orch.fail(task, str(e)[:300])
                WORKER_RESULTS.labels("dead_letter").inc()
            await self._publish_result(task_id)

    async def _loop(self, idx: int) -> None:
        consumer = await self.c.bus.consumer([self.s.topic_tasks], self.s.worker_group)
        self._consumers.append(consumer)
        async for msg in consumer:
            if self._stop.is_set():
                break
            TASKS_IN_FLIGHT.inc()
            try:
                await self.handle(msg)
                await consumer.commit(msg)
            except Exception:
                log.exception("worker %d: handler crashed; message will be redelivered", idx)
            finally:
                TASKS_IN_FLIGHT.dec()

    async def run(self) -> None:
        tasks = [asyncio.create_task(self._loop(i)) for i in range(self.s.worker_concurrency)]
        await self._stop.wait()
        for cns in self._consumers:
            await cns.stop()
        await asyncio.gather(*tasks, return_exceptions=True)

    def stop(self) -> None:
        self._stop.set()


async def amain() -> None:
    s = get_settings()
    setup_logging(s.log_level)
    setup_tracing(s, "agent-worker")
    c = Container.build(s)
    await c.start()
    start_http_server(9100)  # /metrics (also used as liveness probe)
    worker = Worker(c, Orchestrator(c))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker.stop)
    log.info("worker started: %d consumers on %s", s.worker_concurrency, s.topic_tasks)
    await worker.run()
    await c.close()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
