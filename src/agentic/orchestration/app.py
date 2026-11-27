"""Orchestrator API. Port 8002. Only reachable from the gateway (NetworkPolicy + gateway identity)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from agentic.common.errors import NotFound
from agentic.common.http import instrument_app, principal_dependency, require_scope
from agentic.common.logging import setup_logging
from agentic.common.schemas import PendingApproval, Principal, RunRequest, Task, TaskStatus, new_id
from agentic.config import get_settings
from agentic.container import Container
from agentic.orchestration.runner import Orchestrator, RetryableError
from agentic.trust.tracing import inject_headers, instrument_fastapi, setup_tracing


class DecisionIn(BaseModel):
    approve: bool
    reason: str | None = None


def create_app(container: Container | None = None, orchestrator: Orchestrator | None = None) -> FastAPI:
    settings = container.settings if container else get_settings()
    c = container or Container.build(settings)
    orch = orchestrator or Orchestrator(c)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(settings.log_level)
        setup_tracing(settings, "orchestrator")
        await c.start()
        yield
        await c.close()

    app = FastAPI(title="Orchestrator", version=settings.service_version, lifespan=lifespan)
    instrument_app(app, "orchestrator", ready=c.readiness)
    instrument_fastapi(app)
    app.state.container, app.state.orchestrator = c, orch
    who = principal_dependency(c.authn)

    @app.post("/v1/runs")
    async def run(body: RunRequest, p: Principal = Depends(require_scope("agent:run", who))):
        await c.killswitch.check(tenant=p.tenant)
        task = Task(id=new_id("tsk_"), tenant=p.tenant, user_id=p.user_id, conversation_id=body.conversation_id,
                    query=body.query, reply_to=body.reply_to, events=[{"type": "queued"}])
        await c.tasks.create(task)
        if body.run_async:
            await c.bus.publish(settings.topic_tasks, task.id,
                                {"task_id": task.id, "tenant": task.tenant, "user_id": task.user_id},
                                headers={"attempt": "0", **inject_headers()})
            from fastapi.responses import JSONResponse

            return JSONResponse({"task_id": task.id, "status": TaskStatus.QUEUED.value}, status_code=202)
        try:
            return await orch.execute(task)
        except RetryableError as e:
            return await orch.fail(task, str(e)[:200])

    @app.get("/v1/tasks/{task_id}")
    async def get_task(task_id: str, p: Principal = Depends(require_scope("agent:run", who))) -> Task:
        t = await c.tasks.get(task_id, tenant=p.tenant)
        if t.user_id != p.user_id:
            raise NotFound(f"task {task_id} not found")
        return t

    @app.get("/v1/approvals")
    async def approvals(status: str | None = "pending",
                        p: Principal = Depends(require_scope("approvals:read", who))) -> list[PendingApproval]:
        return await c.approvals.list(p.tenant, status)

    @app.post("/v1/approvals/{approval_id}/decision")
    async def decide(approval_id: str, body: DecisionIn,
                     p: Principal = Depends(require_scope("approvals:decide", who))) -> PendingApproval:
        return await orch.decide(approval_id, body.approve, p, body.reason)

    @app.delete("/v1/memory")
    async def forget(p: Principal = Depends(require_scope("memory:delete", who))) -> dict:
        n = await orch.knowledge.forget(p.tenant, p.user_id)
        await c.audit.record(actor=p.user_id, actor_type="user", action="memory.forget", decision="ok",
                             tenant=p.tenant, details={"deleted": n})
        return {"deleted": n}

    return app


def app_factory() -> FastAPI:
    return create_app()
