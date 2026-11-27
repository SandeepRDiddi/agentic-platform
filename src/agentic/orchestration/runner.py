"""Runs tasks through the graph and executes human-approved actions."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agentic.common import logging as clog
from agentic.common.errors import (
    BudgetExceeded,
    Conflict,
    ForbiddenError,
    GuardrailViolation,
    KillSwitchEngaged,
    RateLimited,
    UpstreamError,
)
from agentic.common.metrics import AGENT_RUN_SECONDS, AGENT_RUNS, APPROVALS
from agentic.common.schemas import (
    Citation,
    PendingApproval,
    Principal,
    RunResult,
    Task,
    TaskStatus,
)
from agentic.container import Container
from agentic.knowledge.service import Knowledge
from agentic.models.client import LLMClient
from agentic.models.router import ModelRouter
from agentic.orchestration.graph import build_graph
from agentic.tools.client import McpToolsClient, ToolsClient
from agentic.trust.fallbacks import DEGRADED_ANSWER
from agentic.trust.tracing import span

log = logging.getLogger(__name__)


class RetryableError(Exception):
    """Transient failure (model/tool outage after all fallbacks). The worker retries these."""


class Orchestrator:
    def __init__(self, c: Container, tools: ToolsClient | None = None) -> None:
        self.c = c
        self.llm = LLMClient(c)
        self.router = ModelRouter(self.llm, c.settings)
        self.knowledge = Knowledge(c, self.llm, self.router)
        self.tools = tools or McpToolsClient(c.settings.mcp_url)
        self.graph = build_graph(c, self.router, self.knowledge, self.tools)

    # ── task execution ───────────────────────────────────────────────────────────────────
    async def execute(self, task: Task) -> RunResult:
        """Run one task to a terminal-or-awaiting state. Raises RetryableError on transient failure."""
        clog.bind(task_id=task.id, tenant=task.tenant)
        t0 = time.perf_counter()
        await self.c.tasks.update(task.id, status=TaskStatus.RUNNING, event={"type": "started"})
        await self.c.audit.record(actor=task.user_id, actor_type="user", action="task.start", resource=task.id,
                                  decision="allow", tenant=task.tenant, task_id=task.id)
        state: dict[str, Any] = {
            "task_id": task.id, "tenant": task.tenant, "user_id": task.user_id,
            "conversation_id": task.conversation_id, "query": task.query, "steps": 0, "revisions": 0,
            "trace": [], "errors": [], "usage": {}, "pending_approvals": [], "action_results": [],
        }
        try:
            with span("orchestrator.run", **{"agentic.task_id": task.id, "enduser.id": task.user_id}):
                async with asyncio.timeout(self.c.settings.task_timeout_seconds):
                    final = await self.graph.ainvoke(state, config={"recursion_limit": 60})
        except KillSwitchEngaged as e:
            return await self._finish(task, TaskStatus.KILLED, t0, error=e.message, answer="This task was stopped by an administrator.")
        except (BudgetExceeded, RateLimited, GuardrailViolation) as e:
            return await self._finish(task, TaskStatus.FAILED, t0, error=e.message, answer=e.message)
        except (UpstreamError, TimeoutError) as e:
            await self.c.tasks.update(task.id, event={"type": "retryable_error", "error": str(e)[:200]})
            raise RetryableError(str(e)) from e
        status = TaskStatus(final.get("status", "failed"))
        result = RunResult(
            task_id=task.id, status=status, answer=final.get("answer"),
            citations=[Citation(**x) for x in final.get("citations", [])],
            pending_approvals=[PendingApproval(**x) for x in final.get("pending_approvals", [])],
            steps=final.get("trace", []), usage=final.get("usage", {}), error=final.get("blocked_reason"),
        )
        return await self._finish(task, status, t0, result=result)

    async def _finish(self, task: Task, status: TaskStatus, t0: float, *, result: RunResult | None = None,
                      error: str | None = None, answer: str | None = None) -> RunResult:
        result = result or RunResult(task_id=task.id, status=status, answer=answer, error=error)
        await self.c.tasks.update(task.id, status=status, result=result, event={"type": status.value})
        await self.c.audit.record(actor="orchestrator", actor_type="system", action="task.finish", resource=task.id,
                                  decision=status.value, tenant=task.tenant, task_id=task.id,
                                  details={"usage": result.usage, "error": result.error})
        AGENT_RUNS.labels(status.value).inc()
        AGENT_RUN_SECONDS.observe(time.perf_counter() - t0)
        return result

    async def fail(self, task: Task, reason: str) -> RunResult:
        """Terminal failure after retries are exhausted."""
        return await self._finish(task, TaskStatus.FAILED, time.perf_counter(), error=reason, answer=DEGRADED_ANSWER)

    # ── human approvals ──────────────────────────────────────────────────────────────────
    async def decide(self, approval_id: str, approve: bool, decider: Principal, reason: str | None) -> PendingApproval:
        c, s = self.c, self.c.settings
        a = await c.approvals.get(approval_id)
        if a.tenant != decider.tenant:
            raise ForbiddenError("approval belongs to another tenant")
        who = decider.user_id
        allow = s.approver_list
        if allow and "*" not in allow and who not in allow:
            raise ForbiddenError(f"{who} is not an authorised approver")
        if s.approval_four_eyes and who == a.requested_by:
            raise ForbiddenError("four-eyes policy: you cannot approve your own request")
        await c.killswitch.check(tenant=a.tenant, task_id=a.task_id)
        a = await c.approvals.decide(approval_id, "approved" if approve else "denied", who, reason)
        APPROVALS.labels(a.status).inc()
        await c.audit.record(actor=who, actor_type="user", action="approval.decision", resource=a.tool,
                             decision=a.status, tenant=a.tenant, task_id=a.task_id,
                             details={"approval_id": a.id, "reason": reason, "args": a.args})
        if approve:
            tok = c.identity.issue("executor", a.tenant, on_behalf_of=a.requested_by)
            try:
                out = await self.tools.call(a.tool, {**a.args, "approval_id": a.id}, token=tok)
                a = await c.approvals.get(a.id)
                if a.status != "executed":  # tool server finalises; belt and braces
                    a = await c.approvals.finish(a.id, "executed", out if isinstance(out, dict) else {"result": out})
            except Exception as e:
                log.warning("approved action failed: %s", e)
                a = await c.approvals.finish(a.id, "failed", {"error": str(e)[:300]})
        await self._refresh_task(a)
        return a

    async def _refresh_task(self, a: PendingApproval) -> None:
        try:
            task = await self.c.tasks.get(a.task_id)
        except Exception:
            return
        apps = [x for x in await self.c.approvals.list(task.tenant) if x.task_id == a.task_id]
        if task.result:
            task.result.pending_approvals = apps
            note = {"executed": "was executed", "denied": "was denied", "failed": "failed to execute"}.get(a.status)
            if note and task.result.answer:
                task.result.answer += f"\n\n`{a.tool}` {note} (approval {a.id})."
        done = all(x.status != "pending" for x in apps)
        await self.c.tasks.update(
            a.task_id, status=TaskStatus.SUCCEEDED if done else TaskStatus.AWAITING_APPROVAL,
            result=task.result, event={"type": f"approval.{a.status}", "approval_id": a.id})


def conflict(msg: str) -> Conflict:
    return Conflict(msg)
