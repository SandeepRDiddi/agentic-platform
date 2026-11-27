"""Task and approval persistence (Postgres in prod, in-memory for dev/tests)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Protocol

from agentic.common.db import Database
from agentic.common.errors import Conflict, NotFound
from agentic.common.schemas import PendingApproval, RunResult, Task, TaskStatus, now


def args_hash(tool: str, args: dict[str, Any]) -> str:
    return hashlib.sha256(
        (tool + json.dumps(args, sort_keys=True, default=str)).encode()
    ).hexdigest()


class TaskRepo(Protocol):
    async def create(self, task: Task) -> Task: ...
    async def get(self, task_id: str, tenant: str | None = None) -> Task: ...
    async def claim(self, task_id: str) -> Task | None: ...
    async def update(
        self, task_id: str, *, status: TaskStatus | None = None,
        result: RunResult | None = None, event: dict[str, Any] | None = None,
    ) -> Task: ...


class ApprovalRepo(Protocol):
    async def create(self, a: PendingApproval) -> PendingApproval: ...
    async def get(self, approval_id: str) -> PendingApproval: ...
    async def list(self, tenant: str, status: str | None = None) -> list[PendingApproval]: ...
    async def decide(self, approval_id: str, status: str, decided_by: str, reason: str | None) -> PendingApproval: ...
    async def finish(self, approval_id: str, status: str, result: dict[str, Any]) -> PendingApproval: ...
    async def claim_execution(self, approval_id: str) -> PendingApproval | None: ...


# ── in-memory ─────────────────────────────────────────────────────────────────────────────
class MemoryTaskRepo:
    def __init__(self) -> None:
        self._t: dict[str, Task] = {}

    async def create(self, task: Task) -> Task:
        self._t[task.id] = task.model_copy(deep=True)
        return task

    async def get(self, task_id: str, tenant: str | None = None) -> Task:
        t = self._t.get(task_id)
        if not t or (tenant and t.tenant != tenant):
            raise NotFound(f"task {task_id} not found")
        return t.model_copy(deep=True)

    async def claim(self, task_id: str) -> Task | None:
        """Atomically move queued -> running. Returns None if someone else owns it / already done."""
        t = self._t.get(task_id)
        if not t or t.status not in (TaskStatus.QUEUED, TaskStatus.RUNNING):
            return None
        t.status = TaskStatus.RUNNING
        t.attempts += 1
        t.updated_at = now()
        return t.model_copy(deep=True)

    async def update(self, task_id, *, status=None, result=None, event=None) -> Task:
        t = self._t.get(task_id)
        if not t:
            raise NotFound(f"task {task_id} not found")
        if status:
            t.status = status
        if result:
            t.result = result
        if event:
            t.events.append({"ts": now().isoformat(), **event})
        t.updated_at = now()
        return t.model_copy(deep=True)


class MemoryApprovalRepo:
    def __init__(self) -> None:
        self._a: dict[str, PendingApproval] = {}

    async def create(self, a: PendingApproval) -> PendingApproval:
        self._a[a.id] = a.model_copy(deep=True)
        return a

    async def get(self, approval_id: str) -> PendingApproval:
        a = self._a.get(approval_id)
        if not a:
            raise NotFound(f"approval {approval_id} not found")
        return a.model_copy(deep=True)

    async def list(self, tenant: str, status: str | None = None) -> list[PendingApproval]:
        return [
            a.model_copy(deep=True)
            for a in self._a.values()
            if a.tenant == tenant and (status is None or a.status == status)
        ]

    async def decide(self, approval_id, status, decided_by, reason) -> PendingApproval:
        a = self._a.get(approval_id)
        if not a:
            raise NotFound(f"approval {approval_id} not found")
        if a.status != "pending":
            raise Conflict(f"approval already {a.status}")
        a.status, a.decided_by, a.reason, a.decided_at = status, decided_by, reason, now()
        return a.model_copy(deep=True)

    async def finish(self, approval_id, status, result) -> PendingApproval:
        a = self._a[approval_id]
        a.status, a.result = status, result
        return a.model_copy(deep=True)

    async def claim_execution(self, approval_id) -> PendingApproval | None:
        """Single-use: approved -> executed exactly once (replay protection)."""
        a = self._a.get(approval_id)
        if not a or a.status != "approved":
            return None
        a.status = "executed"
        return a.model_copy(deep=True)


# ── postgres ──────────────────────────────────────────────────────────────────────────────
def _task(r: Any) -> Task:
    return Task(
        id=r["id"], tenant=r["tenant"], user_id=r["user_id"], conversation_id=r["conversation_id"],
        query=r["query"], status=TaskStatus(r["status"]),
        result=RunResult(**r["result"]) if r["result"] else None,
        reply_to=r["reply_to"], attempts=r["attempts"], events=r["events"],
        created_at=r["created_at"], updated_at=r["updated_at"],
    )


class PgTaskRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def create(self, task: Task) -> Task:
        await self.db.execute(
            "INSERT INTO tasks (id, tenant, user_id, conversation_id, query, status, reply_to, events)"
            " VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
            task.id, task.tenant, task.user_id, task.conversation_id, task.query,
            task.status.value, task.reply_to, task.events,
        )
        return task

    async def get(self, task_id: str, tenant: str | None = None) -> Task:
        r = await self.db.fetchrow("SELECT * FROM tasks WHERE id=$1", task_id)
        if not r or (tenant and r["tenant"] != tenant):
            raise NotFound(f"task {task_id} not found")
        return _task(r)

    async def claim(self, task_id: str) -> Task | None:
        r = await self.db.fetchrow(
            "UPDATE tasks SET status='running', attempts=attempts+1, updated_at=now()"
            " WHERE id=$1 AND status IN ('queued','running') RETURNING *",
            task_id,
        )
        return _task(r) if r else None

    async def update(self, task_id, *, status=None, result=None, event=None) -> Task:
        async with self.db.transaction() as c:
            r = await c.fetchrow("SELECT * FROM tasks WHERE id=$1 FOR UPDATE", task_id)
            if not r:
                raise NotFound(f"task {task_id} not found")
            events = list(r["events"])
            if event:
                events.append({"ts": now().isoformat(), **event})
            r = await c.fetchrow(
                "UPDATE tasks SET status=COALESCE($2,status), result=COALESCE($3,result),"
                " events=$4, updated_at=now() WHERE id=$1 RETURNING *",
                task_id, status.value if status else None,
                result.model_dump(mode="json") if result else None, events,
            )
        return _task(r)


def _approval(r: Any) -> PendingApproval:
    return PendingApproval(**{k: r[k] for k in r.keys()})


class PgApprovalRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def create(self, a: PendingApproval) -> PendingApproval:
        await self.db.execute(
            "INSERT INTO approvals (id, task_id, tenant, tool, args, args_hash, status, requested_by)"
            " VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
            a.id, a.task_id, a.tenant, a.tool, a.args, a.args_hash, a.status, a.requested_by,
        )
        return a

    async def get(self, approval_id: str) -> PendingApproval:
        r = await self.db.fetchrow("SELECT * FROM approvals WHERE id=$1", approval_id)
        if not r:
            raise NotFound(f"approval {approval_id} not found")
        return _approval(r)

    async def list(self, tenant: str, status: str | None = None) -> list[PendingApproval]:
        rows = await self.db.fetch(
            "SELECT * FROM approvals WHERE tenant=$1 AND ($2::text IS NULL OR status=$2)"
            " ORDER BY created_at DESC LIMIT 200",
            tenant, status,
        )
        return [_approval(r) for r in rows]

    async def decide(self, approval_id, status, decided_by, reason) -> PendingApproval:
        r = await self.db.fetchrow(
            "UPDATE approvals SET status=$2, decided_by=$3, reason=$4, decided_at=now()"
            " WHERE id=$1 AND status='pending' RETURNING *",
            approval_id, status, decided_by, reason,
        )
        if not r:
            cur = await self.get(approval_id)  # raises NotFound if absent
            raise Conflict(f"approval already {cur.status}")
        return _approval(r)

    async def finish(self, approval_id, status, result) -> PendingApproval:
        r = await self.db.fetchrow(
            "UPDATE approvals SET status=$2, result=$3 WHERE id=$1 RETURNING *",
            approval_id, status, result,
        )
        return _approval(r)

    async def claim_execution(self, approval_id) -> PendingApproval | None:
        r = await self.db.fetchrow(
            "UPDATE approvals SET status='executed' WHERE id=$1 AND status='approved' RETURNING *",
            approval_id,
        )
        return _approval(r) if r else None


__all__ = [
    "ApprovalRepo", "MemoryApprovalRepo", "MemoryTaskRepo", "PgApprovalRepo", "PgTaskRepo",
    "TaskRepo", "args_hash", "datetime",
]
