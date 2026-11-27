"""Wire and domain models shared by every service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:20]}"


class Principal(BaseModel):
    sub: str
    tenant: str = "default"
    scopes: list[str] = Field(default_factory=list)
    kind: Literal["user", "service", "agent"] = "user"
    on_behalf_of: str | None = None

    def has(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes

    @property
    def user_id(self) -> str:
        """The human (or service) this work is ultimately done for."""
        return self.on_behalf_of or self.sub


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"  # stopped by a guardrail
    KILLED = "killed"  # stopped by the kill switch

    @property
    def terminal(self) -> bool:
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.KILLED,
        }


class RunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(min_length=1, max_length=20000)
    conversation_id: str | None = None
    run_async: bool = Field(False, alias="async")
    reply_to: dict[str, Any] | None = None  # channel routing info (slack thread, teams conversation)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    id: str
    source: str
    snippet: str
    score: float = 0.0


class PendingApproval(BaseModel):
    id: str
    task_id: str
    tenant: str
    tool: str
    args: dict[str, Any]
    args_hash: str
    status: Literal["pending", "approved", "denied", "executed", "failed"] = "pending"
    requested_by: str
    decided_by: str | None = None
    decided_at: datetime | None = None
    reason: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=now)


class RunResult(BaseModel):
    task_id: str
    status: TaskStatus
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    pending_approvals: list[PendingApproval] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Task(BaseModel):
    id: str
    tenant: str
    user_id: str
    conversation_id: str | None = None
    query: str
    status: TaskStatus = TaskStatus.QUEUED
    result: RunResult | None = None
    reply_to: dict[str, Any] | None = None
    attempts: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LLMResult(BaseModel):
    content: str
    model: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    tier: str = "small"
    escalated: bool = False
    fallback_used: bool = False
