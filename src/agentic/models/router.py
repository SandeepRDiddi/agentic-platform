"""Model router (box 4): small model first → big model if needed → fallback model.

Escalation triggers (small → large): provider error, truncated output, empty output,
output that fails the agent's JSON schema, or self-reported confidence below the threshold.
Fallback (any → fallback model): provider error / open circuit, or large model quality failure.
Policy errors (kill switch, budget, rate limit) are never retried on another model.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from agentic.common.errors import (
    AgenticError,
    BudgetExceeded,
    KillSwitchEngaged,
    RateLimited,
    UpstreamError,
)
from agentic.common.metrics import ROUTER_ESCALATIONS, ROUTER_FALLBACKS
from agentic.common.schemas import LLMResult
from agentic.config import Settings
from agentic.models.client import LLMClient
from agentic.trust.fallbacks import CircuitBreaker
from agentic.trust.tracing import span

log = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)
_POLICY = (KillSwitchEngaged, BudgetExceeded, RateLimited)


def extract_json(text: str) -> Any:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(t[start : end + 1])


@dataclass
class Routed:
    result: LLMResult
    parsed: BaseModel | None


class ModelRouter:
    def __init__(self, llm: LLMClient, settings: Settings, breaker: CircuitBreaker | None = None) -> None:
        self.llm, self.s = llm, settings
        self.breaker = breaker or CircuitBreaker(failure_threshold=5, recovery_seconds=30)

    @staticmethod
    def is_hard(messages: list[dict[str, str]]) -> bool:
        return sum(len(m["content"]) for m in messages) > 12000

    def _evaluate(self, alias: str, res: LLMResult, schema: type[M] | None) -> tuple[M | None, str | None]:
        if res.finish_reason == "length":
            return None, "truncated"
        if not res.content.strip():
            return None, "empty"
        if schema is None:
            return None, None
        try:
            parsed = schema.model_validate(extract_json(res.content))
        except (ValueError, ValidationError):
            return None, "invalid_output"
        conf = getattr(parsed, "confidence", None)
        if alias == "small" and conf is not None and conf < self.s.escalation_confidence:
            return parsed, "low_confidence"
        return parsed, None

    async def complete(
        self, *, agent: str, system: str, user: str, tenant: str, user_id: str | None,
        schema: type[M] | None = None, hard: bool = False, max_tokens: int | None = 1500,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[LLMResult, M | None]:
        messages = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}]
        chain = ["large", "fallback"] if (hard or self.is_hard(messages)) else ["small", "large", "fallback"]
        first = chain[0]
        last_err: BaseException | None = None
        reason: str | None = None
        with span("router.complete", **{"agentic.agent": agent, "agentic.router.start_tier": first}) as sp:
            for alias in chain:
                try:
                    self.breaker.before(alias)
                    res = await self.llm.chat(
                        alias, messages, agent=agent, tenant=tenant, user=user_id,
                        max_tokens=max_tokens, json_mode=schema is not None,
                    )
                    self.breaker.success(alias)
                except _POLICY:
                    raise
                except (UpstreamError, AgenticError) as e:
                    self.breaker.failure(alias)
                    last_err = e
                    log.warning("router: %s failed (%s)", alias, e)
                    if alias != "fallback":
                        ROUTER_FALLBACKS.labels(alias).inc()
                    continue
                parsed, reason = self._evaluate(alias, res, schema)
                if reason is None:
                    res.escalated = alias == "large" and first == "small"
                    res.fallback_used = alias == "fallback"
                    sp.set_attribute("agentic.router.tier_used", alias)
                    sp.set_attribute("agentic.router.escalated", res.escalated)
                    sp.set_attribute("agentic.router.fallback", res.fallback_used)
                    return res, parsed  # type: ignore[return-value]
                if reason == "low_confidence" and chain[-1] == alias:
                    return res, parsed  # type: ignore[return-value]
                ROUTER_ESCALATIONS.labels(reason).inc()
                last_err = UpstreamError(f"{alias} output rejected: {reason}")
                log.info("router: %s output rejected (%s), escalating", alias, reason)
        raise last_err or UpstreamError("all models failed")
