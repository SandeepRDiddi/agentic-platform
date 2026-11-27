"""AI gateway core (box 2): auth'd, rate-limited, budgeted, cached, routed LLM access.

All model traffic in the platform goes through here, so cost, caching, kill switch and
provider failover are enforced in exactly one place.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from agentic.common.errors import BadRequest, BudgetExceeded, RateLimited, UpstreamError
from agentic.common.metrics import (
    LLM_CACHE_HITS,
    LLM_COST,
    LLM_LATENCY,
    LLM_REQUESTS,
    LLM_TOKENS,
    RATE_LIMITED,
)
from agentic.common.schemas import Principal
from agentic.container import Container
from agentic.gateway import mock_llm
from agentic.trust.fallbacks import CircuitBreaker, run_with_fallbacks
from agentic.trust.tracing import llm_attrs, span

log = logging.getLogger(__name__)
ALIASES = ("small", "large", "fallback", "guard")


class GatewayService:
    def __init__(self, c: Container) -> None:
        self.c = c
        self.s = c.settings
        self.breaker = CircuitBreaker(failure_threshold=5, recovery_seconds=30)
        routes = self.s.model_routes()
        self.routes: dict[str, list[str]] = {
            "small": routes.get("small") or [self.s.model_small],
            "large": routes.get("large") or [self.s.model_large],
            "fallback": routes.get("fallback") or [self.s.model_fallback],
            "guard": routes.get("guard") or [self.s.model_guard],
        }

    # ── policy: killswitch → rate limit → budget ─────────────────────────────────────────
    async def admit(self, p: Principal, *, cost_check: bool = True) -> None:
        agent = p.sub if p.kind == "agent" else None
        await self.c.killswitch.check(tenant=p.tenant, agent=agent)
        minute = int(time.time() // 60)
        n = await self.c.kv.incr(f"rl:{p.tenant}:{p.sub}:{minute}", ttl=70)
        if n > self.s.rate_limit_per_minute:
            RATE_LIMITED.labels(p.tenant).inc()
            raise RateLimited("rate limit exceeded", retry_after=60 - int(time.time()) % 60)
        if cost_check:
            spent = float(await self.c.kv.get(self._spend_key(p.tenant)) or 0)
            if spent >= self.s.tenant_daily_budget_usd:
                raise BudgetExceeded(
                    f"daily LLM budget of ${self.s.tenant_daily_budget_usd:.2f} exhausted",
                    spent_usd=round(spent, 4),
                )

    @staticmethod
    def _spend_key(tenant: str) -> str:
        return f"spend:{tenant}:{datetime.now(UTC):%Y%m%d}"

    async def spend(self, tenant: str) -> float:
        return float(await self.c.kv.get(self._spend_key(tenant)) or 0)

    # ── chat ─────────────────────────────────────────────────────────────────────────────
    async def chat(self, p: Principal, body: dict[str, Any]) -> dict[str, Any]:
        alias = body.get("model", "small")
        if alias not in ALIASES:
            raise BadRequest(f"unknown model alias {alias!r}; use one of {ALIASES}")
        messages = body.get("messages") or []
        if not messages:
            raise BadRequest("messages required")
        await self.admit(p)
        temperature = body.get("temperature", 0)
        cache_key = None
        if temperature == 0 and not body.get("no_cache"):
            cache_key = "llmcache:" + hashlib.sha256(json.dumps(
                [p.tenant, alias, messages, body.get("max_tokens"), body.get("response_format")],
                sort_keys=True, default=str).encode()).hexdigest()
            hit = await self.c.kv.get(cache_key)
            if hit:
                LLM_CACHE_HITS.inc()
                LLM_REQUESTS.labels(alias, "cache", "hit").inc()
                out = json.loads(hit)
                out["x_agentic"]["cached"] = True
                out["x_agentic"]["cost_usd"] = 0.0
                return out

        async def call(model: str) -> dict[str, Any]:
            return await self._provider_chat(alias, model, messages, body)

        t0 = time.perf_counter()
        with span("gateway.llm", **{"agentic.llm.alias": alias, "enduser.id": p.user_id}) as sp:
            try:
                res, model_used, failover = await run_with_fallbacks(
                    [(m, (lambda m=m: call(m))) for m in self.routes[alias]], breaker=self.breaker
                )
            except Exception as e:
                LLM_REQUESTS.labels(alias, "-", "error").inc()
                raise e if isinstance(e, UpstreamError) else UpstreamError(f"provider error: {e}") from e
            LLM_LATENCY.labels(alias).observe(time.perf_counter() - t0)
            llm_attrs(sp, model=res["model"], alias=alias, prompt_tokens=res["prompt_tokens"],
                      completion_tokens=res["completion_tokens"], cost=res["cost_usd"],
                      cached=False, finish_reason=res["finish_reason"])
            sp.set_attribute("agentic.llm.provider_failover", failover)
        LLM_REQUESTS.labels(alias, res["model"], "ok").inc()
        LLM_TOKENS.labels(res["model"], "prompt").inc(res["prompt_tokens"])
        LLM_TOKENS.labels(res["model"], "completion").inc(res["completion_tokens"])
        if res["cost_usd"]:
            LLM_COST.labels(p.tenant).inc(res["cost_usd"])
            await self.c.kv.incrbyfloat(self._spend_key(p.tenant), res["cost_usd"], ttl=2 * 86400)
        out = {
            "id": f"chatcmpl-{hashlib.sha1(str(t0).encode()).hexdigest()[:16]}",  # noqa: S324
            "object": "chat.completion", "model": res["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": res["content"]},
                         "finish_reason": res["finish_reason"]}],
            "usage": {"prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
                      "total_tokens": res["prompt_tokens"] + res["completion_tokens"]},
            "x_agentic": {"alias": alias, "cost_usd": res["cost_usd"], "cached": False,
                          "provider_failover": failover, "provider_model": model_used},
        }
        if cache_key and res["finish_reason"] == "stop":
            await self.c.kv.set(cache_key, json.dumps(out), ttl=self.s.cache_ttl_seconds)
        return out

    async def _provider_chat(self, alias: str, model: str, messages: list[dict], body: dict) -> dict[str, Any]:
        if self.s.llm_mode == "mock":
            return mock_llm.complete(alias, messages)
        import litellm

        kwargs: dict[str, Any] = {
            "model": model, "messages": messages, "temperature": body.get("temperature", 0),
            "timeout": self.s.llm_timeout_seconds, "num_retries": self.s.llm_num_retries,
        }
        if body.get("max_tokens"):
            kwargs["max_tokens"] = body["max_tokens"]
        if body.get("response_format"):
            kwargs["response_format"] = body["response_format"]
        try:
            r = await litellm.acompletion(**kwargs)
        except Exception as e:
            raise UpstreamError(f"{model}: {type(e).__name__}: {e}") from e
        choice = r.choices[0]
        usage = getattr(r, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=r) or 0)
        except Exception:
            cost = 0.0
        return {
            "content": choice.message.content or "", "model": getattr(r, "model", model) or model,
            "finish_reason": choice.finish_reason or "stop",
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0, "cost_usd": cost,
        }

    # ── embeddings ───────────────────────────────────────────────────────────────────────
    async def embed(self, p: Principal, texts: list[str]) -> dict[str, Any]:
        if not texts or len(texts) > 256:
            raise BadRequest("input must contain 1..256 strings")
        await self.admit(p)
        dim = self.s.embedding_dim
        if self.s.llm_mode == "mock":
            vecs = [mock_llm.hash_embedding(t, dim) for t in texts]
        else:
            import litellm

            try:
                r = await litellm.aembedding(
                    model=self.s.model_embedding, input=texts, dimensions=dim,
                    timeout=self.s.llm_timeout_seconds,
                )
            except Exception as e:
                raise UpstreamError(f"embedding error: {e}") from e
            vecs = [d["embedding"] for d in r.data]
        return {"object": "list", "model": self.s.model_embedding if self.s.llm_mode == "live" else "mock/hash",
                "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]}
