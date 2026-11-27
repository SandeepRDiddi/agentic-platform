"""LLM client: every call goes through the AI gateway with the calling agent's identity."""

from __future__ import annotations

from typing import Any

from agentic.common.errors import AgenticError, UpstreamError
from agentic.common.schemas import LLMResult
from agentic.container import Container


class LLMClient:
    def __init__(self, c: Container) -> None:
        self.c = c
        self._http = None

    @property
    def http(self):  # lazily created so tests can swap Container.http first
        if self._http is None:
            self._http = self.c.client(self.c.settings.gateway_url)
        return self._http

    def _headers(self, agent: str, tenant: str, user: str | None) -> dict[str, str]:
        from agentic.trust.tracing import inject_headers

        return {
            "authorization": f"Bearer {self.c.identity.issue(agent, tenant, on_behalf_of=user)}",
            **inject_headers(),
        }

    async def chat(
        self, alias: str, messages: list[dict[str, str]], *, agent: str, tenant: str,
        user: str | None = None, temperature: float = 0, max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        body: dict[str, Any] = {"model": alias, "messages": messages, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = await self.http.post("/v1/chat/completions", json=body, headers=self._headers(agent, tenant, user))
        except Exception as e:
            raise UpstreamError(f"gateway unreachable: {e}") from e
        if r.status_code != 200:
            err = (r.json().get("error") if r.headers.get("content-type", "").startswith("application/json") else None) or {}
            # Policy errors (kill switch, budget, rate limit, auth) must surface as themselves.
            from agentic.common import errors as E

            for cls in (E.KillSwitchEngaged, E.BudgetExceeded, E.RateLimited, E.AuthError, E.ForbiddenError, E.BadRequest):
                if err.get("code") == cls.code:
                    raise cls(err.get("message", ""), **(err.get("details") or {}))
            raise AgenticError(err.get("message", f"gateway {r.status_code}")) if r.status_code < 500 else UpstreamError(
                err.get("message", f"gateway {r.status_code}"))
        d = r.json()
        x = d.get("x_agentic", {})
        return LLMResult(
            content=d["choices"][0]["message"]["content"], model=d["model"],
            finish_reason=d["choices"][0]["finish_reason"], prompt_tokens=d["usage"]["prompt_tokens"],
            completion_tokens=d["usage"]["completion_tokens"], cost_usd=x.get("cost_usd", 0.0),
            cached=x.get("cached", False), tier=alias,
        )

    async def embed(self, texts: list[str], *, agent: str, tenant: str, user: str | None = None) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            r = await self.http.post("/v1/embeddings", json={"input": texts[i : i + 128]},
                                     headers=self._headers(agent, tenant, user))
            if r.status_code != 200:
                raise UpstreamError(f"embedding failed: {r.status_code} {r.text[:200]}")
            out.extend(d["embedding"] for d in r.json()["data"])
        return out
