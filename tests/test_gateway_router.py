import pytest

from agentic.common.errors import KillSwitchEngaged, UpstreamError
from agentic.orchestration.graph import Route
from conftest import stack_with


def llm_headers(s, agent="research", tenant="default"):
    return {"authorization": f"Bearer {s.c.identity.issue(agent, tenant)}"}


BODY = {"model": "small", "messages": [{"role": "user", "content": "hello"}]}


async def test_auth_required_and_scope_enforced(bare_stack):
    s = bare_stack
    assert (await s.api.post("/v1/chat/completions", json=BODY)).status_code == 401
    assert (await s.api.post("/v1/chat/completions", json=BODY, headers={"authorization": "Bearer junk"})).status_code == 401
    # a human user token has no llm:invoke scope → cannot call models directly
    assert (await s.api.post("/v1/chat/completions", json=BODY, headers=s.auth())).status_code == 403
    assert (await s.api.post("/v1/chat/completions", json=BODY, headers=llm_headers(s))).status_code == 200


async def test_api_key_auth():
    import hashlib
    import json

    key = "k-123456"
    keys = {hashlib.sha256(key.encode()).hexdigest(): {"sub": "svc", "tenant": "acme", "scopes": ["agent:run"]}}
    async with stack_with(api_keys_json=json.dumps(keys)) as s:
        r = await s.api.post("/v1/agent/runs", json={"query": "hi"}, headers={"x-api-key": key})
        assert r.status_code == 200
        assert (await s.api.post("/v1/agent/runs", json={"query": "hi"}, headers={"x-api-key": "wrong"})).status_code == 401


async def test_exact_match_cache_and_tenant_isolation(bare_stack):
    s = bare_stack
    a = (await s.api.post("/v1/chat/completions", json=BODY, headers=llm_headers(s))).json()
    b = (await s.api.post("/v1/chat/completions", json=BODY, headers=llm_headers(s))).json()
    other = (await s.api.post("/v1/chat/completions", json=BODY, headers=llm_headers(s, tenant="t2"))).json()
    assert a["x_agentic"]["cached"] is False and b["x_agentic"]["cached"] is True
    assert b["x_agentic"]["cost_usd"] == 0
    assert other["x_agentic"]["cached"] is False  # cache is tenant-scoped
    assert b["choices"] == a["choices"]


async def test_unknown_alias_and_empty_messages(bare_stack):
    s = bare_stack
    assert (await s.api.post("/v1/chat/completions", json={"model": "gpt-9", "messages": [{"role": "user", "content": "x"}]},
                             headers=llm_headers(s))).status_code == 400
    assert (await s.api.post("/v1/chat/completions", json={"model": "small", "messages": []}, headers=llm_headers(s))).status_code == 400


async def test_rate_limit_returns_429_with_retry_after():
    async with stack_with(rate_limit_per_minute=3) as s:
        codes = [(await s.api.post("/v1/chat/completions", json={**BODY, "no_cache": True}, headers=llm_headers(s))) for _ in range(5)]
        assert [r.status_code for r in codes[:3]] == [200, 200, 200]
        assert codes[3].status_code == 429 and "retry-after" in codes[3].headers


async def test_daily_budget_enforced_per_tenant():
    async with stack_with(tenant_daily_budget_usd=1e-6) as s:
        first = await s.api.post("/v1/chat/completions", json={**BODY, "no_cache": True}, headers=llm_headers(s))
        second = await s.api.post("/v1/chat/completions", json={**BODY, "no_cache": True}, headers=llm_headers(s))
        other = await s.api.post("/v1/chat/completions", json={**BODY, "no_cache": True}, headers=llm_headers(s, tenant="t2"))
        assert first.status_code == 200 and second.status_code == 402 and other.status_code == 200
        spend = await s.api.get("/v1/admin/spend", headers=s.auth(scopes=["admin:spend"]))
        assert spend.json()["spent_usd_today"] > 0


async def test_embeddings(bare_stack):
    s = bare_stack
    r = await s.api.post("/v1/embeddings", json={"input": ["alpha", "beta"]}, headers=llm_headers(s))
    d = r.json()["data"]
    assert len(d) == 2 and len(d[0]["embedding"]) == s.settings.embedding_dim
    assert abs(sum(x * x for x in d[0]["embedding"]) - 1) < 1e-6


async def test_admin_endpoints_require_admin_scope(bare_stack):
    s = bare_stack
    assert (await s.api.get("/v1/admin/killswitch", headers=s.auth())).status_code == 403
    assert (await s.api.get("/v1/admin/killswitch", headers=s.auth(scopes=["admin:killswitch"]))).status_code == 200


async def test_killswitch_blocks_llm_calls_at_gateway(bare_stack):
    s = bare_stack
    adm = s.auth("root", scopes=["admin:killswitch"])
    r = await s.api.put("/v1/admin/killswitch", json={"scope": "global", "reason": "drill"}, headers=adm)
    assert r.status_code == 200
    blocked = await s.api.post("/v1/chat/completions", json=BODY, headers=llm_headers(s))
    assert blocked.status_code == 503 and blocked.json()["error"]["code"] == "kill_switch_engaged"
    assert (await s.api.delete("/v1/admin/killswitch/global", headers=adm)).status_code == 200
    assert (await s.api.post("/v1/chat/completions", json=BODY, headers=llm_headers(s))).status_code == 200


async def test_health_ready_metrics(bare_stack):
    s = bare_stack
    assert (await s.api.get("/healthz")).json()["status"] == "ok"
    assert (await s.api.get("/readyz")).status_code == 200
    m = (await s.api.get("/metrics")).text
    assert "agentic_http_requests_total" in m


# ── model router ──────────────────────────────────────────────────────────────────────────


async def route(s, text):
    return await s.orchestrator_core.router.complete(
        agent="supervisor", system="ROLE:supervisor\nroute", user=text, tenant="default", user_id="alice", schema=Route)


async def test_router_small_first(bare_stack):
    res, parsed = await route(bare_stack, "plain")
    assert res.tier == "small" and not res.escalated and not res.fallback_used and parsed.next == "research"


async def test_router_escalates_on_invalid_output(bare_stack):
    res, parsed = await route(bare_stack, "[[force-escalate]]")
    assert res.tier == "large" and res.escalated and parsed is not None


async def test_router_escalates_on_low_confidence(bare_stack):
    res, parsed = await route(bare_stack, "[[low-confidence]]")
    assert res.tier == "large" and res.escalated and parsed.confidence >= 0.6


async def test_router_small_error_goes_to_large(bare_stack):
    res, _ = await route(bare_stack, "[[fail-small]]")
    assert res.tier == "large" and not res.fallback_used


async def test_router_uses_fallback_model_when_primary_models_fail(bare_stack):
    res, _ = await route(bare_stack, "[[fail-small]] [[fail-large]]")
    assert res.tier == "fallback" and res.fallback_used


async def test_router_raises_when_everything_fails(bare_stack):
    with pytest.raises(UpstreamError):
        await route(bare_stack, "[[fail-small]] [[fail-large]] [[fail-fallback]]")


async def test_router_never_retries_policy_errors(bare_stack):
    s = bare_stack
    await s.c.killswitch.engage("global", "drill", "t")
    with pytest.raises(KillSwitchEngaged):
        await route(s, "anything")


async def test_router_circuit_breaker_skips_dead_model(bare_stack):
    r = bare_stack.orchestrator_core.router
    for _ in range(5):
        await route(bare_stack, "[[fail-small]]")
    assert r.breaker.state("small") == "open"
    res, _ = await route(bare_stack, "ok now")  # small is skipped without a network call
    assert res.tier == "large"
