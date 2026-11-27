"""Prometheus metrics. Canary analysis in Argo Rollouts reads these."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter("agentic_http_requests_total", "HTTP requests", ["service", "route", "status"])
HTTP_LATENCY = Histogram(
    "agentic_http_request_seconds",
    "HTTP latency",
    ["service", "route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
LLM_REQUESTS = Counter("agentic_llm_requests_total", "LLM calls", ["alias", "model", "outcome"])
LLM_TOKENS = Counter("agentic_llm_tokens_total", "LLM tokens", ["model", "kind"])
LLM_COST = Counter("agentic_llm_cost_usd_total", "LLM spend (USD)", ["tenant"])
LLM_CACHE_HITS = Counter("agentic_llm_cache_hits_total", "Gateway cache hits")
LLM_LATENCY = Histogram(
    "agentic_llm_seconds", "LLM latency", ["alias"], buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60)
)
ROUTER_ESCALATIONS = Counter("agentic_router_escalations_total", "small->large escalations", ["reason"])
ROUTER_FALLBACKS = Counter("agentic_router_fallbacks_total", "fallback model uses", ["from_tier"])
GUARDRAIL_BLOCKS = Counter("agentic_guardrail_blocks_total", "guardrail outcomes", ["stage", "rule", "action"])
KILLSWITCH_BLOCKS = Counter("agentic_killswitch_blocks_total", "kill switch denials", ["scope"])
AGENT_RUNS = Counter("agentic_agent_runs_total", "agent runs", ["status"])
AGENT_STEP_FAILURES = Counter("agentic_agent_step_failures_total", "agent node failures", ["agent"])
AGENT_RUN_SECONDS = Histogram(
    "agentic_agent_run_seconds", "end-to-end run latency", buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300)
)
TOOL_CALLS = Counter("agentic_tool_calls_total", "tool calls", ["tool", "outcome"])
APPROVALS = Counter("agentic_approvals_total", "approval decisions", ["status"])
RATE_LIMITED = Counter("agentic_rate_limited_total", "429s", ["tenant"])
TASKS_IN_FLIGHT = Gauge("agentic_worker_tasks_in_flight", "tasks being processed")
WORKER_RESULTS = Counter("agentic_worker_results_total", "worker outcomes", ["outcome"])
EVAL_SCORE = Gauge("agentic_eval_score", "last online eval score", ["metric"])
