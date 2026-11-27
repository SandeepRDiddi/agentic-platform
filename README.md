# Agentic Platform — the 9-layer reference architecture, implemented

A runnable, tested monorepo implementing every box of the "Agentic System Design" diagram.
The model is one box; this repo is the other eight and the wiring between them.

```
Channels ─► AI Gateway ─► Orchestration ─► Models (small → large → fallback)
 Web/Slack   auth, rate     LangGraph:       via gateway
 /Teams      limit, budget, supervisor ⇄
             cache, routing research/action/review ─► Tools (MCP) ─► Internal API
                                │                      Knowledge (hybrid RAG + GraphRAG + memory)
                                └─► Async backbone (Kafka bus, workers, retries, DLQ)
 Trust layer across all: identity, guardrails, approvals, audit chain, kill switch, evals, fallbacks, tracing
 CI/CD: tests + eval gate → image → Argo CD → Argo Rollouts canary gated on metrics + evals
```

## Where each box lives

| Layer | Code | Notes |
|---|---|---|
| 1 Channels | `channels/` | Web UI (streaming), Slack (signed requests), Teams (Bot Framework JWT), Kafka result dispatcher |
| 2 AI Gateway | `gateway/` | OIDC/JWKS or API-key auth, per-tenant rate limit and daily budget, exact-match cache, alias routing with provider failover (LiteLLM) |
| 3 Orchestration | `orchestration/` | LangGraph: input guard → memory → supervisor ⇄ research / action / review → finalize. Explicit state machine; action agent capped at 3 tool calls |
| 4 Models | `models/router.py` | small → large → fallback; escalates on invalid output, low confidence, truncation or error; circuit breaker |
| 5 Tools (MCP) | `tools/` | Stateless streamable-HTTP MCP server, identity middleware, per-agent scopes, internal API |
| 6 Knowledge | `knowledge/` | pgvector + FTS fused with RRF, GraphRAG (entity expansion + community summaries), PII-free TTL long-term memory, Redis short-term history |
| 7 Async backbone | `backbone/` | Kafka (or in-memory) bus, at-least-once, commit after processing, idempotent claim, backoff retries, DLQ |
| 8 Trust layer | `trust/` | Per-agent identity JWTs, guardrails (PII, injection), approvals bound to an args hash (single-use, four-eyes), hash-chained audit log (DB triggers block UPDATE/DELETE), kill switch (global/tenant/agent/task, fail-closed), eval gate, OpenTelemetry → Langfuse |
| 9 CI/CD | `.github/workflows/`, `azure-pipelines.yml`, `deploy/` | CI (ruff, pytest with Postgres/Redis/Kafka, eval gate, helm lint, kubeconform, Trivy), CD (OIDC to ACR, cosign, GitOps tag bump), manual prod promotion, Argo CD apps, Helm chart with Argo Rollouts + AnalysisTemplate, KEDA scaling on Kafka lag |

## Quickstart (offline, no API keys)

```bash
cp .env.example .env              # LLM_MODE=mock by default
docker compose up --build         # UI http://localhost:8000, gateway :8001, Jaeger :16686
```

Without Docker:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q                         # 104 tests; integration tests auto-skip without TEST_* URLs
agentic-evals --target inproc     # eval gate
```

With no `DATABASE_URL`/`REDIS_URL`/`KAFKA_BOOTSTRAP`, the app uses in-memory backends. Set them and the same
code uses Postgres+pgvector, Redis and Kafka. For live models set `LLM_MODE=live` and provider keys.

## Production configuration

- `Settings.validate_for_prod()` refuses to start with dev secrets, in-memory backends or `LLM_MODE=mock`.
- Secrets come from one Kubernetes Secret (`secretName`), optionally built by External Secrets from Key Vault.
- `agentic-migrate` runs as a PreSync Job. `agentic-killswitch` is the operator CLI for the kill switch.
- Edge services (channels, gateway) get true 5% traffic canaries through NGINX. Internal services
  (orchestrator) are canaried by replica weight unless you add a service mesh. Canary gates:
  success rate, p95 latency, run-failure ratio, plus an eval Job against the canary using a dedicated `eval` tenant.

## Honest limits — read before relying on this

Verified here: unit and component tests, in-process and real-socket end-to-end (gateway → orchestrator → MCP → internal API),
Postgres+pgvector and Redis against real servers, YAML syntax of all manifests.

**Not verified in the build environment:**
- `helm lint` / `helm template`, Argo Rollouts, KEDA and Strimzi manifests were written but never rendered or applied. Expect to fix small template issues on first `helm lint`.
- Docker image build and `docker compose up` were not run.
- Kafka paths were never run against a broker (`tests/test_integration_kafka.py` exists; CI runs it).
- No live LLM provider, Slack or Teams call was made. Those paths are tested against fakes.
- The canary Prometheus queries assume the ServiceMonitor copies `rollouts-pod-template-hash`; check series labels in your Prometheus.
- No Terraform/infrastructure is included (cluster, ACR, Key Vault, Event Hubs, Entra app registrations).
- The in-process eval gate uses the deterministic mock LLM, so it checks pipeline correctness (routing, guardrails,
  approvals, citations), not answer quality. Real quality gating needs live models and `--ragas` with a judge key.
- Replace `YOUR_ORG`, `myregistry.azurecr.io` and hostnames before use; configure the `production` environment reviewers in GitHub.
- No load, security or penetration testing has been done.
