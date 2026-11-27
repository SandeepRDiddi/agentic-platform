"""Central configuration. Every knob is an environment variable (12-factor)."""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_JWT_SECRET = "dev-secret-change-me-dev-secret-change-me"
DEFAULT_IDENTITY_SECRET = "dev-identity-secret-change-me-dev-identity"


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- runtime -------------------------------------------------------------------------
    env: str = "dev"
    service_name: str = "agentic"
    service_version: str = "1.0.0"
    log_level: str = "INFO"

    # --- infrastructure (None => in-memory backend, dev/test only) -----------------------
    database_url: str | None = None
    redis_url: str | None = None
    kafka_bootstrap: str | None = None
    kafka_security_protocol: str = "PLAINTEXT"
    kafka_sasl_mechanism: str | None = None
    kafka_sasl_username: str | None = None
    kafka_sasl_password: str | None = None
    topic_tasks: str = "agent.tasks"
    topic_results: str = "agent.results"
    topic_dlq: str = "agent.tasks.dlq"
    worker_group: str = "agentic-workers"
    dispatcher_group: str = "agentic-dispatchers"
    worker_max_attempts: int = 3
    worker_concurrency: int = 4
    task_timeout_seconds: int = 600

    # --- service URLs --------------------------------------------------------------------
    gateway_url: str = "http://localhost:8001"
    orchestrator_url: str = "http://localhost:8002"
    mcp_url: str = "http://localhost:8003/mcp"
    internal_api_url: str = "http://localhost:8004"

    # --- authn / authz -------------------------------------------------------------------
    jwt_secret: str = DEFAULT_JWT_SECRET
    jwt_algorithms: str = "HS256"  # prod: RS256 with jwks_url (Entra ID / Okta)
    jwt_issuer: str | None = None
    jwt_audience: str | None = None
    jwks_url: str | None = None
    api_keys_json: str = "{}"  # {"<sha256(key)>": {"sub":"svc","tenant":"acme","scopes":["agent:run"]}}
    identity_secret: str = DEFAULT_IDENTITY_SECRET  # signs per-agent identity tokens
    identity_ttl_seconds: int = 300
    approval_four_eyes: bool = True  # requester may not approve their own action
    approver_users: str = ""  # CSV of subjects allowed to decide approvals ("*" = any approver scope)
    dev_token_endpoint: bool = False  # exposes /dev/token on channels. NEVER in prod.

    # --- LLM / gateway -------------------------------------------------------------------
    llm_mode: Literal["live", "mock"] = "mock"
    model_small: str = "anthropic/claude-haiku-5-5"
    model_large: str = "anthropic/claude-sonnet-5-5"
    model_fallback: str = "openai/gpt-4o-mini"
    model_guard: str = "anthropic/claude-haiku-5-5"
    model_embedding: str = "openai/text-embedding-3-large"
    embedding_dim: int = 1024
    llm_timeout_seconds: float = 60
    llm_num_retries: int = 2
    escalation_confidence: float = 0.6
    max_agent_steps: int = 8
    max_revisions: int = 1
    rate_limit_per_minute: int = 120
    tenant_daily_budget_usd: float = 50.0
    cache_ttl_seconds: int = 3600
    max_request_chars: int = 8000
    model_routes_json: str = "{}"  # {"small": ["azure/gpt-4o-mini", "anthropic/claude-haiku-5-5"]} ordered failover

    # --- trust ---------------------------------------------------------------------------
    killswitch_fail_closed: bool = True
    pii_mode: Literal["redact", "block", "allow"] = "redact"
    llm_guard_enabled: bool = False
    require_approval_tools: str = "create_ticket,update_ticket_status,send_notification"

    # --- observability -------------------------------------------------------------------
    otel_exporter_otlp_endpoint: str | None = None
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    metrics_enabled: bool = True

    # --- channels ------------------------------------------------------------------------
    slack_signing_secret: str | None = None
    slack_bot_token: str | None = None
    teams_app_id: str | None = None
    teams_app_password: str | None = None
    teams_tenant_id: str | None = None
    channel_tenant: str = "default"  # tenant that Slack/Teams workspaces map to
    cors_origins: str = ""

    # ---------------------------------------------------------------------------------------
    @property
    def is_prod(self) -> bool:
        return self.env.lower() in {"prod", "production"}

    @property
    def jwt_algorithm_list(self) -> list[str]:
        return _csv(self.jwt_algorithms)

    @property
    def approver_list(self) -> list[str]:
        return _csv(self.approver_users)

    @property
    def approval_tools(self) -> set[str]:
        return set(_csv(self.require_approval_tools))

    @property
    def cors_origin_list(self) -> list[str]:
        return _csv(self.cors_origins)

    def model_routes(self) -> dict[str, list[str]]:
        return json.loads(self.model_routes_json or "{}")

    def api_keys(self) -> dict[str, dict]:
        return json.loads(self.api_keys_json or "{}")

    def langfuse_auth_header(self) -> str | None:
        if self.langfuse_public_key and self.langfuse_secret_key:
            raw = f"{self.langfuse_public_key}:{self.langfuse_secret_key}".encode()
            return "Basic " + base64.b64encode(raw).decode()
        return None

    def validate_for_prod(self) -> None:
        """Refuse to boot a production process with development defaults."""
        if not self.is_prod:
            return
        problems: list[str] = []
        if self.jwt_secret == DEFAULT_JWT_SECRET and not self.jwks_url:
            problems.append("JWT_SECRET is the dev default and no JWKS_URL is set")
        if self.identity_secret == DEFAULT_IDENTITY_SECRET:
            problems.append("IDENTITY_SECRET is the dev default")
        if self.llm_mode != "live":
            problems.append("LLM_MODE must be 'live' in prod")
        for name in ("database_url", "redis_url", "kafka_bootstrap"):
            if not getattr(self, name):
                problems.append(f"{name.upper()} is required in prod (no in-memory backends)")
        if self.dev_token_endpoint:
            problems.append("DEV_TOKEN_ENDPOINT must be false in prod")
        if not self.killswitch_fail_closed:
            problems.append("KILLSWITCH_FAIL_CLOSED must be true in prod")
        if problems:
            raise RuntimeError("Unsafe production configuration: " + "; ".join(problems))


@lru_cache
def get_settings() -> Settings:
    return Settings()
