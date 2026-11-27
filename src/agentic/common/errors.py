"""Domain errors. HTTP mapping lives in common/http.py."""

from __future__ import annotations


class AgenticError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str = "", **details):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = details


class BadRequest(AgenticError):
    status_code = 400
    code = "bad_request"


class AuthError(AgenticError):
    status_code = 401
    code = "unauthenticated"


class ForbiddenError(AgenticError):
    status_code = 403
    code = "forbidden"


class RateLimited(AgenticError):
    status_code = 429
    code = "rate_limited"


class BudgetExceeded(AgenticError):
    status_code = 402
    code = "budget_exceeded"


class KillSwitchEngaged(AgenticError):
    status_code = 503
    code = "kill_switch_engaged"


class GuardrailViolation(AgenticError):
    status_code = 422
    code = "guardrail_violation"


class NotFound(AgenticError):
    status_code = 404
    code = "not_found"


class Conflict(AgenticError):
    status_code = 409
    code = "conflict"


class UpstreamError(AgenticError):
    status_code = 502
    code = "upstream_error"


class CircuitOpen(UpstreamError):
    code = "circuit_open"
