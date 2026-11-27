"""Guardrails (trust layer: "input + output").

Deterministic, fast, no-network rules that run on every request, plus an optional LLM judge
hook for borderline cases. Rules:

  input   : size cap, prompt-injection patterns, secret/PII redaction (or block)
  context : indirect prompt injection in retrieved documents / tool results is neutralised
  tool    : arguments must not contain secrets or injection payloads
  output  : secrets always redacted, PII redacted, size cap, citation-required check
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agentic.common.metrics import GUARDRAIL_BLOCKS

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|rules|messages)",
    r"disregard (all |any |the )?(previous|prior|above|system) ",
    r"forget (everything|all) (you|that) (know|were told)",
    r"(reveal|show|print|repeat|leak) (me )?(your|the) (system|hidden|initial) (prompt|instructions|message)",
    r"you are now (dan|in developer mode|an? unrestricted)",
    r"</?\s*(system|assistant)\s*>",
    r"\bjailbreak\b",
    r"pretend (that )?you (have|are) no (rules|restrictions|guidelines)",
    r"override (your )?(safety|security|guardrails?)",
]
_INJ = [re.compile(p, re.I) for p in INJECTION_PATTERNS]

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_key": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    "azure_key": re.compile(r"(AccountKey|SharedAccessKey)=[A-Za-z0-9+/=]{20,}"),
}
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "pan_in": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "aadhaar": re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,5}[\s-]\d{3,5}(?:[\s-]\d{2,5})?(?!\d)"),
}
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _luhn(num: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", num)]
    if not 13 <= len(digits) <= 19:
        return False
    s = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            d -= 9 if d > 9 else 0
        s += d
    return s % 10 == 0


@dataclass
class GuardrailResult:
    allowed: bool
    text: str
    violations: list[str] = field(default_factory=list)
    redactions: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return ", ".join(self.violations) or "ok"


LLMJudge = Callable[[str, str], Awaitable[bool]]  # (stage, text) -> safe?


class Guardrails:
    def __init__(
        self, *, pii_mode: str = "redact", max_input_chars: int = 8000, max_output_chars: int = 20000,
        llm_judge: LLMJudge | None = None,
    ) -> None:
        self.pii_mode = pii_mode
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars
        self.llm_judge = llm_judge

    # ── helpers ───────────────────────────────────────────────────────────────────────────
    def _redact(self, text: str, *, secrets: bool, pii: bool) -> tuple[str, list[str]]:
        found: list[str] = []
        if secrets:
            for name, pat in SECRET_PATTERNS.items():
                if pat.search(text):
                    found.append(name)
                    text = pat.sub(f"[REDACTED:{name}]", text)
        if pii:
            def card(m: re.Match[str]) -> str:
                if _luhn(m.group(0)):
                    found.append("credit_card")
                    return "[REDACTED:credit_card]"
                return m.group(0)

            text = _CARD.sub(card, text)
            for name, pat in PII_PATTERNS.items():
                if pat.search(text):
                    found.append(name)
                    text = pat.sub(f"[REDACTED:{name}]", text)
        return text, found

    def redact(self, text: str) -> tuple[str, list[str]]:
        """Redact secrets + PII regardless of pii_mode. Returns (clean_text, found_names)."""
        return self._redact(text, secrets=True, pii=True)

    def scan(self, text: str) -> list[str]:
        """Names of sensitive items (secrets + PII) present in text, without modifying it."""
        return self._redact(text, secrets=True, pii=True)[1]

    @staticmethod
    def _injection(text: str) -> list[str]:
        return [p.pattern for p in _INJ if p.search(text)]

    # ── stages ────────────────────────────────────────────────────────────────────────────
    async def check_input(self, text: str) -> GuardrailResult:
        violations: list[str] = []
        if len(text) > self.max_input_chars:
            GUARDRAIL_BLOCKS.labels("input", "too_long", "block").inc()
            return GuardrailResult(False, text[: self.max_input_chars], ["input_too_long"])
        if self._injection(text):
            GUARDRAIL_BLOCKS.labels("input", "prompt_injection", "block").inc()
            return GuardrailResult(False, text, ["prompt_injection"])
        redact_pii = self.pii_mode != "allow"
        clean, found = self._redact(text, secrets=True, pii=redact_pii)
        if found and self.pii_mode == "block":
            GUARDRAIL_BLOCKS.labels("input", "sensitive_data", "block").inc()
            return GuardrailResult(False, text, [f"sensitive:{f}" for f in found])
        for f in found:
            GUARDRAIL_BLOCKS.labels("input", f, "redact").inc()
        if self.llm_judge and not await self.llm_judge("input", clean):
            GUARDRAIL_BLOCKS.labels("input", "llm_judge", "block").inc()
            return GuardrailResult(False, clean, ["llm_judge_unsafe"], found)
        return GuardrailResult(True, clean, violations, found)

    def sanitize_context(self, text: str, source: str = "retrieved") -> tuple[str, bool]:
        """Neutralise indirect prompt injection inside retrieved docs / tool outputs.

        Instruction-like lines are dropped; the remainder is wrapped as inert data."""
        flagged = False
        kept = []
        for line in text.splitlines():
            if self._injection(line):
                flagged = True
                GUARDRAIL_BLOCKS.labels("context", "indirect_injection", "strip").inc()
                continue
            kept.append(line)
        clean, _ = self._redact("\n".join(kept), secrets=True, pii=False)
        return clean, flagged

    def check_tool_call(self, tool: str, args: dict[str, Any]) -> GuardrailResult:
        blob = " ".join(str(v) for v in args.values())
        if self._injection(blob):
            GUARDRAIL_BLOCKS.labels("tool", "prompt_injection", "block").inc()
            return GuardrailResult(False, blob, ["tool_args_injection"])
        for name, pat in SECRET_PATTERNS.items():
            if pat.search(blob):
                GUARDRAIL_BLOCKS.labels("tool", name, "block").inc()
                return GuardrailResult(False, blob, [f"tool_args_secret:{name}"])
        return GuardrailResult(True, blob)

    async def check_output(self, text: str, *, require_citations: bool = False, n_citations: int = 0) -> GuardrailResult:
        violations: list[str] = []
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars] + "\n[truncated]"
            violations.append("output_truncated")
        clean, found = self._redact(text, secrets=True, pii=self.pii_mode != "allow")
        for f in found:
            GUARDRAIL_BLOCKS.labels("output", f, "redact").inc()
        if require_citations and n_citations == 0:
            violations.append("ungrounded_answer")
            GUARDRAIL_BLOCKS.labels("output", "ungrounded", "flag").inc()
        if self.llm_judge and not await self.llm_judge("output", clean):
            GUARDRAIL_BLOCKS.labels("output", "llm_judge", "block").inc()
            return GuardrailResult(False, "", ["llm_judge_unsafe"], found)
        return GuardrailResult(True, clean, violations, found)
