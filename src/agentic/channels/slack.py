"""Slack channel helpers: request signature verification and Block Kit rendering."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

from agentic.common.errors import AuthError


def verify_signature(signing_secret: str | None, body: bytes, timestamp: str | None, signature: str | None,
                     tolerance: int = 300, now: float | None = None) -> None:
    if not signing_secret:
        raise AuthError("slack signing secret not configured")
    if not timestamp or not signature:
        raise AuthError("missing slack signature headers")
    try:
        ts = int(timestamp)
    except ValueError as e:
        raise AuthError("bad slack timestamp") from e
    if abs((now or time.time()) - ts) > tolerance:
        raise AuthError("stale slack request")
    base = f"v0:{timestamp}:".encode() + body
    expected = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise AuthError("invalid slack signature")


def strip_mention(text: str) -> str:
    import re

    return re.sub(r"<@[A-Z0-9]+>", "", text).strip()


def blocks_for(answer: str, approvals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [{"type": "section", "text": {"type": "mrkdwn", "text": answer[:2900] or "(no answer)"}}]
    for a in approvals:
        if a.get("status") != "pending":
            continue
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Approval needed:* `{a['tool']}`\n```{a['args']}```"}})
        blocks.append({"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "Approve"}, "style": "primary",
             "action_id": f"approve:{a['id']}", "value": a["id"]},
            {"type": "button", "text": {"type": "plain_text", "text": "Deny"}, "style": "danger",
             "action_id": f"deny:{a['id']}", "value": a["id"]}]})
    return blocks
