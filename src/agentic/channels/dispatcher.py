"""Delivers async results back to the originating channel (Slack thread / Teams conversation)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from agentic.channels import slack, teams
from agentic.container import Container

log = logging.getLogger(__name__)


class ResultDispatcher:
    def __init__(self, c: Container, ext: httpx.AsyncClient, bot_tokens: teams.BotTokenProvider) -> None:
        self.c, self.ext, self.bot_tokens = c, ext, bot_tokens
        self._stop = asyncio.Event()
        self._consumer: Any = None

    async def send_slack(self, reply_to: dict[str, Any], text: str, approvals: list[dict[str, Any]]) -> None:
        tok = self.c.settings.slack_bot_token
        if not tok:
            log.warning("slack bot token not configured; dropping reply")
            return
        r = await self.ext.post("https://slack.com/api/chat.postMessage", headers={"authorization": f"Bearer {tok}"},
                                json={"channel": reply_to["slack_channel"], "thread_ts": reply_to.get("thread_ts"),
                                      "text": text[:3000], "blocks": slack.blocks_for(text, approvals)})
        if r.status_code != 200 or not r.json().get("ok"):
            log.error("slack post failed: %s %s", r.status_code, r.text[:200])

    async def send_teams(self, reply_to: dict[str, Any], text: str, approvals: list[dict[str, Any]]) -> None:
        url = reply_to.get("service_url", "")
        if not teams.trusted_service_url(url):
            log.error("refusing to reply to untrusted serviceUrl %r", url)
            return
        token = await self.bot_tokens.get(self.ext)
        activity: dict[str, Any] = {"type": "message", "text": text, "textFormat": "markdown"}
        cards = [teams.approval_card(a) for a in approvals if a.get("status") == "pending"]
        if cards:
            activity["attachments"] = cards
        path = f"/v3/conversations/{reply_to['conversation_id']}/activities"
        if reply_to.get("activity_id"):
            path += f"/{reply_to['activity_id']}"
        r = await self.ext.post(url.rstrip("/") + path, headers={"authorization": f"Bearer {token}"}, json=activity)
        if r.status_code >= 300:
            log.error("teams post failed: %s %s", r.status_code, r.text[:200])

    async def handle(self, value: dict[str, Any]) -> None:
        reply_to = value.get("reply_to") or {}
        text = value.get("answer") or f"Task {value['status']}."
        if reply_to.get("channel") == "slack":
            await self.send_slack(reply_to, text, value.get("pending_approvals", []))
        elif reply_to.get("channel") == "teams":
            await self.send_teams(reply_to, text, value.get("pending_approvals", []))

    async def run(self) -> None:
        s = self.c.settings
        self._consumer = await self.c.bus.consumer([s.topic_results], s.dispatcher_group)
        async for msg in self._consumer:
            if self._stop.is_set():
                break
            try:
                await self.handle(msg.value)
            except Exception:
                log.exception("result dispatch failed for %s", msg.key)  # best effort; the task result is durable
            await self._consumer.commit(msg)

    def stop(self) -> None:
        self._stop.set()
