"""Kill switch (trust layer: "stop any agent").

Flags live in Redis so every replica of every service sees a change within one request.
Scopes (most to least specific): task:<id>, agent:<name>, tenant:<id>, global.

Checked at: gateway ingress, every LLM call, before every agent node, before every tool call
(in both the orchestrator and the MCP server) and by the worker before it picks up a task.
By default it fails CLOSED: if the flag store is unreachable we stop rather than run blind.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

from agentic.common.errors import KillSwitchEngaged
from agentic.common.kv import KV
from agentic.common.metrics import KILLSWITCH_BLOCKS

log = logging.getLogger(__name__)
PREFIX = "killswitch:"


class KillSwitch:
    def __init__(self, kv: KV, fail_closed: bool = True) -> None:
        self.kv = kv
        self.fail_closed = fail_closed

    async def engage(self, scope: str, reason: str, actor: str) -> dict:
        self._validate(scope)
        rec = {"scope": scope, "reason": reason, "actor": actor, "ts": time.time()}
        await self.kv.set(PREFIX + scope, json.dumps(rec))
        log.warning("kill switch ENGAGED %s by %s: %s", scope, actor, reason)
        return rec

    async def release(self, scope: str) -> None:
        self._validate(scope)
        await self.kv.delete(PREFIX + scope)
        log.warning("kill switch released %s", scope)

    async def status(self) -> list[dict]:
        out = []
        for k in await self.kv.keys(PREFIX):
            raw = await self.kv.get(k)
            if raw:
                out.append(json.loads(raw))
        return sorted(out, key=lambda r: r["scope"])

    async def check(
        self, *, agent: str | None = None, tenant: str | None = None, task_id: str | None = None
    ) -> None:
        scopes = ["global"]
        if tenant:
            scopes.append(f"tenant:{tenant}")
        if agent:
            scopes.append(f"agent:{agent}")
        if task_id:
            scopes.append(f"task:{task_id}")
        try:
            for s in scopes:
                raw = await self.kv.get(PREFIX + s)
                if raw:
                    rec = json.loads(raw)
                    KILLSWITCH_BLOCKS.labels(s.split(":")[0]).inc()
                    raise KillSwitchEngaged(
                        f"kill switch engaged for {s}: {rec.get('reason', '')}", scope=s
                    )
        except KillSwitchEngaged:
            raise
        except Exception as e:  # flag store down
            if self.fail_closed:
                KILLSWITCH_BLOCKS.labels("store_unavailable").inc()
                raise KillSwitchEngaged("kill switch store unavailable (fail-closed)", scope="store") from e
            log.error("kill switch store unavailable, failing open: %s", e)

    @staticmethod
    def _validate(scope: str) -> None:
        if scope != "global" and scope.split(":")[0] not in {"tenant", "agent", "task"}:
            raise ValueError("scope must be global | tenant:<id> | agent:<name> | task:<id>")


def main() -> None:
    """Break-glass CLI usable from any pod/laptop with REDIS_URL: agentic-killswitch engage global -r why"""
    from agentic.common.kv import build_kv
    from agentic.config import get_settings

    p = argparse.ArgumentParser(prog="agentic-killswitch")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("engage")
    e.add_argument("scope")
    e.add_argument("-r", "--reason", required=True)
    e.add_argument("--actor", default="cli")
    r = sub.add_parser("release")
    r.add_argument("scope")
    sub.add_parser("status")
    a = p.parse_args()

    async def run() -> None:
        ks = KillSwitch(build_kv(get_settings()))
        if a.cmd == "engage":
            print(json.dumps(await ks.engage(a.scope, a.reason, a.actor)))
        elif a.cmd == "release":
            await ks.release(a.scope)
            print("released", a.scope)
        else:
            print(json.dumps(await ks.status(), indent=2))
        await ks.kv.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
