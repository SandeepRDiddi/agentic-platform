"""Single source of truth for tool policy: scope, write-ness, argument contract, approval requirement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    scope: str  # scope the CALLER (agent identity) must hold
    write: bool
    description: str
    params: str  # human/LLM readable signature
    required: tuple[str, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)
    coerce: dict[str, type] = field(default_factory=dict)

    def normalize(self, args: dict[str, Any]) -> dict[str, Any]:
        """Canonical argument dict. The approval hash is computed over THIS form, so what a human
        approves is byte-for-byte what the tool server later executes."""
        missing = [k for k in self.required if args.get(k) in (None, "")]
        if missing:
            raise ValueError(f"missing required args for {self.name}: {', '.join(missing)}")
        out = {k: args[k] for k in self.required}
        for k, d in self.defaults.items():
            out[k] = args.get(k, d)
        for k, t in self.coerce.items():
            if k in out:
                out[k] = t(out[k])
        return out


CATALOG: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec("search_tickets", "tools:read", False, "Search tickets by text and/or status",
                 "query: str = '', status: str = ''", defaults={"query": "", "status": ""}),
        ToolSpec("get_ticket", "tools:read", False, "Fetch one ticket", "ticket_id: int",
                 required=("ticket_id",), coerce={"ticket_id": int}),
        ToolSpec("search_knowledge", "tools:read", False, "Hybrid RAG+GraphRAG search over the knowledge base",
                 "query: str, k: int = 4", required=("query",), defaults={"k": 4}, coerce={"k": int}),
        ToolSpec("create_ticket", "tools:write", True, "Create a ticket (REQUIRES HUMAN APPROVAL)",
                 "title: str, description: str = '', priority: low|medium|high|critical = medium",
                 required=("title",), defaults={"description": "", "priority": "medium"}),
        ToolSpec("update_ticket_status", "tools:write", True, "Change ticket status (REQUIRES HUMAN APPROVAL)",
                 "ticket_id: int, status: open|in_progress|resolved|closed",
                 required=("ticket_id", "status"), coerce={"ticket_id": int}),
        ToolSpec("send_notification", "tools:write", True, "Post a notification (REQUIRES HUMAN APPROVAL)",
                 "channel: str, message: str", required=("channel", "message")),
    ]
}


def describe() -> str:
    return "\n".join(f"- {t.name}({t.params}): {t.description}" for t in CATALOG.values())
