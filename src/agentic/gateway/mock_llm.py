"""Deterministic offline LLM used when LLM_MODE=mock.

Lets the whole platform (and CI) run with zero API keys and no network. Behaviour is selected
by a `ROLE:<name>` tag at the top of the system prompt. Special tags in the user text drive
failure-path tests:  [[force-escalate]]  -> small model returns unparseable output
                     [[low-confidence]]  -> small model returns confidence 0.2
                     [[fail-small]] / [[fail-large]] -> that tier raises (exercises fallbacks)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from agentic.common.errors import UpstreamError


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def hash_embedding(text: str, dim: int = 1024) -> list[float]:
    """Signed feature hashing of unigrams+bigrams. Deterministic, lexical-semantic enough for dev."""
    toks = _tokens(text)
    grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
    v = [0.0] * dim
    for g in grams:
        h = hashlib.blake2b(g.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        v[idx] += sign
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _role(messages: list[dict[str, Any]]) -> str:
    sys = next((m["content"] for m in messages if m["role"] == "system"), "")
    m = re.match(r"\s*ROLE:(\w+)", sys)
    return m.group(1) if m else "chat"


def _last_user(messages: list[dict[str, Any]]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


def _json_between(text: str, tag: str) -> Any:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


ACTION_WORDS = re.compile(r"\b(create|open|file|raise|log|update|close|send|notify)\b.*\b(ticket|issue|notification|alert)\b", re.I)


def complete(alias: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    user = _last_user(messages)
    role = _role(messages)
    if f"[[fail-{alias}]]" in user:
        raise UpstreamError(f"mock provider failure for {alias}")
    if alias == "small" and "[[force-escalate]]" in user:
        return _wrap("I think the answer is probably something, not sure {{", alias)
    low_conf = alias == "small" and "[[low-confidence]]" in user
    conf = 0.2 if low_conf else 0.9

    if role == "supervisor":
        st = _json_between(user, "state") or {}
        if st.get("steps", 0) >= st.get("max_steps", 8):
            nxt, why = "finish", "step budget exhausted"
        elif not st.get("has_findings"):
            nxt, why = "research", "need evidence first"
        elif st.get("needs_action") and not st.get("actions_done"):
            nxt, why = "action", "request asks for a side effect"
        elif not st.get("reviewed"):
            nxt, why = "review", "draft needs review before release"
        else:
            nxt, why = "finish", "reviewed and complete"
        return _wrap(json.dumps({"next": nxt, "reason": why, "confidence": conf}), alias)

    if role == "research":
        ctx = _json_between(user, "context") or []
        q = (_json_between(user, "question") or "")
        if not ctx:
            return _wrap(json.dumps({"answer": "I could not find supporting evidence in the knowledge base.", "citations": [], "confidence": conf}), alias)
        qt = set(_tokens(q)) - {"the", "a", "an", "of", "to", "is", "are", "what", "how", "when", "who", "does", "do", "and", "in", "for", "must", "be"}
        sents = []
        for c in ctx:
            for sent in re.split(r"(?<=[.!?])\s+", c["text"].replace("\n", " ")):
                if sent.strip():
                    sents.append((len(qt & set(_tokens(sent))), sent.strip(), c["id"]))
        best = sorted(sents, key=lambda x: -x[0])[:2]
        best = [b for b in best if b[0] > 0] or sents[:1]
        text = " ".join(b[1] for b in best)[:600]
        ids = list(dict.fromkeys(b[2] for b in best))
        cites = "".join(f" [{i}]" for i in ids)
        return _wrap(json.dumps({"answer": f"{text}{cites}", "citations": ids, "confidence": conf}), alias)

    if role == "action":
        q = _json_between(user, "request") or ""
        calls: list[dict[str, Any]] = []
        if ACTION_WORDS.search(q):
            if re.search(r"\b(send|notify)\b", q, re.I):
                calls.append({"tool": "send_notification", "args": {"channel": "ops", "message": q[:200]}})
            else:
                calls.append({"tool": "create_ticket", "args": {"title": q[:80], "description": q, "priority": "medium"}})
        else:
            calls.append({"tool": "search_tickets", "args": {"query": q[:80]}})
        return _wrap(json.dumps({"tool_calls": calls, "confidence": conf}), alias)

    if role == "review":
        d = _json_between(user, "draft") or {}
        issues = []
        if not d.get("answer"):
            issues.append("empty draft")
        if d.get("has_evidence") and not d.get("has_citations"):
            issues.append("claims lack citations")
        return _wrap(json.dumps({"approved": not issues, "issues": issues, "confidence": conf}), alias)

    if role == "guard":
        return _wrap(json.dumps({"safe": "[[unsafe]]" not in user}), alias)

    if role == "memory":
        q = _json_between(user, "turn") or {}
        facts = []
        m = re.search(r"\bmy (?:team|project|stack|preferred \w+) (?:is|are) ([^.]{3,60})", q.get("user", ""), re.I)
        if m:
            facts.append(m.group(0).strip())
        return _wrap(json.dumps({"facts": facts}), alias)

    if role == "graph_extract":
        text = _json_between(user, "text") or ""
        ents = list(dict.fromkeys(re.findall(r"\b([A-Z][A-Za-z0-9]+(?:\s[A-Z][A-Za-z0-9]+)*)\b", text)))
        ents = [e for e in ents if len(e) > 2 and e not in {"The", "This", "That", "When", "Each", "All"}][:12]
        rels = [{"src": a, "dst": b, "relation": "related_to"} for a, b in zip(ents, ents[1:], strict=False)]
        return _wrap(json.dumps({"entities": [{"name": e, "type": "entity"} for e in ents], "relations": rels}), alias)

    if role == "summarize":
        return _wrap("Community of related entities: " + ", ".join((_json_between(user, "members") or [])[:8]), alias)

    return _wrap(f"[mock:{alias}] " + user[:200], alias)


def _wrap(content: str, alias: str) -> dict[str, Any]:
    pt = max(1, len(content) // 4)
    return {
        "content": content, "model": f"mock/{alias}", "finish_reason": "stop",
        "prompt_tokens": 50, "completion_tokens": pt, "cost_usd": (50 + pt) * 1e-7,
    }
