"""Vector/keyword store, graph store and long-term memory store.

Postgres (pgvector + FTS) in prod; exact in-memory equivalents for dev/tests.
Everything is tenant-scoped at the query level.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol

from agentic.common.db import Database, vec
from agentic.common.schemas import new_id, now


@dataclass
class Hit:
    id: str
    doc_id: str
    source: str
    title: str
    content: str
    score: float = 0.0


@dataclass
class Node:
    name: str
    type: str = "entity"
    description: str = ""

    @property
    def id(self) -> str:
        return self.name.strip().lower()


@dataclass
class Edge:
    src: str  # node id
    dst: str
    relation: str
    chunk_id: str | None = None


@dataclass
class Community:
    id: str
    level: int
    members: list[str]
    summary: str
    embedding: list[float] = field(default_factory=list)
    score: float = 0.0


def _tok(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def cosine(a: list[float], b: list[float]) -> float:
    d = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return d / (na * nb)


class VectorStore(Protocol):
    async def upsert_document(self, tenant: str, doc_id: str, source: str, title: str,
                              chunks: list[tuple[str, int, str, list[float]]]) -> None: ...
    async def vector_search(self, tenant: str, emb: list[float], k: int) -> list[Hit]: ...
    async def keyword_search(self, tenant: str, query: str, k: int) -> list[Hit]: ...
    async def get_chunks(self, tenant: str, ids: list[str]) -> list[Hit]: ...
    async def count(self, tenant: str) -> int: ...


class GraphStore(Protocol):
    async def upsert(self, tenant: str, nodes: list[Node], edges: list[Edge]) -> None: ...
    async def find_nodes(self, tenant: str, ids: list[str]) -> list[Node]: ...
    async def neighbors(self, tenant: str, ids: list[str]) -> tuple[list[Node], list[Edge]]: ...
    async def dump(self, tenant: str) -> tuple[list[Node], list[Edge]]: ...
    async def put_communities(self, tenant: str, cs: list[Community]) -> None: ...
    async def search_communities(self, tenant: str, emb: list[float], k: int) -> list[Community]: ...


class MemoryStore(Protocol):
    async def add(self, tenant: str, user: str, content: str, emb: list[float], ttl_days: int | None) -> str: ...
    async def search(self, tenant: str, user: str, emb: list[float], k: int) -> list[Hit]: ...
    async def delete_user(self, tenant: str, user: str) -> int: ...


# ── in-memory ─────────────────────────────────────────────────────────────────────────────
class MemVectorStore:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.chunks: dict[str, dict] = {}

    async def upsert_document(self, tenant, doc_id, source, title, chunks) -> None:
        for cid in [c for c, v in self.chunks.items() if v["doc_id"] == doc_id]:
            del self.chunks[cid]
        self.docs[doc_id] = {"tenant": tenant, "source": source, "title": title}
        for cid, ord_, content, emb in chunks:
            self.chunks[cid] = {"doc_id": doc_id, "tenant": tenant, "ord": ord_, "content": content, "emb": emb}

    def _hit(self, cid: str, score: float) -> Hit:
        c = self.chunks[cid]
        d = self.docs[c["doc_id"]]
        return Hit(cid, c["doc_id"], d["source"], d["title"], c["content"], score)

    async def vector_search(self, tenant, emb, k) -> list[Hit]:
        scored = [(cosine(emb, c["emb"]), cid) for cid, c in self.chunks.items() if c["tenant"] == tenant]
        return [self._hit(cid, s) for s, cid in sorted(scored, reverse=True)[:k]]

    async def keyword_search(self, tenant, query, k) -> list[Hit]:
        q = set(_tok(query))
        n = max(1, len(self.chunks))
        df: dict[str, int] = {}
        for c in self.chunks.values():
            for t in set(_tok(c["content"])):
                df[t] = df.get(t, 0) + 1
        out = []
        for cid, c in self.chunks.items():
            if c["tenant"] != tenant:
                continue
            toks = _tok(c["content"])
            s = sum((toks.count(t) / (len(toks) or 1)) * math.log(1 + n / df.get(t, 1)) for t in q)
            if s > 0:
                out.append((s, cid))
        return [self._hit(cid, s) for s, cid in sorted(out, reverse=True)[:k]]

    async def get_chunks(self, tenant, ids) -> list[Hit]:
        return [self._hit(i, 0.0) for i in ids if i in self.chunks and self.chunks[i]["tenant"] == tenant]

    async def count(self, tenant) -> int:
        return sum(1 for c in self.chunks.values() if c["tenant"] == tenant)


class MemGraphStore:
    def __init__(self) -> None:
        self.nodes: dict[tuple[str, str], Node] = {}
        self.edges: dict[tuple[str, str, str, str], Edge] = {}
        self.comms: dict[str, tuple[str, Community]] = {}

    async def upsert(self, tenant, nodes, edges) -> None:
        for n in nodes:
            self.nodes.setdefault((tenant, n.id), n)
        for e in edges:
            self.edges.setdefault((tenant, e.src, e.dst, e.relation), e)

    async def find_nodes(self, tenant, ids) -> list[Node]:
        return [self.nodes[(tenant, i)] for i in ids if (tenant, i) in self.nodes]

    async def neighbors(self, tenant, ids):
        s = set(ids)
        es = [e for (t, *_), e in self.edges.items() if t == tenant and (e.src in s or e.dst in s)]
        nids = {e.src for e in es} | {e.dst for e in es}
        return await self.find_nodes(tenant, sorted(nids)), es

    async def dump(self, tenant):
        return ([n for (t, _), n in self.nodes.items() if t == tenant],
                [e for (t, *_), e in self.edges.items() if t == tenant])

    async def put_communities(self, tenant, cs) -> None:
        for k in [k for k, (t, _) in self.comms.items() if t == tenant]:
            del self.comms[k]
        for c in cs:
            self.comms[c.id] = (tenant, c)

    async def search_communities(self, tenant, emb, k):
        scored = [(cosine(emb, c.embedding), c) for t, c in self.comms.values() if t == tenant]
        out = []
        for s, c in sorted(scored, key=lambda x: -x[0])[:k]:
            out.append(Community(c.id, c.level, c.members, c.summary, [], s))
        return out


class MemMemoryStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    async def add(self, tenant, user, content, emb, ttl_days) -> str:
        mid = new_id("mem_")
        exp = now().timestamp() + ttl_days * 86400 if ttl_days else None
        self.rows[mid] = {"tenant": tenant, "user": user, "content": content, "emb": emb, "exp": exp}
        return mid

    async def search(self, tenant, user, emb, k) -> list[Hit]:
        t = now().timestamp()
        sc = [(cosine(emb, r["emb"]), i, r) for i, r in self.rows.items()
              if r["tenant"] == tenant and r["user"] == user and (r["exp"] is None or r["exp"] > t)]
        return [Hit(i, "", "memory", "memory", r["content"], s) for s, i, r in sorted(sc, key=lambda x: -x[0])[:k]]

    async def delete_user(self, tenant, user) -> int:
        ids = [i for i, r in self.rows.items() if r["tenant"] == tenant and r["user"] == user]
        for i in ids:
            del self.rows[i]
        return len(ids)


# ── postgres ──────────────────────────────────────────────────────────────────────────────
class PgVectorStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def upsert_document(self, tenant, doc_id, source, title, chunks) -> None:
        async with self.db.transaction() as c:
            await c.execute(
                "INSERT INTO documents (id, tenant, source, title) VALUES ($1,$2,$3,$4)"
                " ON CONFLICT (id) DO UPDATE SET source=$3, title=$4", doc_id, tenant, source, title)
            await c.execute("DELETE FROM chunks WHERE doc_id=$1", doc_id)
            await c.executemany(
                "INSERT INTO chunks (id, doc_id, tenant, ord, content, embedding) VALUES ($1,$2,$3,$4,$5,$6::vector)",
                [(cid, doc_id, tenant, o, txt, vec(e)) for cid, o, txt, e in chunks])

    @staticmethod
    def _hits(rows) -> list[Hit]:
        return [Hit(r["id"], r["doc_id"], r["source"], r["title"] or "", r["content"], float(r["score"])) for r in rows]

    async def vector_search(self, tenant, emb, k) -> list[Hit]:
        return self._hits(await self.db.fetch(
            "SELECT c.id, c.doc_id, d.source, d.title, c.content, 1 - (c.embedding <=> $2::vector) AS score"
            " FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.tenant=$1"
            " ORDER BY c.embedding <=> $2::vector LIMIT $3", tenant, vec(emb), k))

    async def keyword_search(self, tenant, query, k) -> list[Hit]:
        return self._hits(await self.db.fetch(
            "SELECT c.id, c.doc_id, d.source, d.title, c.content, ts_rank(c.tsv, q) AS score"
            " FROM chunks c JOIN documents d ON d.id=c.doc_id, websearch_to_tsquery('english', $2) q"
            " WHERE c.tenant=$1 AND c.tsv @@ q ORDER BY score DESC LIMIT $3", tenant, query, k))

    async def get_chunks(self, tenant, ids) -> list[Hit]:
        return self._hits(await self.db.fetch(
            "SELECT c.id, c.doc_id, d.source, d.title, c.content, 0.0 AS score FROM chunks c"
            " JOIN documents d ON d.id=c.doc_id WHERE c.tenant=$1 AND c.id = ANY($2::text[])", tenant, ids))

    async def count(self, tenant) -> int:
        return int(await self.db.fetchval("SELECT count(*) FROM chunks WHERE tenant=$1", tenant))


class PgGraphStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def upsert(self, tenant, nodes, edges) -> None:
        async with self.db.transaction() as c:
            await c.executemany(
                "INSERT INTO graph_nodes (id, tenant, name, type, description) VALUES ($1,$2,$3,$4,$5)"
                " ON CONFLICT (id) DO NOTHING",
                [(f"{tenant}:{n.id}", tenant, n.name, n.type, n.description) for n in nodes])
            await c.executemany(
                "INSERT INTO graph_edges (tenant, src, dst, relation, chunk_id) VALUES ($1,$2,$3,$4,$5)"
                " ON CONFLICT (tenant, src, dst, relation) DO NOTHING",
                [(tenant, e.src, e.dst, e.relation, e.chunk_id) for e in edges])

    async def find_nodes(self, tenant, ids) -> list[Node]:
        rows = await self.db.fetch(
            "SELECT name, type, description FROM graph_nodes WHERE tenant=$1 AND lower(name) = ANY($2::text[])",
            tenant, ids)
        return [Node(r["name"], r["type"], r["description"]) for r in rows]

    async def neighbors(self, tenant, ids):
        rows = await self.db.fetch(
            "SELECT src, dst, relation, chunk_id FROM graph_edges WHERE tenant=$1 AND (src = ANY($2::text[]) OR dst = ANY($2::text[])) LIMIT 500",
            tenant, ids)
        es = [Edge(r["src"], r["dst"], r["relation"], r["chunk_id"]) for r in rows]
        return await self.find_nodes(tenant, sorted({e.src for e in es} | {e.dst for e in es})), es

    async def dump(self, tenant):
        ns = await self.db.fetch("SELECT name, type, description FROM graph_nodes WHERE tenant=$1", tenant)
        es = await self.db.fetch("SELECT src, dst, relation, chunk_id FROM graph_edges WHERE tenant=$1", tenant)
        return ([Node(r["name"], r["type"], r["description"]) for r in ns],
                [Edge(r["src"], r["dst"], r["relation"], r["chunk_id"]) for r in es])

    async def put_communities(self, tenant, cs) -> None:
        async with self.db.transaction() as c:
            await c.execute("DELETE FROM graph_communities WHERE tenant=$1", tenant)
            await c.executemany(
                "INSERT INTO graph_communities (id, tenant, level, members, summary, embedding) VALUES ($1,$2,$3,$4,$5,$6::vector)",
                [(x.id, tenant, x.level, x.members, x.summary, vec(x.embedding)) for x in cs])

    async def search_communities(self, tenant, emb, k):
        rows = await self.db.fetch(
            "SELECT id, level, members, summary, 1 - (embedding <=> $2::vector) AS score FROM graph_communities"
            " WHERE tenant=$1 ORDER BY embedding <=> $2::vector LIMIT $3", tenant, vec(emb), k)
        return [Community(r["id"], r["level"], r["members"], r["summary"], [], float(r["score"])) for r in rows]


class PgMemoryStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def add(self, tenant, user, content, emb, ttl_days) -> str:
        mid = new_id("mem_")
        await self.db.execute(
            "INSERT INTO memories (id, tenant, user_id, content, embedding, expires_at)"
            " VALUES ($1,$2,$3,$4,$5::vector, CASE WHEN $6::int IS NULL THEN NULL ELSE now() + make_interval(days => $6::int) END)",
            mid, tenant, user, content, vec(emb), ttl_days)
        return mid

    async def search(self, tenant, user, emb, k) -> list[Hit]:
        rows = await self.db.fetch(
            "SELECT id, content, 1 - (embedding <=> $3::vector) AS score FROM memories"
            " WHERE tenant=$1 AND user_id=$2 AND (expires_at IS NULL OR expires_at > now())"
            " ORDER BY embedding <=> $3::vector LIMIT $4", tenant, user, vec(emb), k)
        return [Hit(r["id"], "", "memory", "memory", r["content"], float(r["score"])) for r in rows]

    async def delete_user(self, tenant, user) -> int:
        r = await self.db.execute("DELETE FROM memories WHERE tenant=$1 AND user_id=$2", tenant, user)
        return int(r.split()[-1])
