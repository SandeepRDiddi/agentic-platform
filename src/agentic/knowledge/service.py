"""Knowledge layer (box 6): hybrid RAG + GraphRAG + long-term memory.

  retrieve():  vector ⊕ keyword (reciprocal-rank fusion)  +  graph local search (entity link →
               2-hop expansion → supporting chunks)  +  graph global search (community summaries)
  ingest():    chunk → embed → store; LLM entity/relation extraction → graph; communities → summaries
  memory:      Redis short-term conversation window + pgvector per-user long-term facts (PII-free, TTL'd)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import networkx as nx
from pydantic import BaseModel, Field

from agentic.common.schemas import Citation, new_id
from agentic.container import Container
from agentic.knowledge.stores import (
    Community,
    Edge,
    GraphStore,
    Hit,
    MemGraphStore,
    MemMemoryStore,
    MemoryStore,
    MemVectorStore,
    Node,
    PgGraphStore,
    PgMemoryStore,
    PgVectorStore,
    VectorStore,
)
from agentic.models.client import LLMClient
from agentic.models.router import ModelRouter

log = logging.getLogger(__name__)


def chunk_text(text: str, max_words: int = 180, overlap: int = 30) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    for p in paras:
        words = p.split()
        if len(words) > max_words:  # hard-split oversized paragraphs
            if cur:
                chunks.append(" ".join(cur)); cur = []  # noqa: E702
            for i in range(0, len(words), max_words - overlap):
                chunks.append(" ".join(words[i : i + max_words]))
            continue
        if sum(len(x.split()) for x in cur) + len(words) > max_words and cur:
            chunks.append(" ".join(cur))
            cur = cur[-1:] if overlap else []
        cur.append(p)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


@dataclass
class RetrievedContext:
    chunks: list[Hit] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    communities: list[Community] = field(default_factory=list)
    flagged_injection: bool = False

    def citations(self) -> list[Citation]:
        return [Citation(id=h.id, source=h.source, snippet=h.content[:240], score=round(h.score, 4)) for h in self.chunks]


class ExtractedGraph(BaseModel):
    entities: list[dict] = Field(default_factory=list)
    relations: list[dict] = Field(default_factory=list)


class Facts(BaseModel):
    facts: list[str] = Field(default_factory=list)


class Knowledge:
    def __init__(self, c: Container, llm: LLMClient, router: ModelRouter,
                 vectors: VectorStore | None = None, graph: GraphStore | None = None,
                 memory: MemoryStore | None = None) -> None:
        self.c, self.llm, self.router = c, llm, router
        db = c.db
        self.vectors = vectors or (PgVectorStore(db) if db else c.stores.setdefault("vectors", MemVectorStore()))
        self.graph = graph or (PgGraphStore(db) if db else c.stores.setdefault("graph", MemGraphStore()))
        self.memory = memory or (PgMemoryStore(db) if db else c.stores.setdefault("memory", MemMemoryStore()))

    async def _embed(self, texts: list[str], tenant: str, user: str | None = None, agent: str = "research") -> list[list[float]]:
        return await self.llm.embed(texts, agent=agent, tenant=tenant, user=user)

    # ── retrieval ────────────────────────────────────────────────────────────────────────
    async def retrieve(self, tenant: str, query: str, *, k: int = 6, user: str | None = None) -> RetrievedContext:
        emb = (await self._embed([query], tenant, user))[0]
        vec_hits = await self.vectors.vector_search(tenant, emb, k * 2)
        kw_hits = await self.vectors.keyword_search(tenant, query, k * 2)
        fused = self._rrf([vec_hits, kw_hits])

        facts, graph_chunk_ids, flagged = await self._graph_local(tenant, query)
        have = {h.id for h in fused}
        extra = [h for h in await self.vectors.get_chunks(tenant, [i for i in graph_chunk_ids if i not in have])]
        for h in extra:
            h.score = 0.01
        top = (fused + extra)[: k + 2]

        comms = [x for x in await self.c_search_comm(tenant, emb) if x.score > 0.15]

        guard = self.c.guardrails
        for h in top:
            h.content, f = guard.sanitize_context(h.content)
            flagged = flagged or f
        return RetrievedContext(top, facts, comms, flagged)

    async def c_search_comm(self, tenant: str, emb: list[float]) -> list[Community]:
        return await self.graph.search_communities(tenant, emb, 2)

    @staticmethod
    def _rrf(lists: list[list[Hit]], k: int = 60) -> list[Hit]:
        score: dict[str, float] = {}
        byid: dict[str, Hit] = {}
        for lst in lists:
            for rank, h in enumerate(lst):
                score[h.id] = score.get(h.id, 0) + 1 / (k + rank + 1)
                byid[h.id] = h
        out = []
        for i, s in sorted(score.items(), key=lambda x: -x[1]):
            h = byid[i]
            h.score = s
            out.append(h)
        return out

    async def _graph_local(self, tenant: str, query: str, hops: int = 2) -> tuple[list[str], list[str], bool]:
        toks = re.findall(r"[A-Za-z0-9]+", query.lower())
        grams = {" ".join(toks[i : i + n]) for n in (1, 2, 3) for i in range(len(toks) - n + 1)}
        seeds = [n.id for n in await self.graph.find_nodes(tenant, sorted(grams))]
        if not seeds:
            return [], [], False
        seen, frontier, edges = set(seeds), set(seeds), []
        for _ in range(hops):
            if not frontier:
                break
            _, es = await self.graph.neighbors(tenant, sorted(frontier))
            edges.extend(es)
            frontier = ({e.src for e in es} | {e.dst for e in es}) - seen
            seen |= frontier
        facts, chunk_ids, flagged = [], [], False
        for e in {(e.src, e.relation, e.dst, e.chunk_id) for e in edges}:
            f, bad = self.c.guardrails.sanitize_context(f"{e[0]} -[{e[1]}]-> {e[2]}")
            flagged = flagged or bad
            facts.append(f)
            if e[3]:
                chunk_ids.append(e[3])
        return sorted(facts)[:25], chunk_ids[:10], flagged

    # ── ingestion ────────────────────────────────────────────────────────────────────────
    async def ingest(self, tenant: str, source: str, title: str, text: str, *, extract_graph: bool = True) -> int:
        pieces = chunk_text(text)
        if not pieces:
            return 0
        embs = await self._embed(pieces, tenant, agent="ingest")
        doc_id = "doc_" + re.sub(r"\W+", "_", f"{tenant}_{source}")[:60]
        chunks = [(f"{doc_id}#{i}", i, p, e) for i, (p, e) in enumerate(zip(pieces, embs, strict=True))]
        await self.vectors.upsert_document(tenant, doc_id, source, title, chunks)
        if extract_graph:
            for cid, _, p, _ in chunks:
                await self._extract_graph(tenant, cid, p)
        return len(chunks)

    async def _extract_graph(self, tenant: str, chunk_id: str, text: str) -> None:
        system = ("ROLE:graph_extract\nExtract entities and relations from the text. Reply JSON: "
                  '{"entities":[{"name","type"}],"relations":[{"src","dst","relation"}]}')
        try:
            _, g = await self.router.complete(
                agent="ingest", system=system, user=f"<text>{json.dumps(text)}</text>", tenant=tenant,
                user_id=None, schema=ExtractedGraph)
        except Exception as e:
            log.warning("graph extraction failed for %s: %s", chunk_id, e)
            return
        assert g is not None
        nodes = {Node(str(e["name"]).strip(), str(e.get("type", "entity"))).id: Node(str(e["name"]).strip(), str(e.get("type", "entity")))
                 for e in g.entities if e.get("name")}
        edges = []
        for r in g.relations:
            if r.get("src") and r.get("dst"):
                s, d = Node(str(r["src"]).strip()), Node(str(r["dst"]).strip())
                nodes.setdefault(s.id, s)
                nodes.setdefault(d.id, d)
                edges.append(Edge(s.id, d.id, str(r.get("relation", "related_to"))[:60], chunk_id))
        await self.graph.upsert(tenant, list(nodes.values()), edges)

    async def build_communities(self, tenant: str) -> int:
        """GraphRAG global layer: Louvain communities → LLM summaries → embedded for retrieval."""
        nodes, edges = await self.graph.dump(tenant)
        g = nx.Graph()
        g.add_nodes_from(n.id for n in nodes)
        g.add_edges_from((e.src, e.dst) for e in edges)
        comms = [sorted(c) for c in nx.community.louvain_communities(g, seed=42) if len(c) >= 2]
        out: list[Community] = []
        for members in comms:
            rel = [f"{e.src} {e.relation} {e.dst}" for e in edges if e.src in members and e.dst in members][:30]
            res, _ = await self.router.complete(
                agent="ingest", system="ROLE:summarize\nSummarize this entity community in 2-3 sentences.",
                user=f"<members>{json.dumps(members[:40])}</members>\n<relations>{json.dumps(rel)}</relations>",
                tenant=tenant, user_id=None)
            out.append(Community(new_id("comm_"), 0, members, res.content.strip()))
        if out:
            embs = await self._embed([c.summary for c in out], tenant, agent="ingest")
            for c, e in zip(out, embs, strict=True):
                c.embedding = e
        await self.graph.put_communities(tenant, out)
        return len(out)

    # ── memory ───────────────────────────────────────────────────────────────────────────
    async def history(self, tenant: str, conversation_id: str | None) -> list[dict[str, str]]:
        if not conversation_id:
            return []
        raw = await self.c.kv.get(f"conv:{tenant}:{conversation_id}")
        return json.loads(raw) if raw else []

    async def long_term(self, tenant: str, user: str, query: str, k: int = 3) -> list[str]:
        emb = (await self._embed([query], tenant, user, agent="memory"))[0]
        return [h.content for h in await self.memory.search(tenant, user, emb, k) if h.score > 0.2]

    async def remember(self, tenant: str, user: str, conversation_id: str | None, user_msg: str, answer: str) -> int:
        if conversation_id:
            hist = (await self.history(tenant, conversation_id))[-8:]
            hist += [{"role": "user", "content": user_msg}, {"role": "assistant", "content": answer[:1500]}]
            await self.c.kv.set(f"conv:{tenant}:{conversation_id}", json.dumps(hist), ttl=86400)
        safe_msg, _ = self.c.guardrails.redact(user_msg)  # PII never reaches the extractor or the store
        try:
            _, facts = await self.router.complete(
                agent="memory", system="ROLE:memory\nExtract durable user facts/preferences worth remembering. JSON {\"facts\":[...]}",
                user=f"<turn>{json.dumps({'user': safe_msg})}</turn>", tenant=tenant, user_id=user, schema=Facts)
        except Exception as e:
            log.warning("memory extraction failed: %s", e)
            return 0
        stored = 0
        for f in (facts.facts if facts else [])[:5]:
            if "[REDACTED" in f or self.c.guardrails.scan(f) or self.c.guardrails._injection(f):  # noqa: SLF001
                continue  # never persist PII/secrets/instructions
            emb = (await self._embed([f], tenant, user, agent="memory"))[0]
            await self.memory.add(tenant, user, f, emb, ttl_days=180)
            stored += 1
        return stored

    async def forget(self, tenant: str, user: str) -> int:
        return await self.memory.delete_user(tenant, user)
