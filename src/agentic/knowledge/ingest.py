"""Batch ingestion CLI (run as a K8s Job / CronJob):

    agentic-ingest ./data/sample_docs --tenant default [--no-graph] [--communities]
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib

from agentic.config import get_settings
from agentic.container import Container
from agentic.knowledge.service import Knowledge
from agentic.models.client import LLMClient
from agentic.models.router import ModelRouter


async def run(path: str, tenant: str, graph: bool, communities: bool) -> None:
    s = get_settings()
    c = Container.build(s)
    await c.start()
    llm = LLMClient(c)
    k = Knowledge(c, llm, ModelRouter(llm, s))
    total = 0
    for f in sorted(pathlib.Path(path).rglob("*")):
        if f.suffix.lower() in {".md", ".txt"} and f.is_file():
            n = await k.ingest(tenant, str(f.name), f.stem.replace("_", " "), f.read_text(), extract_graph=graph)
            print(f"ingested {f.name}: {n} chunks")
            total += n
    if communities:
        print("communities:", await k.build_communities(tenant))
    print("total chunks:", total)
    await c.close()


def main() -> None:
    p = argparse.ArgumentParser(prog="agentic-ingest")
    p.add_argument("path")
    p.add_argument("--tenant", default="default")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--communities", action="store_true")
    a = p.parse_args()
    asyncio.run(run(a.path, a.tenant, not a.no_graph, a.communities))


if __name__ == "__main__":
    main()
