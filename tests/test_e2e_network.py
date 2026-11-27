"""Real sockets: four uvicorn servers (gateway, orchestrator, MCP tool server, internal API) talking
over HTTP, including the MCP streamable-HTTP transport. Memory infra + mock LLM."""

import asyncio
import socket

import httpx
import pytest_asyncio
import uvicorn

from agentic.backbone.worker import Worker
from agentic.container import Container
from agentic.devstack import SAMPLE_DOCS, make_settings
from agentic.gateway.app import create_app as gateway
from agentic.orchestration.app import create_app as orchestrator
from agentic.orchestration.runner import Orchestrator
from agentic.tools.internal_api import create_app as internal
from agentic.tools.server import create_app as tools


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Net:
    pass


@pytest_asyncio.fixture
async def net():
    ports = {n: free_port() for n in ("gw", "orch", "mcp", "int")}
    s = make_settings(gateway_url=f"http://127.0.0.1:{ports['gw']}", orchestrator_url=f"http://127.0.0.1:{ports['orch']}",
                      mcp_url=f"http://127.0.0.1:{ports['mcp']}/mcp", internal_api_url=f"http://127.0.0.1:{ports['int']}")
    c = Container.build(s)
    orch_core = Orchestrator(c)
    apps = {"gw": gateway(c), "orch": orchestrator(c, orch_core), "mcp": tools(c), "int": internal(c)}
    servers = {k: uvicorn.Server(uvicorn.Config(a, host="127.0.0.1", port=ports[k], log_level="error", lifespan="on"))
               for k, a in apps.items()}
    tasks = {k: asyncio.create_task(sv.serve()) for k, sv in servers.items()}
    for _ in range(200):
        if all(sv.started for sv in servers.values()):
            break
        await asyncio.sleep(0.05)
    assert all(sv.started for sv in servers.values())
    n = Net()
    n.c, n.s, n.settings, n.orch, n.apps = c, s, s, orch_core, apps
    n.worker = Worker(c, orch_core)
    n.http = httpx.AsyncClient(base_url=s.gateway_url, timeout=60)
    from agentic.devstack import Stack

    n.auth = lambda sub="alice", tenant="default", scopes=None: Stack.auth(n, sub, tenant, scopes)  # reuse token helper
    n.token = lambda sub="alice", tenant="default", scopes=None: Stack.token(n, sub, tenant, scopes)
    for f in sorted(SAMPLE_DOCS.glob("*.md")):
        await orch_core.knowledge.ingest("default", f.name, f.stem, f.read_text())
    yield n
    await n.http.aclose()
    for sv in servers.values():
        sv.should_exit = True
    await asyncio.gather(*tasks.values(), return_exceptions=True)


async def test_full_flow_over_real_http(net):
    r = await net.http.post("/v1/agent/runs", json={"query": "How long do access grants last before renewal?"}, headers=net.auth())
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "succeeded" and "90 days" in r.json()["answer"]

    r = await net.http.post("/v1/agent/runs", json={"query": "Create a ticket for the failing nightly ingestion job"}, headers=net.auth("alice"))
    ap = r.json()["pending_approvals"][0]
    d = await net.http.post(f"/v1/agent/approvals/{ap['id']}/decision", json={"approve": True}, headers=net.auth("bob"))
    assert d.json()["status"] == "executed", d.text
    assert len(net.apps["int"].state.tickets.rows) == 1  # MCP → internal API over real HTTP


async def test_async_over_real_http_with_worker(net):
    r = await net.http.post("/v1/agent/runs", json={"query": "What is NNDM?", "async": True}, headers=net.auth())
    tid = r.json()["task_id"]
    runner = asyncio.create_task(net.worker.run())
    try:
        for _ in range(200):
            t = (await net.http.get(f"/v1/agent/tasks/{tid}", headers=net.auth())).json()
            if t["status"] == "succeeded":
                break
            await asyncio.sleep(0.05)
        assert t["status"] == "succeeded" and "Data Marketplace" in t["result"]["answer"]
    finally:
        net.worker.stop()
        await runner


async def test_kill_switch_over_real_http(net):
    adm = net.auth("root", scopes=["admin:killswitch"])
    await net.http.put("/v1/admin/killswitch", json={"scope": "global", "reason": "drill"}, headers=adm)
    assert (await net.http.post("/v1/agent/runs", json={"query": "What is NNDM?"}, headers=net.auth())).status_code == 503
    await net.http.delete("/v1/admin/killswitch/global", headers=adm)
    assert (await net.http.post("/v1/agent/runs", json={"query": "What is NNDM?"}, headers=net.auth())).status_code == 200
