"""Evals (trust layer: "Ragas") and the CI/CD *eval gate*.

    agentic-evals --target inproc                       # CI: full stack in-process, mock LLM (pipeline gate)
    agentic-evals --target http --url https://canary…   # Argo Rollouts analysis Job against the canary
    agentic-evals --target http --url … --ragas         # add Ragas faithfulness/answer-relevancy (live LLM judge)

Per-case deterministic checks (status, expected facts, citations, forbidden strings, expected
approval tool) are aggregated into metrics compared against evals/thresholds.yaml. Exit code 1 =
gate failed → build is blocked (CI) or the canary is aborted and rolled back (CD).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import yaml


@dataclass
class CaseResult:
    id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    cost_usd: float = 0.0
    status: str = ""
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    adversarial: bool = False
    expects_citations: bool = False
    cited: bool = False


def load_dataset(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]


def check_case(case: dict[str, Any], resp: dict[str, Any], latency: float) -> CaseResult:
    res = resp.get("result") or resp  # sync run returns RunResult; task fetch nests it
    status = res.get("status") or "error"
    answer = res.get("answer") or ""
    fails: list[str] = []
    if "expect_status" in case and status != case["expect_status"]:
        fails.append(f"status {status!r} != {case['expect_status']!r}")
    for s in case.get("expect_contains", []):
        if s.lower() not in answer.lower():
            fails.append(f"missing {s!r}")
    for s in case.get("expect_not_contains", []):
        if s.lower() in answer.lower():
            fails.append(f"forbidden {s!r} present")
    cites = res.get("citations") or []
    if case.get("expect_citations") and not cites:
        fails.append("no citations")
    if tool := case.get("expect_pending_tool"):
        if tool not in [a["tool"] for a in res.get("pending_approvals", [])]:
            fails.append(f"no pending approval for {tool}")
    if case.get("expect_no_pending") and res.get("pending_approvals"):
        fails.append("unexpected pending approvals")
    return CaseResult(
        id=case["id"], passed=not fails, failures=fails, latency_s=latency, status=status, answer=answer,
        cost_usd=float((res.get("usage") or {}).get("cost_usd", 0)), contexts=[c["snippet"] for c in cites],
        adversarial=bool(case.get("adversarial")), expects_citations=bool(case.get("expect_citations")), cited=bool(cites),
    )


def aggregate(rs: list[CaseResult]) -> dict[str, float]:
    n = len(rs) or 1
    adv = [r for r in rs if r.adversarial]
    cit = [r for r in rs if r.expects_citations]
    lat = sorted(r.latency_s for r in rs)
    return {
        "task_success_rate": sum(r.passed for r in rs) / n,
        "adversarial_block_rate": (sum(r.passed for r in adv) / len(adv)) if adv else 1.0,
        "citation_rate": (sum(r.cited for r in cit) / len(cit)) if cit else 1.0,
        "p95_latency_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
        "avg_cost_usd": statistics.fmean(r.cost_usd for r in rs) if rs else 0.0,
    }


def gate(metrics: dict[str, float], thresholds: dict[str, dict[str, float]]) -> list[str]:
    bad = []
    for name, rule in thresholds.items():
        v = metrics.get(name)
        if v is None:
            continue
        if "min" in rule and v < rule["min"]:
            bad.append(f"{name}={v:.3f} < min {rule['min']}")
        if "max" in rule and v > rule["max"]:
            bad.append(f"{name}={v:.3f} > max {rule['max']}")
    return bad


def ragas_scores(rs: list[CaseResult], questions: dict[str, str]) -> dict[str, float]:
    """Optional LLM-judged metrics. Requires `pip install .[eval]` and a judge model (e.g. OPENAI_API_KEY)."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        print("ragas not installed; skipping (pip install '.[eval]')", file=sys.stderr)
        return {}
    rows = [r for r in rs if r.contexts and r.answer and r.status == "succeeded"]
    if not rows:
        return {}
    ds = Dataset.from_dict({"question": [questions[r.id] for r in rows], "answer": [r.answer for r in rows],
                            "contexts": [r.contexts for r in rows]})
    out = evaluate(ds, metrics=[faithfulness, answer_relevancy])
    df = out.to_pandas()
    return {"ragas_faithfulness": float(df["faithfulness"].mean()), "ragas_answer_relevancy": float(df["answer_relevancy"].mean())}


async def run_cases(client: httpx.AsyncClient, headers: dict[str, str], cases: list[dict[str, Any]],
                    concurrency: int = 4, path: str = "/v1/agent/runs") -> list[CaseResult]:
    sem = asyncio.Semaphore(concurrency)

    async def one(case: dict[str, Any]) -> CaseResult:
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.post(path, json={"query": case["query"]}, headers=headers, timeout=180)
                body = r.json()
                if r.status_code >= 400:
                    body = {"status": "error", "answer": json.dumps(body.get("error", body))}
            except Exception as e:  # network failure counts as a failed case, not a crashed gate
                body = {"status": "error", "answer": str(e)}
            return check_case(case, body, time.perf_counter() - t0)

    return list(await asyncio.gather(*(one(c) for c in cases)))


async def amain(a: argparse.Namespace) -> int:
    from agentic.common.logging import setup_logging

    setup_logging("WARNING")
    cases = load_dataset(a.dataset)
    thresholds = yaml.safe_load(pathlib.Path(a.thresholds).read_text())
    if a.target == "inproc":
        from agentic.devstack import Stack

        async with Stack() as st:
            results = await run_cases(st.api, st.auth("eval-user"), cases)
    else:
        key = os.environ.get("EVAL_API_KEY")
        token = os.environ.get("EVAL_TOKEN")
        if a.mint_token:  # in-cluster: mint a gateway identity for the dedicated eval tenant (never prod data)
            from agentic.config import get_settings
            from agentic.trust.identity import IdentityProvider

            st = get_settings()
            token = IdentityProvider(st.identity_secret, 900).issue("gateway", a.tenant, on_behalf_of="eval-bot")
        headers = {"x-api-key": key} if key else {"authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(base_url=a.url) as client:
            results = await run_cases(client, headers, cases, path=a.path)
    metrics = aggregate(results)
    if a.ragas:
        metrics.update(ragas_scores(results, {c["id"]: c["query"] for c in cases}))
    failures = gate(metrics, thresholds)
    report = {"target": a.target, "git_sha": os.environ.get("GITHUB_SHA") or os.environ.get("BUILD_SOURCEVERSION"),
              "metrics": metrics, "gate_failures": failures, "passed": not failures,
              "cases": [r.__dict__ | {"answer": r.answer[:300], "contexts": []} for r in results]}
    pathlib.Path(a.report).write_text(json.dumps(report, indent=2))
    print(f"{'ID':28} {'RESULT':7} {'STATUS':18} LATENCY  FAILURES")
    for r in results:
        print(f"{r.id:28} {'PASS' if r.passed else 'FAIL':7} {r.status:18} {r.latency_s:6.2f}s  {'; '.join(r.failures)}")
    print("\nmetrics:", json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    if failures:
        print("EVAL GATE FAILED:", "; ".join(failures), file=sys.stderr)
        return 1
    print("eval gate passed")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="agentic-evals")
    p.add_argument("--target", choices=["inproc", "http"], default="inproc")
    p.add_argument("--url", default=os.environ.get("EVAL_URL", "http://localhost:8001"))
    p.add_argument("--dataset", default="evals/dataset.jsonl")
    p.add_argument("--thresholds", default="evals/thresholds.yaml")
    p.add_argument("--report", default="eval-report.json")
    p.add_argument("--ragas", action="store_true")
    p.add_argument("--path", default="/v1/agent/runs", help="/v1/agent/runs (gateway) or /v1/runs (orchestrator canary)")
    p.add_argument("--mint-token", action="store_true", help="mint a gateway identity token from IDENTITY_SECRET")
    p.add_argument("--tenant", default="eval", help="tenant for minted tokens")
    raise SystemExit(asyncio.run(amain(p.parse_args())))


if __name__ == "__main__":
    main()
