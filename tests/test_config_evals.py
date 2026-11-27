import pytest

from agentic.config import Settings
from agentic.trust import evals


def prod(**over):
    base = dict(env="prod", llm_mode="live", jwt_secret="j" * 40, identity_secret="i" * 40,
                database_url="postgresql://x", redis_url="redis://x", kafka_bootstrap="k:9092")
    base.update(over)
    return Settings(**base)


def test_prod_refuses_dev_defaults():
    prod().validate_for_prod()
    for bad in (dict(jwt_secret="dev-secret-change-me-dev-secret-change-me"),
                dict(identity_secret="dev-identity-secret-change-me-dev-identity"),
                dict(llm_mode="mock"), dict(database_url=None), dict(redis_url=None), dict(kafka_bootstrap=None),
                dict(dev_token_endpoint=True), dict(killswitch_fail_closed=False)):
        with pytest.raises(RuntimeError):
            prod(**bad).validate_for_prod()


def test_dev_allows_defaults():
    Settings(env="dev").validate_for_prod()


def test_eval_gate_logic():
    rs = [evals.CaseResult("a", True, cited=True, expects_citations=True, latency_s=1),
          evals.CaseResult("b", False, adversarial=True, latency_s=2)]
    m = evals.aggregate(rs)
    assert m["task_success_rate"] == 0.5 and m["adversarial_block_rate"] == 0.0
    fails = evals.gate(m, {"task_success_rate": {"min": 0.9}, "p95_latency_s": {"max": 5}, "ragas_faithfulness": {"min": 0.8}})
    assert len(fails) == 1  # ragas metric absent → not enforced


def test_check_case():
    case = {"id": "x", "expect_status": "succeeded", "expect_contains": ["5 minutes"], "expect_not_contains": ["4111"], "expect_citations": True}
    ok = evals.check_case(case, {"status": "succeeded", "answer": "within 5 minutes", "citations": [{"snippet": "s"}]}, 0.1)
    bad = evals.check_case(case, {"status": "blocked", "answer": "4111", "citations": []}, 0.1)
    assert ok.passed and not bad.passed and len(bad.failures) == 4


async def test_inproc_eval_gate_passes_on_main(tmp_path):
    import argparse

    rc = await evals.amain(argparse.Namespace(target="inproc", url="", dataset="evals/dataset.jsonl", path="/v1/agent/runs", mint_token=False, tenant="eval",
                                              thresholds="evals/thresholds.yaml", report=str(tmp_path / "eval-report.json"), ragas=False))
    assert rc == 0
