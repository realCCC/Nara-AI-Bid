"""최소 실행 경로 통과 기준 (4번 문서 13장)."""
import json

import pytest

from backend.tools.loaders import load_company_file, load_tenders
from scripts.run_demo import SAMPLE, run

EXPECTED_EXCLUDED_RULES = sorted([["H1"], ["H1"], ["H3"], ["H3"], ["H5"], ["H6"], ["H6"]])


@pytest.fixture(scope="module")
def result():
    return run(SAMPLE / "tenders_sample.json", SAMPLE / "company.json", SAMPLE / "region_limits_sample.csv")


def test_top10_sorted(result):
    scores = [r["soft_score"] for r in result["top"]]
    assert len(scores) == 10
    assert scores == sorted(scores, reverse=True)


def test_excluded_rule_ids(result):
    assert len(result["excluded"]) == 7
    assert sorted(e["rules"] for e in result["excluded"]) == EXPECTED_EXCLUDED_RULES


def test_score_ranges(result):
    for r in result["top"]:
        assert 0 <= r["soft_score"] <= 100
        assert all(0 <= v <= 1 for v in r["scores"].values())


def test_reasons_cite_real_fields(result):
    tenders = {t["bidNtceNo"]: t for t in load_tenders(SAMPLE / "tenders_sample.json")}
    for r in result["top"]:
        assert r["reasons"]
        for reason in r["reasons"]:
            scope, _, name = reason["field"].partition(".")
            assert scope == "tender" and name in tenders[r["bidNtceNo"]]


def test_deterministic(result):
    again = run(SAMPLE / "tenders_sample.json", SAMPLE / "company.json", SAMPLE / "region_limits_sample.csv")
    assert json.dumps(again, ensure_ascii=False) == json.dumps(result, ensure_ascii=False)


def test_password_not_in_profile():
    assert "비밀번호" not in load_company_file(SAMPLE / "company.json")
