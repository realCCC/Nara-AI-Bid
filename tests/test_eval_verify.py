import pytest

from scripts.eval_verify import company_metrics, ndcg, precision_at, summarize

GOLD = {"A": "추천", "B": "부적격", "C": "적격-비추천"}


def test_ndcg_and_precision():
    assert ndcg(["A", "C", "B"], GOLD) == 1.0
    assert ndcg(["B", "C", "A"], GOLD) < ndcg(["A", "B", "C"], GOLD)
    assert precision_at(["A", "B", "C"], GOLD) == pytest.approx(1 / 3)
    assert ndcg(["X"], GOLD) is None and precision_at(["X"], GOLD) is None


def test_company_metrics_and_summary():
    m = company_metrics(["B", "A", "C"], ["A", "C", "B"], {"B"}, GOLD)
    assert m["bad"] == 1 and (m["below_before"], m["below_after"], m["pairs"]) == (0, 2, 2)
    assert m["flagged_bad"] == 1 and m["changed"]
    assert m["ndcg_after"] == 1.0 > m["ndcg_before"]
    s = summarize([m, company_metrics(["A", "C", "B"], ["A", "C", "B"], set(), GOLD)])
    assert s["부적격 하위 배치율"] == {"before": 0.5, "after": 1.0}
    assert s["플래그 정밀도"] == 1.0 and s["플래그 재현율"] == 0.5
    assert s["순위 변경률"] == 0.5
