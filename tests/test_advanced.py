"""고도화: 가중치 계산, 인터뷰 → 가중치 변환, 키워드 의미 확장, 검증 실패 폴백, 임베딩 실패 폴백."""
import numpy as np
import pytest

from backend import config
from backend.graphs.proposal import apply_verification
from backend.tools import llm, search
from backend.tools import interview
from backend.tools.interview import boost_weights, interview_filter, to_prefs
from backend.tools.rules import recommend_by_rules, score_items, soft_score
from tests.test_rules import COMPANY, tender

BASE = (0.35, 0.30, 0.20, 0.15)


# ---------- 가중치 계산 ----------
def test_score_items_sum_to_total():
    t = tender()
    s = soft_score(t, COMPANY, BASE)
    rows = score_items(t, COMPANY, s, BASE)
    assert [r["key"] for r in rows] == ["S1", "S2", "S3", "S4"]
    assert [r["weight"] for r in rows] == list(BASE)
    assert sum(r["points"] for r in rows) == pytest.approx(s["total"], abs=0.2)
    assert all(r["reason"] and not r["interview"] for r in rows)


def test_recommend_includes_score_detail():
    top = recommend_by_rules([tender()], COMPANY, {})["top"]
    assert top[0]["score_detail"]["total"] == top[0]["soft_score"]


# ---------- 인터뷰 → 가중치 변환 ----------
def test_boost_weights_keeps_sum():
    w = boost_weights(BASE, "S2")
    assert w[1] == pytest.approx(0.45)
    assert sum(w) == pytest.approx(1.0, abs=1e-3)
    assert all(a > b for a, b in zip(BASE[::2], w[::2]))  # 나머지는 비례 감소


def test_interview_skip_and_unknown_are_default():
    for answers in (None, {}, {"priority": "all", "region": None}, {"priority": "S9", "x": "1"}):
        p = to_prefs(answers, BASE)
        assert p["answers"] == {} and p["weights"] == BASE and p["budget"] is None


def test_interview_marks_adjusted_rows():
    p = to_prefs({"priority": "S1"}, BASE)
    t = tender()
    rows = score_items(t, COMPANY, soft_score(t, COMPANY, p["weights"]), p["weights"], p["base_weights"])
    assert rows[0]["weight"] == pytest.approx(0.5) and rows[0]["base_weight"] == 0.35
    assert all(r["interview"] for r in rows)


def test_interview_filters_and_bonus():
    p = to_prefs({"budget": "lt100"}, BASE)
    assert not interview_filter(tender(budget=90_000_000), COMPANY, p)
    assert interview_filter(tender(budget=150_000_000), COMPANY, p)
    assert not interview_filter(tender(budget=None), COMPANY, p)  # 예산 없음은 통과

    p = to_prefs({"region": "only"}, BASE)
    assert interview_filter(tender(dminsttNm="부산광역시 교육청"), COMPANY, p)
    assert not interview_filter(tender(dminsttNm="서울특별시 교육청"), COMPANY, p)

    p = to_prefs({"schedule": "14"}, BASE)
    assert interview_filter(tender(bidNtceDt="2026-05-01 09:00:00", bidClseDt="2026-05-08 10:00:00"), COMPANY, p)
    assert not interview_filter(tender(bidNtceDt="2026-05-01 09:00:00", bidClseDt=""), COMPANY, p)

    res = recommend_by_rules([tender(dminsttNm="서울특별시 교육청")], COMPANY, {}, prefs=to_prefs({"region": "bonus"}, BASE))
    top = res["top"][0]
    assert top["score_detail"]["items"][-1]["key"] == "B1"
    assert top["soft_score"] == pytest.approx(soft_score(tender(), COMPANY, BASE)["total"] + 5, abs=0.1)

    res = recommend_by_rules([tender(budget=150_000_000)], COMPANY, {}, prefs=to_prefs({"budget": "lt100"}, BASE))
    assert res["excluded"][0]["rules"] == ["I1"] and res["rule_counts"]["I1"] == 1


# ---------- 검증 실패 폴백 ----------
RECS = [{"rank": i, "bidNtceNo": f"T{i}", "bidNtceNm": f"공고{i}", "soft_score": 90 - i, "reasons": []} for i in range(1, 7)]
IDS = ["T1", "T2", "T3"]


def _result(order, items=None):
    return {"order": order, "items": [{"bidNtceNo": i, "fit_score": 80, "comment": "c"} for i in (items or order)]}


def test_check_verify_failures():
    assert llm.check_verify(None, IDS)[1].startswith("형식 오류")
    assert llm.check_verify({"order": IDS}, IDS)[1].startswith("형식 오류")
    assert llm.check_verify(_result(["T1", "T2", "T9"]), IDS)[1] == "없는 공고ID: T9"
    assert "누락" in llm.check_verify(_result(["T1", "T2"]), IDS)[1]
    assert llm.check_verify(_result(["T3", "T1", "T2"]), IDS) == (_result(["T3", "T1", "T2"]), None)


def test_apply_verification_keeps_order_on_failure():
    recs, v = apply_verification(RECS, None, "LLM 키 없음", 3)
    assert recs == RECS and v == {"status": "skipped", "reason": "LLM 키 없음", "checked": 3, "changed": False}


def test_apply_verification_reorders_head_only():
    recs, v = apply_verification(RECS, _result(["T3", "T1", "T2"]), None, 3)
    assert [r["bidNtceNo"] for r in recs] == ["T3", "T1", "T2", "T4", "T5", "T6"]
    assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5, 6]
    assert recs[0]["verify"]["rank_before"] == 3 and "verify" not in recs[3]
    assert v["status"] == "passed" and v["changed"]


class _FakeLLM:
    def __init__(self, out):
        self.out = out

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


@pytest.mark.parametrize("out, reason", [
    (RuntimeError("timeout"), "예외: RuntimeError"),
    (llm.VerifyResult(order=["T1", "T2", "T99"], items=[]), "없는 공고ID: T99"),
    (None, "형식 오류: 빈 응답"),
])
def test_verify_ranking_failures(monkeypatch, out, reason):
    monkeypatch.setattr(llm, "get_llm", lambda: _FakeLLM(out))
    tenders = {r["bidNtceNo"]: {} for r in RECS}
    assert llm.verify_ranking({}, RECS[:3], tenders) == (None, reason)


def test_verify_ranking_no_key(monkeypatch):
    monkeypatch.setattr(llm, "get_llm", lambda: None)
    assert llm.verify_ranking({}, RECS[:3], {}) == (None, "LLM 키 없음")


# ---------- 임베딩 실패 폴백 ----------
DOCS = {"A": "국제 심포지엄 행사 운영", "B": "도로 포장 공사", "C": "축제 기획 및 운영 대행"}


def test_embedding_off_is_keyword_fallback(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "keyword")
    r = search.rerank(search.hybrid_search(DOCS, "행사 축제 운영"), DOCS)
    assert r["path"] == "keyword_fallback" and r["rerank"] == "skipped"
    assert r["order"][-1] == "B" and r["scores"]["B"]["bm25"] is None  # 겹침 없는 문서는 꼴찌
    assert all(r["scores"][i]["dense"] is None for i in DOCS)


def test_embedding_failure_falls_back(monkeypatch):
    import backend.tools.retrieval as retrieval

    def boom():
        raise ImportError("No module named 'sentence_transformers'")

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bge")
    monkeypatch.setattr(retrieval, "_bge", boom)
    r = search.hybrid_search(DOCS, "행사 축제 운영", rule_order=["C", "A", "B"])
    assert r["path"] == "keyword_fallback" and "sentence_transformers" in r["reason"]
    assert set(r["order"]) == set(DOCS) and r["lists"] == ["rule", "bm25"]


@pytest.fixture(autouse=True)
def _isolated_vec_cache(monkeypatch, tmp_path):
    """가짜 벡터가 실제 디스크 캐시(data/emb_cache.npz)에 섞이지 않게."""
    monkeypatch.setattr(config, "EMB_CACHE", tmp_path / "emb.npz")
    monkeypatch.setattr(search, "_DISK_LOADED", True)
    monkeypatch.setattr(search, "_VEC_CACHE", {})


class _FakeModel:
    def encode(self, texts, normalize_embeddings=True, **kw):
        return np.array([[1.0, 0.0] if "도로" not in t else [0.0, 1.0] for t in texts])


def test_semantic_with_reranker_failure(monkeypatch):
    import backend.tools.retrieval as retrieval

    def no_reranker():
        raise OSError("model download failed")

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bge")
    monkeypatch.setattr(retrieval, "_bge", lambda: _FakeModel())
    monkeypatch.setattr(search, "_reranker", no_reranker)
    r = search.hybrid_search(DOCS, "행사 축제 운영")
    assert r["path"] == "semantic" and r["scores"]["A"]["dense"] == 1.0
    rr = search.rerank(r, DOCS)
    assert rr["rerank"] == "skipped" and "download" in rr["rerank_reason"] and rr["order"] == r["order"]


def test_rrf():
    fused = search.rrf({"a": ["X", "Y"], "b": ["Y", "X"]})
    assert fused["X"] == pytest.approx(1 / 61 + 1 / 62) == pytest.approx(fused["Y"])


# ---------- 키워드 의미 확장 (인터뷰 주력 분야·유사 표현) ----------
QS = [
    {"id": "focus", "type": "single", "options": [
        {"value": "all", "label": "전체"},
        {"value": "f0", "label": "교육", "keywords": ["교육", "연수"], "query": "교원 연수 운영 용역"}]},
    {"id": "expand", "type": "multi", "options": [
        {"value": "워크숍", "label": "워크숍", "group": "연수"}, {"value": "인공지능", "label": "인공지능", "group": "AI"}]},
    *interview.QUESTIONS,
]


def test_multi_answers_and_focus():
    p = to_prefs({"focus": "f0", "expand": ["워크숍", "없는말", "워크숍"], "priority": "S2"}, BASE, QS)
    assert p["answers"] == {"focus": "f0", "expand": ["워크숍"], "priority": "S2"}
    assert p["focus"] == {"label": "교육", "keywords": ["교육", "연수"], "query": "교원 연수 운영 용역"}
    assert p["expanded"] == [{"term": "워크숍", "group": "연수"}]
    assert to_prefs({"expand": []}, BASE, QS)["answers"] == {}  # 빈 목록 = 건너뜀


def test_expand_company_and_semantic_keywords():
    p = to_prefs({"focus": "f0", "expand": ["워크숍", "인공지능"]}, BASE, QS)
    assert interview.expand_company(COMPANY, p)["역량키워드"] == ["교육", "AI", "연수", "워크숍", "인공지능"]
    assert COMPANY["역량키워드"] == ["교육", "AI", "연수"]  # 원본 프로필은 그대로
    assert interview.semantic_keywords(COMPANY, p) == ["교육", "연수", "워크숍"]  # 주력 분야 키워드 + 그 확장어만
    assert interview.semantic_keywords(COMPANY, None) == ["교육", "AI", "연수"]


def test_expanded_term_counts_in_s2():
    t = tender(bidNtceNm="교원 워크숍 운영")
    p = to_prefs({"expand": ["워크숍"]}, BASE, QS)
    before = soft_score(t, COMPANY)["keyword_hits"]
    top = recommend_by_rules([t], interview.expand_company(COMPANY, p), {}, prefs=p)["top"][0]
    s2 = top["score_detail"]["items"][1]
    assert "워크숍" not in before and "워크숍(확장어)" in s2["reason"] and s2["interview"]


def test_semantic_s2_takes_max():
    t = tender(bidNtceNm="교직원 역량강화 과정")  # 문자열 일치 0개
    assert soft_score(t, COMPANY)["S2"] == 0
    sem = {"keyword": "연수", "cos": 0.6, "score": 0.75}
    s = soft_score(t, COMPANY, BASE, sem)
    assert s["S2"] == 0.75
    reason = score_items(t, COMPANY, s, BASE)[1]["reason"]
    assert "가까운 키워드 '연수'" in reason and "0.75" in reason


def test_keyword_similarity(monkeypatch):
    import backend.tools.retrieval as retrieval

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "keyword")
    assert search.keyword_similarity(["행사"], DOCS)[0] is None
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bge")
    monkeypatch.setattr(retrieval, "_bge", lambda: _FakeModel())
    corpus = ["도로 1", "도로 2", "도로 3", "행사"]  # 백분위 기준 분포
    sims, reason = search.keyword_similarity(["행사"], DOCS, corpus)
    assert reason is None and sims["A"]["score"] == 1.0 and sims["B"]["score"] == 0.0


def test_suggest_cleaning_and_fallback(monkeypatch):
    raw = {"focus": [{"label": "교육", "keywords": ["교육", "없는키워드"], "query": "q"},
                     {"label": "빈분야", "keywords": ["없음"], "query": "q"}],
           "expansions": [{"keyword": "교육", "similar": ["연수", "강좌", "용역", "아주아주아주긴표현입니다"]},
                          {"keyword": "모름", "similar": ["x1"]}]}
    s = interview._clean_suggest(raw, COMPANY)
    assert s["focus"] == [{"label": "교육", "keywords": ["교육"], "query": "q"}]
    assert s["expansions"] == [{"keyword": "교육", "similar": ["강좌"]}]  # 업체 키워드·넓은 단어·긴 표현 제외

    monkeypatch.setattr(config, "llm_enabled", lambda: False)
    interview._SUGGEST_CACHE.clear()
    qs = interview.build_questions(COMPANY)
    assert [q["id"] for q in qs][:1] == ["focus"] and "expand" not in [q["id"] for q in qs]  # LLM 없으면 실적 기반 분야만
    assert qs[0]["options"][1]["keywords"] == ["AI", "연수"]
