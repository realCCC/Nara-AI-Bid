"""고도화: 가중치 계산, 인터뷰 → 가중치 변환, 키워드 의미 확장, 검증 실패 폴백, 임베딩 실패 폴백."""
import numpy as np
import pytest

from backend import config
from backend.agents.verify import apply_verification, confirm_issues
from backend.tools import llm, search
from backend.tools import interview, keyword_profile
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
    return {"order": order,
            "items": [{"bidNtceNo": i, "fit_score": 80, "comment": "c", "issues": []} for i in (items or order)]}


def test_check_verify_failures():
    assert llm.check_verify(None, IDS)[1].startswith("형식 오류")
    assert llm.check_verify({"order": IDS}, IDS)[1].startswith("형식 오류")
    assert llm.check_verify(_result(["T1", "T2", "T9"]), IDS)[1] == "없는 공고ID: T9"
    assert "누락" in llm.check_verify(_result(["T1", "T2"]), IDS)[1]
    assert llm.check_verify(_result(["T3", "T1", "T2"]), IDS) == (_result(["T3", "T1", "T2"]), None)


def test_apply_verification_keeps_order_on_failure():
    recs, v = apply_verification(RECS, None, "LLM 키 없음", 3)
    assert recs == RECS and v == {"status": "skipped", "reason": "LLM 키 없음", "checked": 3, "changed": False,
                                  "flagged": 0, "issues_dropped": 0}


def test_apply_verification_reorders_head_only():
    recs, v = apply_verification(RECS, _result(["T3", "T1", "T2"]), None, 3)
    assert [r["bidNtceNo"] for r in recs] == ["T3", "T1", "T2", "T4", "T5", "T6"]
    assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5, 6]
    assert recs[0]["verify"]["rank_before"] == 3 and "verify" not in recs[3]
    assert v["status"] == "passed" and v["changed"]


QUAL_CHUNKS = {"T1": [{"file": "공고.pdf", "page": 2, "text": "입찰참가자격: 공고일 현재 부산광역시에 본사를 둔 업체"}]}


def _issue(**kw):
    return {"type": "지역제한", "text": "부산 소재 업체만 참가 가능", "source_type": "pdf", "file": "공고.pdf", "page": 2,
            "quote": "부산광역시에 본사를 둔 업체", "company_field": "company.소재지", "company_value": "서울", **kw}


def test_confirm_issues_drops_unsourced():
    result = _result(["T1", "T2", "T3"])
    result["items"][0]["issues"] = [_issue(), _issue(quote="서울특별시에 본사를 둔 업체"), _issue(page=3),
                                    _issue(company_field="company.면허", company_value="확인 불가"),
                                    _issue(company_field="소재지", quote="부산광역시에 ... 둔 업체"),  # 접두어 생략·줄임표 인용 허용
                                    _issue(quote="부산광역시에 ... 서울 업체")]
    result["items"][1]["issues"] = [_issue(source_type="field", field="tender.rgnLmtBidLocplcJdgmBssNm", value="부산"),
                                    _issue(company_value="")]
    out, dropped = confirm_issues(result, {"T1": {}, "T2": {"rgnLmtBidLocplcJdgmBssNm": "부산광역시"}},
                                  {"소재지": "서울"}, QUAL_CHUNKS)
    assert [len(i["issues"]) for i in out["items"]] == [2, 1, 0] and dropped == 5


def test_conflicts_only_qualification_fields():
    from backend.agents.verify import _conflicts

    company = {"소재지": "강원", "중소기업여부": False, "선호계약방식": ["제한경쟁"], "업종": {"대분류": ["연구조사서비스"]}}
    assert _conflicts({"company_field": "company.소재지", "company_value": "강원"}, company)
    assert _conflicts({"company_field": "중소기업여부", "company_value": "False"}, company)
    assert _conflicts({"company_field": "company.업종", "company_value": "연구조사서비스"}, company)
    assert _conflicts({"company_field": "업종", "company_value": '{"대분류":["연구조사서비스"]}'}, company)  # 공백 없는 JSON
    assert not _conflicts({"company_field": "업종", "company_value": '{"대분류":["ICT 서비스"]}'}, company)
    assert not _conflicts({"company_field": "company.선호계약방식", "company_value": "제한경쟁"}, company)  # 선호는 자격 충돌 아님
    assert not _conflicts({"company_field": "company.소재지", "company_value": "부산"}, company)
    assert not _conflicts({"company_field": "중소기업여부", "company_value": "True"}, {"중소기업여부": True})  # '소기업 확인 불가' 오탐


def test_apply_verification_demotes_flagged():
    result = _result(["T1", "T2", "T3"])
    result["items"][0]["issues"] = [_issue()]
    recs, v = apply_verification(RECS, result, None, 3, dropped=1)
    assert [r["bidNtceNo"] for r in recs] == ["T2", "T3", "T1", "T4", "T5", "T6"]
    assert recs[2]["verify"]["issues"][0]["type"] == "지역제한"
    assert v["changed"] and v["flagged"] == 1 and v["issues_dropped"] == 1


def test_checklist_includes_verify_issues():
    from backend.tools import checklist

    made = checklist.build({"bidNtceNo": "X"}, {"verify": {"issues": [_issue()]}})
    item = next(i for i in made["items"] if i["origin"] == "verify")
    assert item["evidence"]["quote"] == "부산광역시에 본사를 둔 업체"


class _FakeLLM:
    def __init__(self, out):
        self.out = out

    def with_structured_output(self, schema):
        return self

    def with_config(self, **kw):
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


def test_body_ranking(monkeypatch):
    import backend.tools.retrieval as retrieval

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bge")
    monkeypatch.setattr(config, "BODY_SEARCH", True)
    monkeypatch.setattr(retrieval, "_bge", lambda: _FakeModel())
    monkeypatch.setattr(search, "body_search", lambda q, ids: [("B", 0.9), ("C", 0.5)])  # A는 청크 없음
    r = search.hybrid_search(DOCS, "행사 축제 운영")
    assert r["lists"] == ["bm25", "dense", "body"]
    assert r["scores"]["B"]["body"] == 0.9 and r["scores"]["A"]["body"] is None


def test_body_ranking_db_failure(monkeypatch):
    import backend.tools.retrieval as retrieval

    def no_db(q, ids):
        raise RuntimeError("pgvector 사용 불가")

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bge")
    monkeypatch.setattr(config, "BODY_SEARCH", True)
    monkeypatch.setattr(retrieval, "_bge", lambda: _FakeModel())
    monkeypatch.setattr(search, "body_search", no_db)
    r = search.hybrid_search(DOCS, "행사 축제 운영")
    assert r["path"] == "semantic" and r["lists"] == ["bm25", "dense"]


def test_rrf():
    fused = search.rrf({"a": ["X", "Y"], "b": ["Y", "X"]})
    assert fused["X"] == pytest.approx(1 / 61 + 1 / 62) == pytest.approx(fused["Y"])


# ---------- RAG 청크 하이브리드 검색 ----------
CHUNKS = [{"file": "a.pdf", "page": 1, "text": "입찰 참가자격 제한 사항 안내"},
          {"file": "a.pdf", "page": 2, "text": "축제 기획 및 행사 운영 과업"},
          {"file": "a.pdf", "page": 3, "text": "도로 포장 공사 시방서"}]


def test_chunk_search_bm25_only_without_embedding(monkeypatch):
    import backend.tools.retrieval as retrieval

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "keyword")
    out, path = retrieval.retrieve_for_company(CHUNKS, {"역량키워드": ["축제"]}, k=1, query="행사 운영", tender_no="T1")
    assert path == "bm25" and all(c["match"]["dense"] is None for c in out)
    assert "도로" not in " ".join(c["text"] for c in out)  # 키워드 겹침 없는 청크는 BM25만으로는 안 나온다


@pytest.fixture
def fake_store(monkeypatch):
    """pgvector 대신 메모리 dict (DB 없이 retrieval 로직만 시험). {(공고번호, 해시): 벡터}."""
    from backend.db import vector_store

    db = {}
    monkeypatch.setattr(vector_store, "load", lambda no, keys: {k: db[(no, k)] for k in keys if (no, k) in db})
    monkeypatch.setattr(vector_store, "save", lambda no, rows, keep: db.update({(no, k): v for k, _, v in rows}))
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bge")
    monkeypatch.setattr(config, "CHUNK_EMBEDDING", True)
    return db


def test_chunk_search_hybrid_caches_per_tender(monkeypatch, fake_store):
    import backend.tools.retrieval as retrieval

    calls = []

    class Model(_FakeModel):
        def encode(self, texts, **kw):
            calls.append(len(texts))
            return super().encode(texts, **kw)

    monkeypatch.setattr(retrieval, "_bge", lambda: Model())
    out, path = retrieval.retrieve_for_company(CHUNKS, {"역량키워드": []}, k=1, query="행사 운영", tender_no="T1")
    assert path == "hybrid" and len(fake_store) == 3
    assert all(c["match"]["dense"] is not None for c in out)
    retrieval.chunk_vectors("T1", CHUNKS)  # 두 번째는 DB에서 읽기만
    assert calls[0] == 3 and calls.count(3) == 1  # 청크 3개는 한 번만 임베딩 (나머지 호출은 질의 1개씩)


def test_retrieve_reuses_saved_chunk_vectors(monkeypatch, fake_store):
    """검증·체크리스트·평가항목의 retrieve(tender_no=...)는 RAG가 저장한 청크 벡터를 다시 쓴다."""
    import backend.tools.retrieval as retrieval
    from backend.tools import search

    calls = []

    class Model(_FakeModel):
        def encode(self, texts, **kw):
            calls.append(len(texts))
            return super().encode(texts, **kw)

    monkeypatch.setattr(retrieval, "_bge", lambda: Model())
    monkeypatch.setattr(search, "_VEC_CACHE", {})
    retrieval.chunk_vectors("T1", CHUNKS)  # RAG 단계가 저장
    calls.clear()
    out = retrieval.retrieve(CHUNKS, "입찰 참가자격", k=2, tender_no="T1")
    assert len(out) == 2 and calls == [1]  # 질의 1개만 임베딩, 청크는 DB에서 읽음
    calls.clear()
    retrieval.retrieve(CHUNKS, "입찰 참가자격 제출", k=2)  # tender_no 없으면 청크를 임베딩
    assert 3 in calls


def test_retrieve_falls_back_to_keywords_when_embedding_fails(monkeypatch, fake_store):
    import backend.tools.retrieval as retrieval

    def broken():
        raise RuntimeError("모델 로드 실패")

    monkeypatch.setattr(retrieval, "_bge", broken)
    out = retrieval.retrieve(CHUNKS, "입찰 참가자격", k=1, tender_no="T1")
    assert out and "참가자격" in out[0]["text"]


def test_pgvector_unavailable_uses_bm25_and_retries(monkeypatch):
    """DB에 연결할 수 없으면 청크 검색은 BM25만, 실패는 RETRY_SEC 동안 기억했다가 지나면 다시 연결을 시도한다."""
    import backend.tools.retrieval as retrieval
    from backend.db import vector_store

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bge")
    monkeypatch.setattr(config, "CHUNK_EMBEDDING", True)
    monkeypatch.setattr(config, "PGVECTOR_DSN", "postgresql://x:x@127.0.0.1:1/none")
    monkeypatch.setattr(vector_store, "_CONN", None)
    monkeypatch.setattr(vector_store, "_ERROR", None)
    monkeypatch.setattr(retrieval, "_bge", lambda: _FakeModel())
    out, path = retrieval.retrieve_for_company(CHUNKS, {"역량키워드": ["축제"]}, k=1, query="행사 운영", tender_no="T1")
    assert path == "bm25" and out and all(c["match"]["dense"] is None for c in out)
    failed_at = vector_store._ERROR[0]
    with pytest.raises(RuntimeError):
        vector_store._conn()  # 대기 시간 안: 연결을 다시 시도하지 않고 바로 실패
    assert vector_store._ERROR[0] == failed_at
    monkeypatch.setattr(vector_store, "_ERROR", (failed_at - vector_store.RETRY_SEC - 1, "옛 실패"))
    with pytest.raises(RuntimeError):
        vector_store._conn()  # 대기 시간이 지나면 다시 시도 (여기서는 또 실패)
    assert vector_store._ERROR[0] > failed_at


def _pg_or_skip(monkeypatch):
    """실제 pgvector DB (docker compose up -d)가 있을 때만. 테스트 전용 모델명으로 행을 분리하고 끝나면 지운다."""
    from backend.db import vector_store

    monkeypatch.setattr(config, "EMBEDDING_MODEL", "test/fake-1024")
    monkeypatch.setattr(vector_store, "_CONN", None)
    monkeypatch.setattr(vector_store, "_ERROR", None)
    try:
        conn = vector_store._conn()
    except RuntimeError as e:
        pytest.skip(str(e))
    conn.execute("DELETE FROM tender_chunks WHERE model = %s", (config.EMBEDDING_MODEL,))
    return vector_store, conn


class _FakeModel1024:
    def encode(self, texts, normalize_embeddings=True, **kw):
        out = np.zeros((len(texts), 1024), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 1 if "도로" in t else 0] = 1.0
        return out


def test_pgvector_stores_chunks_once_and_searches(monkeypatch):
    import backend.tools.retrieval as retrieval

    vector_store, conn = _pg_or_skip(monkeypatch)
    calls = []

    class Model(_FakeModel1024):
        def encode(self, texts, **kw):
            calls.append(len(texts))
            return super().encode(texts, **kw)

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bge")
    monkeypatch.setattr(config, "CHUNK_EMBEDDING", True)
    monkeypatch.setattr(retrieval, "_bge", lambda: Model())
    try:
        chunks = [{**c, "section": "도로" if "도로" in c["text"] else "일반"} for c in CHUNKS]
        retrieval.chunk_vectors("T1", chunks)
        vecs = retrieval.chunk_vectors("T1", chunks)  # 두 번째는 DB에서 읽기만
        assert calls == [3] and vecs.shape == (3, 1024)
        q = Model().encode(["도로 공사"])[0]
        hit = vector_store.search(q, k=1)
        assert hit[0]["bidNtceNo"] == "T1" and "도로" in hit[0]["text"] and hit[0]["score"] == pytest.approx(1.0)
        assert all(r["section"] == "일반" for r in vector_store.search(q, k=5, section="일반"))
        retrieval.chunk_vectors("T1", chunks[:2] + [{"file": "a.pdf", "page": 4, "text": "평가 방법 배점"}])
        n = conn.execute("SELECT count(*) FROM tender_chunks WHERE model = %s", (config.EMBEDDING_MODEL,)).fetchone()[0]
        assert n == 3  # 청킹이 바뀌면 옛 청크는 지우고 새 청크만 추가
    finally:
        conn.execute("DELETE FROM tender_chunks WHERE model = %s", (config.EMBEDDING_MODEL,))


# ---------- 키워드 의미 확장 (인터뷰 주력 분야·유사 표현) ----------
QS = [
    {"id": "focus", "type": "single", "options": [
        {"value": "all", "label": "전체", "effect": {}},
        {"value": "f0", "label": "교육", "effect": {"focus": {"label": "교육", "keywords": ["교육", "연수"],
                                                              "query": "교원 연수 운영 용역"}}}]},
    {"id": "expand", "type": "multi", "options": [
        {"value": "워크숍", "label": "워크숍", "effect": {"expand": [{"term": "워크숍", "group": "연수"}]}},
        {"value": "인공지능", "label": "인공지능", "effect": {"expand": [{"term": "인공지능", "group": "AI"}]}}]},
    *interview.QUESTIONS,
    interview.NOTE_QUESTION,
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


# ---------- AI 생성 인터뷰 · 자유 입력 ----------
def E(**kw):  # LLM Effect 형식 (빈 칸 기본값)
    return {"budget_min_won": None, "budget_max_won": None, "priority": None, "region": None, "min_prep_days": None,
            "focus_label": None, "focus_keywords": [], "focus_query": None, "expand_group": None, "expand_terms": [],
            "exclude_terms": [], **kw}


def test_clean_effect():
    eff = interview.clean_effect(E(budget_max_won=200_000_000, priority="S9", region="x", min_prep_days=99,
                                   focus_keywords=["교육", "없는키워드"], focus_query="q",
                                   expand_group="교육", expand_terms=["연수", "강좌", "용역", "아주아주아주긴표현입니다"],
                                   exclude_terms=["시설 공사", "x"]), COMPANY)
    assert eff == {"budget": [0, 200_000_000], "focus": {"label": "교육", "keywords": ["교육"], "query": "q"},
                   "expand": [{"term": "강좌", "group": "교육"}], "exclude": ["시설공사"]}  # 범위 밖 값·업체 키워드·넓은 단어 제외
    assert interview.clean_effect(E(expand_group="모름", expand_terms=["강좌"]), COMPANY) == {}  # 업체에 없는 키워드 그룹
    assert interview.clean_effect(E(budget_min_won=3, budget_max_won=2), COMPANY) == {}


def test_clean_questions_from_llm():
    raw = {"questions": [
        {"title": "이번 사업 규모", "desc": "d", "type": "single", "options": [
            {"label": "6천만 원 이하", "effect": E(budget_max_won=60_000_000, priority="S3")},  # 섞인 효과는 버림
            {"label": "효과 없음", "effect": E()},
            {"label": "6천만~3억", "effect": E(budget_min_won=60_000_000, budget_max_won=300_000_000)}]},
        {"title": "선택지 하나뿐", "desc": "", "type": "single", "options": [{"label": "a", "effect": E(priority="S1")}]},
        {"title": "유사 표현", "desc": "", "type": "multi", "options": [
            {"label": "강좌", "effect": E(expand_group="교육", expand_terms=["강좌"])}]},
    ]}
    qs = interview._clean_questions(raw, COMPANY)
    assert [(q["id"], q["type"], q["source"]) for q in qs] == [("q0", "single", "llm"), ("q1", "multi", "llm")]
    assert [o["value"] for o in qs[0]["options"]] == ["all", "o0", "o1"]  # '효과 없음' 버림, single은 '전체' 추가
    assert qs[1]["options"][0]["group"] == "교육" and "확장어 강좌" in qs[1]["options"][0]["hint"]
    assert qs[0]["options"][1]["effect"] == {"budget": [0, 60_000_000]}
    p = to_prefs({"q0": "o1", "q1": ["o0"]}, BASE, qs)
    assert p["budget"] == (60_000_000, 300_000_000) and p["expanded"] == [{"term": "강좌", "group": "교육"}]


def test_build_questions_llm_and_fallback(monkeypatch):
    good = {"questions": [{"title": f"문항{i}", "desc": "", "type": "single",
                           "options": [{"label": "a", "effect": E(priority="S1")}, {"label": "b", "effect": E(priority="S2")}]}
                          for i in range(3)]}
    monkeypatch.setattr(config, "llm_enabled", lambda: True)
    monkeypatch.setattr(llm, "generate_interview", lambda c: good)
    interview._QUESTION_CACHE.clear()
    qs = interview.build_questions(COMPANY)
    assert [q["id"] for q in qs] == ["q0", "q1", "q2", "note"] and qs[-1]["type"] == "text"

    monkeypatch.setattr(llm, "generate_interview", lambda c: None)  # LLM 실패 → 기본 문항
    interview._QUESTION_CACHE.clear()
    qs = interview.build_questions(COMPANY)
    assert [q["id"] for q in qs] == ["focus", "budget", "priority", "region", "schedule", "note"]
    assert qs[0]["options"][1]["effect"]["focus"]["keywords"] == ["AI", "연수"]  # 과거 실적 기반 분야
    interview._QUESTION_CACHE.clear()


def test_note_answer_and_effect_overrides(monkeypatch):
    note = "  시설 공사는 빼고 2억 이하로  "
    p = to_prefs({"budget": "gt300", "note": note}, BASE, QS,
                 note_effect={"budget": [0, 200_000_000], "exclude": ["시설공사"]})
    assert p["note"] == note.strip() and p["budget"] == (0, 200_000_000)  # 자유 입력이 선택 답보다 우선
    assert interview_filter(tender(bidNtceNm="청사 시설 공사"), COMPANY, p)
    assert not interview_filter(tender(budget=90_000_000), COMPANY, p)
    p = to_prefs({"focus": "f0", "note": note}, BASE, QS,
                 note_effect={"expand": [{"term": "강좌", "group": interview.NOTE_GROUP}]})
    assert interview.semantic_keywords(COMPANY, p) == ["교육", "연수", "강좌"]  # 직접 입력 키워드는 주력 분야와 무관하게
    assert interview.with_note(COMPANY, p)["추가요청"] == note.strip()
    assert to_prefs({"note": "   "}, BASE, QS)["note"] is None

    monkeypatch.setattr(config, "llm_enabled", lambda: True)
    monkeypatch.setattr(llm, "parse_note", lambda n, c: {**E(exclude_terms=["공사"]), "include_terms": ["강좌", "교육"]})
    assert interview.parse_note("x", COMPANY) == {  # 업체 키워드(교육)는 주력 분야로, 나머지는 확장어로
        "focus": {"label": "교육", "keywords": ["교육"], "query": "x"},
        "expand": [{"term": "강좌", "group": interview.NOTE_GROUP}], "exclude": ["공사"]}
    monkeypatch.setattr(config, "llm_enabled", lambda: False)
    assert interview.parse_note("x", COMPANY) == {}


# ---------- 키워드 의미 프로필 ----------
KP_RAW = {"keywords": [
    {"keyword": "교육", "description": "교원 대상 연수 과정 운영", "synonyms": ["강좌", "연수", "아카데미", "용역", "강 좌"],
     "sample_title": "2026 ○○교육청 교원 역량강화 과정 운영 용역"},
    {"keyword": "AI", "description": "", "synonyms": ["인공지능"], "sample_title": ""},
    {"keyword": "없는키워드", "description": "x", "synonyms": ["무엇"], "sample_title": ""},
]}


def test_keyword_profile_clean_and_fallback(monkeypatch):
    p = keyword_profile.clean(KP_RAW, COMPANY)
    assert [k["keyword"] for k in p["keywords"]] == ["교육", "AI"] and p["source"] == "llm"
    assert p["keywords"][0]["synonyms"] == ["강좌", "아카데미"]  # 업체 키워드·넓은 단어·중복 제외, 공백 제거
    assert keyword_profile.build(COMPANY) == {"source": "none", "keywords": []}  # LLM 없음 → 빈 프로필
    assert interview.expand_company(COMPANY, None) is COMPANY  # 기존 동작 그대로

    keyword_profile._CACHE.clear()
    monkeypatch.setattr(config, "llm_enabled", lambda: True)
    monkeypatch.setattr(llm, "keyword_profile", lambda c: KP_RAW)
    assert keyword_profile.synonyms(COMPANY) == {"교육": ["강좌", "아카데미"], "AI": ["인공지능"]}
    texts = keyword_profile.embed_texts(COMPANY)
    assert texts == {"교육": "교육: 교원 대상 연수 과정 운영 예) 2026 ○○교육청 교원 역량강화 과정 운영 용역", "AI": "AI"}


def test_synonyms_count_once_in_s2(monkeypatch):
    monkeypatch.setattr(config, "llm_enabled", lambda: True)
    monkeypatch.setattr(llm, "keyword_profile", lambda c: KP_RAW)
    eff = interview.expand_company(COMPANY, None)
    assert eff["유사어"] == {"교육": ["강좌", "아카데미"], "AI": ["인공지능"]}
    t = tender(bidNtceNm="인공지능 아카데미 강좌", pubPrcrmntClsfcNm="")
    assert soft_score(t, COMPANY)["S2"] == 0  # 유사어 없으면 0
    s = soft_score(t, eff)
    assert s["keyword_hits"] == ["강좌", "아카데미", "인공지능"] and s["keyword_matched"] == 2  # 교육·AI 2개
    assert s["S2"] == pytest.approx(2 / 3, abs=1e-3)
    reason = score_items(t, eff, s)[1]["reason"]
    assert "문자열 일치 2개" in reason and "강좌(←교육)" in reason


def test_keyword_similarity_uses_profile_texts(monkeypatch):
    seen = []
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bge")
    monkeypatch.setattr(search, "embed", lambda texts, persist=False: seen.append(list(texts)) or np.ones((len(texts), 2)))
    search.keyword_similarity(["교육", "AI"], {"A": "a"}, texts={"교육": "교육: 교원 연수"})
    assert seen[0] == ["교육: 교원 연수", "AI"]
