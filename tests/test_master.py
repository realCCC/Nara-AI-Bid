"""핵심 사업: 후보 → 담당자 확인 → 검색 문장·순서·LLM 판단 프로필."""
from backend import config
from backend.graphs import recommend
from backend.tools import interview, llm, search
from backend.tools.interview import to_prefs

COMPANY = {"업체ID": "T1", "업종": {"대분류": ["교육"]}, "역량키워드": ["교육", "연수", "운영"],
           "과거실적": [{"프로젝트명": "공무원 AI 교육", "계약금액": 1, "키워드": ["AI교육"]}]}
DRAFT = {"source": "llm", "candidates": [{"key": "AI 교육", "reason": "실적", "prompt": "AI 교육 위탁 운영 용역"},
                                         {"key": "직무 연수", "reason": "역량", "prompt": "직무 연수 운영 용역"}]}


def test_fallback_candidates_without_llm():
    d = interview.build_masters(COMPANY, interview.QUESTIONS, to_prefs({}))
    assert d["source"] == "template"
    assert [c["key"] for c in d["candidates"]] == ["AI교육", "교육", "연수"]  # 넓은 말(운영)은 후보에서 뺀다
    assert all(c["key"] in c["prompt"] for c in d["candidates"])


def test_llm_candidates_are_cleaned_and_anchored(monkeypatch):
    monkeypatch.setattr(config, "llm_enabled", lambda: True)
    monkeypatch.setattr(llm, "master_candidates", lambda c, a: [
        {"key": " AI  교육 ", "reason": "r", "prompt": "인공지능 강의 용역"},
        {"key": "AI 교육", "reason": "중복", "prompt": "x"}, {"key": "", "reason": "", "prompt": "y"},
        {"key": "연수", "reason": "r", "prompt": "직무 연수 용역"}])
    d = interview.build_masters(COMPANY, interview.QUESTIONS, to_prefs({}))
    assert d["source"] == "llm" and [c["key"] for c in d["candidates"]] == ["AI 교육", "연수"]
    assert d["candidates"][0]["prompt"] == "AI 교육 인공지능 강의 용역"  # 프롬프트에 빠진 핵심 사업은 붙인다

    monkeypatch.setattr(llm, "master_candidates", lambda c, a: None)  # LLM 실패 → 역량 키워드 후보
    assert interview.build_masters(COMPANY, interview.QUESTIONS, to_prefs({}))["source"] == "template"


def test_resolve_master():
    first = interview.resolve_master(DRAFT, {"key": "", "prompt": ""}, COMPANY)  # 비우면 첫 후보
    assert first == {"key": "AI 교육", "prompt": "AI 교육 위탁 운영 용역", "source": "llm", "edited": False}
    edited = interview.resolve_master(DRAFT, {"key": "직무 연수", "prompt": "직무 연수 교육"}, COMPANY)
    assert edited["prompt"] == "직무 연수 교육" and edited["edited"]
    custom = interview.resolve_master(DRAFT, {"key": "드론 교육"}, COMPANY)  # 후보에 없는 직접 입력
    assert custom["source"] == "custom" and custom["prompt"].startswith("드론 교육")
    assert interview.resolve_master({"candidates": [], "source": "template"}, {}, COMPANY) is None


def test_master_drives_query_and_llm_profile():
    master = {"key": "AI 교육", "prompt": "AI 교육 위탁 운영 용역", "source": "llm", "edited": False}
    prefs = interview.with_master(to_prefs({"budget": "lt100"}), master, ["사업 규모: 1억 원 미만"])
    assert recommend.make_query(COMPANY, prefs) == ("AI 교육 위탁 운영 용역", "master")
    assert {"term": "AI교육", "group": interview.NOTE_GROUP} in prefs["expanded"]  # S2·청크 검색에도
    profile = interview.with_note(COMPANY, prefs)
    assert profile["핵심사업"] == "AI 교육" and profile["인터뷰답변"] == ["사업 규모: 1억 원 미만"]
    assert recommend.make_query(COMPANY, interview.with_master(to_prefs({}), None, []))[1] == "template"  # 키 없음 → 기존 경로


def test_master_first_moves_matches_up_without_excluding():
    docs = {"A": "시설 청소 용역", "B": "공무원 AI 활용 교육", "C": "AI교육 위탁", "D": "청소년 캠프"}
    r = search.master_first({"order": ["A", "B", "C", "D"]}, docs, "AI 교육")
    assert r["order"] == ["B", "C", "A", "D"]  # 일치끼리·나머지끼리 기존 순서 유지, 제외 없음
    assert r["master"] == {"key": "AI 교육", "hits": 2}


def test_master_match_ignores_generic_words():
    assert search.master_match("지역 축제 운영", "제21회 한마음 축제 기획")
    assert not search.master_match("지역 축제 운영", "시설 운영 지역 용역")  # 일반어만 맞으면 일치 아님
    assert not search.master_match("운영 지원", "시설 관리 용역")  # 일반어뿐인 키는 문자열 일치만


def test_rerank_keeps_master_first(monkeypatch):
    tenders = {n: {"bidNtceNo": n, "bidNtceNm": name} for n, name in (("A", "시설 청소"), ("B", "AI 교육"))}
    cands = [{"bidNtceNo": "A"}, {"bidNtceNo": "B"}]
    monkeypatch.setattr(recommend, "rerank", lambda s, d: {**s, "order": ["A", "B"], "rerank": "applied", "rerank_reason": None})
    found = {"order": ["B", "A"], "scores": {"A": {}, "B": {}}, "path": "semantic", "master": {"key": "AI 교육", "hits": 1}}
    out = recommend.rerank_stage(found, cands, tenders, top_k=2)
    assert [t["bidNtceNo"] for t in out["top"]] == ["B", "A"] and out["search"]["master"]["hits"] == 1
    j = out["top"][1]["journey"]
    assert {k: j[k] for k in ("rule", "search", "rerank")} == {"rule": None, "search": 2, "rerank": 2}  # 단계별 순위
    assert out["top"][0]["journey"]["why"]["master"] and not j["why"]["master"]  # 순위 변동 근거: 핵심 사업 일치
