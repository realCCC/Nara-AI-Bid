"""담당자 확인 체크리스트: 공고 필드 항목, AI 항목 원문 대조, 폴백, 이력 저장."""
from backend import config
from backend.db import repository as repo
from backend.tools import checklist, llm, pdf, retrieval
from tests.test_rules import tender

CHUNKS = [{"file": "공고문.pdf", "page": 3, "text": "입찰참가자격: 행사대행 직접생산확인증명서를 보유한 업체"},
          {"file": "공고문.pdf", "page": 5, "text": "제출서류: 제안서 10부, 가격입찰서 1부"}]


def T(**kw):
    return tender(bidNtceNo="T-CHK", bidClseDt="2026-05-15 10:00:00", rgnLmtBidLocplcJdgmBssNm="", **kw)


def test_field_items_from_tender():
    items = checklist.field_items(T(bidPrtcptLmtYn="Y"))
    assert [i["category"] for i in items] == ["참가자격", "참가자격", "일정"]  # 계약방식·중소기업 제한·마감 (지역 제한 없음)
    assert "2026-05-15 10:00" in items[-1]["text"]
    assert items[0]["evidence"] == {"source_type": "field", "field": "tender.cntrctCnclsMthdNm", "value": "제한경쟁"}


def test_build_fallback_without_llm():
    rec = {"source": "rag", "pdf_evidence": False, "llm": {"qualification_flags": ["지역 제한 해석 필요"]}}
    cl = checklist.build(T(), rec)
    assert cl["source"] == "fallback"
    assert [i["id"] for i in cl["items"]] == [f"c{n}" for n in range(len(cl["items"]))]
    origins = [i["origin"] for i in cl["items"]]
    assert origins.count("common") == 2 and origins[-2:] == ["flag", "flag"]  # AI 확인 사항은 맨 뒤
    assert "지역 제한 해석 필요" in cl["items"][-2]["text"]


def test_ai_items_keep_only_quoted(monkeypatch):
    checklist._CACHE.clear()
    calls = []
    raw = {"items": [
        {"category": "참가자격", "text": "직접생산확인증명서 보유", "file": "공고문.pdf", "page": 3, "quote": "직접생산확인증명서를 보유한 업체"},
        {"category": "실적요건", "text": "지어낸 실적 요건", "file": "공고문.pdf", "page": 3, "quote": "최근 3년 실적 5억 이상"},
        {"category": "제출서류", "text": "제안서 10부", "file": "공고문.pdf", "page": 4, "quote": "제안서 10부"},  # 페이지가 틀림
    ]}
    monkeypatch.setattr(config, "llm_enabled", lambda: True)
    monkeypatch.setattr(pdf, "tender_chunks", lambda t: CHUNKS)
    monkeypatch.setattr(retrieval, "retrieve", lambda chunks, q, k=3, tender_no=None: chunks)
    monkeypatch.setattr(llm, "extract_checklist", lambda t, c: calls.append(1) or raw)
    cl = checklist.build(T())
    ai = [i for i in cl["items"] if i["origin"] == "ai"]
    assert cl["source"] == "llm" and [i["text"] for i in ai] == ["직접생산확인증명서 보유"]  # 원문에 없는 인용은 버림
    commons = [i["category"] for i in cl["items"] if i["origin"] == "common"]
    assert commons == ["제출서류"]  # AI가 참가자격을 뽑았으니 참가자격 공통 문구는 빠짐
    checklist.build(T())
    assert len(calls) == 1  # 같은 공고는 캐시
    checklist._CACHE.clear()


def test_save_checklist_and_history(tmp_path):
    db = tmp_path / "t.db"
    repo.init_db(db)
    repo.save_run("th1", "C001", "selected", selection="T-CHK", db_path=db)
    assert repo.list_runs("C001", db)[0]["confirmed_at"] is None
    repo.save_checklist("th1", "T-CHK", checklist.build(T()), db)
    assert repo.list_runs("C001", db)[0]["confirmed_at"]
    saved = repo.get_run("th1", db)["checklist"]
    assert saved["source"] == "fallback" and saved["items"][0]["id"] == "c0"
