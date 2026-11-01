"""SC-002 초안: 원문에 있는 세부항목·과업만 남기고, 공고 필드로 개요·일정을 채운다."""
from backend.tools import proposal

CHUNKS = [{"file": "공고서.pdf", "page": 3, "text": "과업 내용: 행사 기획 및 운영 인력 배치. 세부 평가항목 사업 이해도 추진 전략"}]
TENDER = {"bidNtceNo": "T1", "bidNtceNm": "지역 축제 운영 용역", "dminsttNm": "OO시", "asignBdgtAmt": "100000000",
          "bidClseDt": "2026-05-06 14:00:00"}
COMPANY = {"회사명": "A사", "소재지": "서울", "중소기업여부": True, "역량키워드": ["축제"], "보유인력": {"PM": 2, "기획자": 3},
           "과거실적": [{"프로젝트명": "도서관 리모델링", "계약금액": 1}, {"프로젝트명": "봄 축제 운영", "계약금액": 2, "키워드": ["축제"]}]}


def test_normalize_keeps_only_quoted_tasks_and_sub_items():
    crit = proposal.normalize_criteria({
        "items": [{"item": "기술 능력", "score": 60, "sub_items": ["사업 이해도", "지어낸 항목"], "file": "공고서.pdf", "page": 3}],
        "period": "계약일로부터 3개월",
        "tasks": [{"text": "운영 인력 배치", "file": "공고서.pdf", "page": 3, "quote": "행사 기획 및 운영 인력 배치"},
                  {"text": "지어낸 과업", "file": "공고서.pdf", "page": 3, "quote": "원문에 없는 문장"}]}, TENDER, CHUNKS)
    assert crit["items"][0]["sub_items"] == ["사업 이해도"]
    assert [t["text"] for t in crit["tasks"]] == ["운영 인력 배치"]


def test_assemble_fills_overview_schedule_and_related_history():
    crit = proposal.fallback_criteria(TENDER)
    html = proposal.assemble_proposal(COMPANY, TENDER, crit, rec={"reasons": [
        {"text": "역량 키워드 일치", "source_type": "field", "field": "tender.bidNtceNm"}]})
    assert "OO시" in html and "10,000만 원" in html
    assert "입찰서 제출 마감" in html and "2026-05-06 14:00" in html
    assert "총 5명" in html and "역량 키워드 일치" in html
    assert html.index("[유사 실적]") < html.index("도서관 리모델링")


# ---------- LLM 섹션 초안 (업체 문서 근거) ----------
from backend import config  # noqa: E402
from backend.tools import company_docs, llm, proposal_writer  # noqa: E402

DOC = """# A사 회사소개서

1. 보유 인증 및 면허
- 국제회의기획업(PCO) 등록 (서울특별시, 2014년)
- ISO 20121 이벤트 지속가능경영시스템 인증 (2021년)

2. 핵심 인력
- 총괄 PM 김도윤 이사: 행사 기획 경력 15년, 지역 문화축제 총괄 10건 이상 수행.
"""


def _docs(tmp_path, monkeypatch):
    d = tmp_path / "C9"
    d.mkdir()
    (d / "소개서.md").write_text(DOC, encoding="utf-8")
    monkeypatch.setattr(config, "COMPANY_DOCS_DIR", tmp_path)
    return {**COMPANY, "업체ID": "C9"}


def test_company_docs_chunks_markdown_with_sections(tmp_path, monkeypatch):
    chunks = company_docs.load_chunks(_docs(tmp_path, monkeypatch))
    assert chunks and all(c["file"] == "소개서.md" for c in chunks)
    assert any("PCO" in c["text"] for c in chunks)
    assert company_docs.load_chunks({**COMPANY, "업체ID": "없음"}) == []
    hits = company_docs.search(chunks, "총괄 PM 경력", k=1)
    assert "김도윤" in hits[0]["text"]


def test_fill_grounded_only_when_every_quote_is_in_docs():
    chunks = [{"file": "소개서.md", "text": "총괄 PM 김도윤 이사: 행사 기획 경력 15년"}]
    ok = proposal_writer.fill({"text": "PM 김도윤", "cites": [{"file": "소개서.md", "quote": "총괄 PM 김도윤 이사"}]}, chunks)
    assert ok == {"text": "PM 김도윤", "grounded": True, "sources": ["업체 문서 · 소개서.md"]}
    bad = {"text": "PM 홍길동", "cites": [{"file": "소개서.md", "quote": "총괄 PM 홍길동"}]}
    assert proposal_writer.fill(bad, chunks)["grounded"] is False
    assert proposal_writer.fact(bad, chunks) is None                       # 사실형 칸은 근거 없으면 [확인 필요]
    assert proposal_writer.fill({"text": "일반 전략", "cites": []}, chunks)["grounded"] is False
    assert proposal_writer.fill({"text": " "}, chunks) is None


def test_write_fills_sections_and_validates(tmp_path, monkeypatch):
    company = _docs(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "llm_enabled", lambda: True)
    monkeypatch.setattr(config, "embedding_enabled", lambda: False)
    cite = [{"file": "소개서.md", "quote": "국제회의기획업(PCO) 등록"}]
    monkeypatch.setattr(llm, "draft_understanding", lambda *a: {"strategy": {"text": "축제 운영 전략", "cites": []}})
    monkeypatch.setattr(llm, "draft_company", lambda *a: {"certifications": {"text": "PCO 등록", "cites": cite},
                                                          "vision": {"text": "지어낸 비전", "cites": [{"file": "소개서.md", "quote": "없는 문장"}]}})
    monkeypatch.setattr(llm, "draft_plan", lambda *a: {
        "phases": [{"step": "착수", "content": "착수 보고", "period": "1주"}, {"step": "엉뚱", "content": "x"}],
        "staffing": [{"role": "PM", "count": 5, "duty": "총괄"}, {"role": "기획자", "count": 2, "duty": "기획"}, {"role": "없는역할", "count": 1, "duty": "-"}],
        "pm": {"text": "PM 홍길동", "cites": []}})
    monkeypatch.setattr(llm, "draft_item", lambda item, *a: {
        "plan": {"text": f"{item['item']} 방안", "cites": []}, "evidence": None,
        "sub_items": [{"name": "사업 이해도", "text": {"text": "이해도 대응", "cites": []}}, {"name": "지어낸", "text": {"text": "x"}}]})
    crit = {"items": [{"item": "기술 능력", "score": 60, "sub_items": ["사업 이해도"]}, {"item": "가격 평가", "score": 10}],
            "period": None, "tasks": []}
    ai = proposal_writer.write(company, TENDER, crit)
    assert ai["understanding"]["grounded"] is False
    assert ai["certifications"]["grounded"] is True and ai["certifications"]["sources"] == ["업체 문서 · 소개서.md"]
    assert ai["vision"]["grounded"] is False                              # 인용이 원문에 없으면 AI 초안
    assert ai["pm"] is None                                               # 근거 없는 사람 이름은 버림
    assert list(ai["phases"]) == ["착수"]
    assert ai["staffing"] == {"PM": {"count": None, "duty": "총괄"}, "기획자": {"count": 2, "duty": "기획"}}  # 보유 2명 < 5명
    assert list(ai["items"]) == ["기술 능력"]                              # 가격 항목은 호출 안 함
    assert list(ai["items"]["기술 능력"]["sub_items"]) == ["사업 이해도"]
    assert ai["doc_search"] == "bm25"

    html = proposal.assemble_proposal(company, TENDER, crit, ai=ai)
    assert "축제 운영 전략" in html and "[AI 초안·검토 필요]" in html
    assert "PCO 등록" in html and "업체 문서 · 소개서.md" in html
    assert "기술 능력 방안" in html and "이해도 대응" in html and "착수 보고" in html


def test_write_returns_none_without_llm():
    assert proposal_writer.write(COMPANY, TENDER, proposal.fallback_criteria(TENDER)) is None
    html = proposal.assemble_proposal(COMPANY, TENDER, proposal.fallback_criteria(TENDER))
    assert "[확인 필요]" in html and "[AI 초안" not in html


def test_find_period_from_chunks_when_llm_misses():
    chunks = [{"file": "a.pdf", "page": 1, "text": "제안서 제출기간 2026. 5. 18. ~ 2026. 5. 19."},
              {"file": "a.pdf", "page": 2, "text": "사 업 명 『연수』운영 사업기간 2026. 착수일 ~ 2026.10.31. 운영 내용 1. 입문"}]
    assert proposal.find_period(chunks) == "2026. 착수일 ~ 2026.10.31. (PDF · a.pdf p.2)"
    assert proposal.find_period([{"file": "b.pdf", "page": 3, "text": "계약기간: 계약일로부터 3개월간 수행"}]).startswith("계약일로부터 3개월간")
    assert proposal.find_period([{"file": "c.pdf", "page": 1, "text": "기간 정보 없음"}]) is None
    crit = proposal.normalize_criteria({"items": [{"item": "기술", "score": 80}], "period": None, "tasks": []}, TENDER, chunks)
    assert "2026.10.31." in crit["period"]


# ---------- 대화형 보완 ----------
from backend.tools import proposal_chat  # noqa: E402


def _ctx():
    crit = {"items": [{"item": "기술 능력", "score": 60, "sub_items": ["사업 이해도"], "source": "PDF"}], "period": None, "tasks": []}
    return {"company": COMPANY, "tender": TENDER, "criteria": crit, "rec": {}, "checks": [], "ai": {}}


def test_chat_slots_cover_empty_fields():
    keys = {s["key"]: s for s in proposal_chat.slots(_ctx())}
    assert {"understanding", "certifications", "pm", "period", "items/0/plan", "items/0/sub/0",
            "phases/착수/period", "staffing/PM/count"} <= set(keys)
    assert keys["pm"]["current"] is None and keys["staffing/PM/count"]["kind"] == "count"


def test_chat_validate_and_apply():
    ctx = _ctx()
    tender_chunks = [{"file": "공고서.pdf", "page": 2, "text": "사업기간 2026. 착수일 ~ 2026.10.31."}]
    raw = {"reply": "변경안입니다", "edits": [
        {"key": "pm", "text": "홍길동 부장 (경력 12년)", "source": "user"},
        {"key": "certifications", "text": "ISO 9001", "source": "general"},            # 사실형 + 근거 없음 → 제외
        {"key": "period", "text": "2026. 착수일 ~ 2026.10.31.", "source": "tender",
         "cites": [{"file": "공고서.pdf", "quote": "2026. 착수일 ~ 2026.10.31."}]},
        {"key": "items/0/plan", "text": "단계별 수행", "source": "general"},
        {"key": "items/0/sub/0", "text": "과업 이해", "source": "tender", "cites": [{"file": "공고서.pdf", "quote": "없는 문장"}]},
        {"key": "staffing/PM/count", "text": "5명", "source": "user"},                # 보유 2명 초과 → 제외
        {"key": "staffing/기획자/count", "text": "2", "source": "user"},
        {"key": "phases/착수/period", "text": "1주", "source": "tender"},              # 인용 없는 '공고서' 주장 → AI 표시
        {"key": "없는칸", "text": "x", "source": "user"}]}
    edits, skipped = proposal_chat.validate(raw, ctx, [], tender_chunks)
    by = {e["key"]: e for e in edits}
    assert set(by) == {"pm", "period", "items/0/plan", "items/0/sub/0", "staffing/기획자/count", "phases/착수/period"}
    assert by["period"]["grounded"] is True and by["period"]["source"] == "공고서 p.2"
    assert by["phases/착수/period"]["grounded"] is False
    assert {s["key"] for s in skipped} == {"certifications", "staffing/PM/count"}
    assert by["pm"]["source"] == "담당자 입력" and by["items/0/sub/0"]["grounded"] is False

    ai, crit = proposal_chat.apply(ctx, edits)
    assert crit["period"] == "2026. 착수일 ~ 2026.10.31." and ctx["criteria"]["period"] is None  # 원본은 그대로
    html = proposal.assemble_proposal(COMPANY, TENDER, crit, ai=ai)
    assert "홍길동 부장 (경력 12년)" in html and "(담당자 입력)" in html
    assert "단계별 수행" in html and "과업 이해" in html and 'data-slot="pm"' in html
    assert "2명" in html.split('data-slot="staffing/기획자/count"')[1][:30]


def test_chat_evidence_edit_overrides_rule_evidence():
    ctx = _ctx()
    ctx["criteria"]["items"][0]["item"] = "투입 인력"   # 룰 근거(보유인력)가 생기는 항목
    edits, _ = proposal_chat.validate({"edits": [{"key": "items/0/evidence", "text": "강사 5명 보유", "source": "user"}]}, ctx, [], [])
    ai, crit = proposal_chat.apply(ctx, edits)
    html = proposal.assemble_proposal(COMPANY, TENDER, crit, ai=ai)
    assert "강사 5명 보유" in html


def test_chat_steps_emit_real_progress(monkeypatch):
    ctx = _ctx()
    monkeypatch.setattr(config, "llm_enabled", lambda: True)
    monkeypatch.setattr(config, "embedding_enabled", lambda: False)
    monkeypatch.setattr("backend.tools.pdf.tender_chunks", lambda t: [{"file": "공고서.pdf", "page": 2, "section": "", "text": "사업기간 2026. 착수일 ~ 2026.10.31. 운영"}])
    monkeypatch.setattr(llm, "chat_edit", lambda *a: {"reply": "제안", "edits": [{"key": "pm", "text": "홍길동", "source": "user"}]})
    events = list(proposal_chat.chat_steps(ctx, "사업기간 PM 홍길동", []))
    steps = [(d["id"], d["status"]) for ev, d in events if ev == "step"]
    assert steps == [("prepare", "running"), ("prepare", "done"), ("search", "running"), ("search", "done"),
                     ("llm", "running"), ("llm", "done"), ("verify", "running"), ("verify", "done")]
    assert "공고서 · 공고서.pdf p.2" in next(d["detail"] for ev, d in events if ev == "step" and d["id"] == "search" and d["status"] == "done")
    assert events[-1][0] == "result" and events[-1][1]["edits"][0]["key"] == "pm"
    assert [d["id"] for ev, d in proposal_chat.chat_steps(ctx, "다시", []) if ev == "step" and d["status"] == "done"] == ["search", "llm", "verify"]
