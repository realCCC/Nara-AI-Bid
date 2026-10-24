"""담당자 확인 체크리스트 (HITL ①): AI가 공고서에서 확인할 조건을 뽑고, 담당자가 원문과 대조해 체크한다.

항목 = 공고 필드에서 바로 나오는 조건(계약방식·지역 제한·중소기업 제한·마감) + LLM이 공고서 PDF에서 뽑은 조건
(인용이 해당 페이지 청크에 있어야 통과) + 추천 단계에서 AI가 표시한 자격 의심 사항 + 근거 부족 안내.
LLM이 없거나 못 뽑은 분류(참가자격·제출서류)는 공통 문구로 채운다.
"""
from backend import config
from backend.tools.evidence import check_reason

QUERY = "입찰 참가자격 자격요건 등록 면허 직접생산 실적 제출서류 제안서 제출 마감 일시 제안설명회 공동수급 하도급"
MAX_AI_ITEMS = 8
COMMON = {"참가자격": "공고서의 참가 자격 요건(등록·면허·실적)을 확인했습니다",
          "제출서류": "제출 서류 목록과 제출 방법을 확인했습니다"}
_CACHE: dict[str, list[dict]] = {}  # 공고번호 → 검증 통과한 AI 항목 (같은 공고를 다시 골라도 LLM을 또 부르지 않음)


def field_items(tender: dict) -> list[dict]:
    """공고 필드에서 바로 확인할 조건. 출처는 필드 값."""
    out = []

    def add(category, text, field):
        out.append({"category": category, "text": text, "origin": "field",
                    "evidence": {"source_type": "field", "field": f"tender.{field}", "value": str(tender[field])}})

    if tender.get("cntrctCnclsMthdNm"):
        add("참가자격", f"{tender['cntrctCnclsMthdNm']} 입찰입니다. 참가 자격 제한 내용을 확인했습니다", "cntrctCnclsMthdNm")
    if tender.get("rgnLmtBidLocplcJdgmBssNm"):
        add("참가자격", f"지역 제한 입찰({tender['rgnLmtBidLocplcJdgmBssNm']})입니다. 우리 소재지로 참여할 수 있는지 확인했습니다",
            "rgnLmtBidLocplcJdgmBssNm")
    if tender.get("bidPrtcptLmtYn") == "Y":
        add("참가자격", "중소기업 참여 제한 공고입니다. 우리 업체가 해당하는지 확인했습니다", "bidPrtcptLmtYn")
    if tender.get("bidClseDt"):
        add("일정", f"입찰 마감 {tender['bidClseDt'][:16]}까지 제출 준비가 가능한지 확인했습니다", "bidClseDt")
    return out


def ai_items(tender: dict) -> list[dict]:
    """LLM이 공고서 청크에서 뽑은 조건 중 인용이 원문에 있는 것만."""
    no = tender["bidNtceNo"]
    if no in _CACHE:
        return _CACHE[no]
    if not config.llm_enabled():
        return []
    from backend.tools import llm
    from backend.tools.pdf import tender_chunks
    from backend.tools.retrieval import retrieve

    chunks = tender_chunks(tender)
    raw = llm.extract_checklist(tender, retrieve(chunks, QUERY, k=8)) if chunks else None
    items = []
    for it in (raw or {}).get("items", [])[:MAX_AI_ITEMS]:
        ev = {"source_type": "pdf", "file": it.get("file"), "page": it.get("page"), "quote": it.get("quote")}
        if (it.get("text") or "").strip() and check_reason(ev, tender, {}, chunks):
            items.append({"category": it.get("category") or "기타", "text": it["text"].strip()[:120], "origin": "ai",
                          "evidence": ev})
    if raw is not None:  # LLM 실패(None)는 캐시하지 않아 다음 선택에서 다시 시도
        _CACHE[no] = items
    return items


def build(tender: dict, rec: dict | None = None) -> dict:
    rec = rec or {}
    ai = ai_items(tender)
    items = [*field_items(tender), *ai]
    covered = {i["category"] for i in ai}
    items += [{"category": c, "text": t, "origin": "common"} for c, t in COMMON.items() if c not in covered]
    flags = rec.get("qualification_flags") or (rec.get("llm") or {}).get("qualification_flags") or []
    items += [{"category": "AI 확인 사항", "text": f"추천 단계에서 AI가 표시한 사항을 검토했습니다: {f}", "origin": "flag"}
              for f in flags]
    if rec.get("source") == "rag" and not rec.get("pdf_evidence"):
        items.append({"category": "AI 확인 사항", "text": "공고서 PDF 근거가 부족한 공고임을 알고 진행합니다", "origin": "flag"})
    order = {"참가자격": 0, "실적요건": 1, "제출서류": 2, "일정": 3, "기타": 4, "AI 확인 사항": 5}
    items.sort(key=lambda i: order.get(i["category"], 4))
    return {"source": "llm" if ai else "fallback",
            "items": [{"id": f"c{n}", **i} for n, i in enumerate(items)]}
