"""하드 필터 규칙 단위 테스트."""
from app.tools.evidence import check_reason
from app.tools.rules import hard_filter, soft_score

COMPANY = {
    "업체ID": "C999", "중소기업여부": True, "소재지": "서울",
    "업종": {"대분류": ["교육 및 전문직종/기술서비스"], "세부분류번호": ["86101696"]},
    "참여가능금액": {"최소": 30_000_000, "최대": 300_000_000},
    "역량키워드": ["교육", "AI", "연수"], "선호계약방식": ["제한경쟁"],
    "과거실적": [{"프로젝트명": "교원 AI 연수", "계약금액": 60_000_000, "키워드": ["AI", "연수"]}],
}


def tender(**kw):
    base = {"bidNtceNo": "T1", "bidNtceNm": "교원 AI 연수 운영", "pdfs": [{"name": "공고문.pdf", "url": "u"}],
            "pubPrcrmntLrgClsfcNm": "교육 및 전문직종/기술서비스", "pubPrcrmntClsfcNo": "86101696",
            "bidPrtcptLmtYn": "N", "rgnLmtBidLocplcJdgmBssNm": "", "budget": 90_000_000, "cntrctCnclsMthdNm": "제한경쟁"}
    return {**base, **kw}


def test_pass():
    assert hard_filter(tender(), COMPANY, {}) == []


def test_h1_no_pdf():
    assert hard_filter(tender(pdfs=[]), COMPANY, {}) == ["H1"]


def test_h3_industry():
    assert hard_filter(tender(pubPrcrmntLrgClsfcNm="ICT 서비스", pubPrcrmntClsfcNo="1"), COMPANY, {}) == ["H3"]
    # 대분류가 달라도 세부분류가 같으면 통과
    assert hard_filter(tender(pubPrcrmntLrgClsfcNm="ICT 서비스"), COMPANY, {}) == []


def test_h4_sme_only():
    assert hard_filter(tender(bidPrtcptLmtYn="Y"), COMPANY, {}) == []
    assert hard_filter(tender(bidPrtcptLmtYn="Y"), {**COMPANY, "중소기업여부": False}, {}) == ["H4"]


def test_h5_region():
    t = tender(rgnLmtBidLocplcJdgmBssNm="본사소재지")
    assert hard_filter(t, COMPANY, {}) == ["H5"]  # 매핑표에 없으면 보수적으로 제외
    assert hard_filter(t, COMPANY, {"T1": ["서울", "경기"]}) == []
    assert hard_filter(t, COMPANY, {"T1": ["부산"]}) == ["H5"]


def test_h6_budget():
    assert hard_filter(tender(budget=10_000_000), COMPANY, {}) == ["H6"]
    assert hard_filter(tender(budget=None), COMPANY, {}) == []  # 예산 없음 → 통과 + 확인 필요


def test_soft_score_full_match():
    s = soft_score(tender(bidNtceNm="교원 AI 연수 교육 운영"), COMPANY)  # 키워드 3개 → S2 = 1
    assert s["S1"] == 1.0 and s["S4"] == 1.0 and s["S2"] == 1.0
    assert 0 <= s["total"] <= 100


def test_evidence_check():
    t, chunks = tender(), [{"file": "공고문.pdf", "page": 1, "text": "교원 대상 AI 디지털 역량 강화 연수 운영"}]
    assert check_reason({"source_type": "pdf", "file": "공고문.pdf", "page": 1, "quote": "AI 디지털 역량"}, t, COMPANY, chunks)
    assert not check_reason({"source_type": "pdf", "file": "공고문.pdf", "page": 2, "quote": "AI 디지털 역량"}, t, COMPANY, chunks)
    assert not check_reason({"source_type": "pdf", "file": "공고문.pdf", "page": 1, "quote": "없는 문장입니다"}, t, COMPANY, chunks)
    assert check_reason({"source_type": "field", "field": "tender.pubPrcrmntClsfcNo", "value": "86101696"}, t, COMPANY, [])
    assert not check_reason({"source_type": "field", "field": "tender.pubPrcrmntClsfcNo", "value": "123"}, t, COMPANY, [])
