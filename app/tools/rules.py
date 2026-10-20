"""하드 필터(H1, H3-H6)와 소프트 점수(S1-S4). LLM 미사용 → 항상 같은 결과."""
import math
import re

from app import config

RULES = {
    "H1": "첨부 PDF 없음",
    "H3": "업종 대분류·세부분류 불일치",
    "H4": "참가 제한 공고인데 업체가 중소기업 아님",
    "H5": "지역 제한 허용 지역 밖 (또는 매핑표에 없음)",
    "H6": "참여 가능 금액 범위 밖",
}
RULE_LABELS = {"H1": "H1 PDF 보유", "H3": "H3 업종", "H4": "H4 참가 제한", "H5": "H5 지역 제한", "H6": "H6 예산"}


def hard_filter(tender: dict, company: dict, region_limits: dict[str, list[str]]) -> list[str]:
    """걸린 규칙 ID 목록. 빈 목록이면 통과. (H2 마감일은 테스트 데이터라 미적용)"""
    failed = []
    if not tender["pdfs"]:
        failed.append("H1")

    biz = company.get("업종", {})
    if tender.get("pubPrcrmntLrgClsfcNm") not in biz.get("대분류", []) and \
            tender.get("pubPrcrmntClsfcNo") not in biz.get("세부분류번호", []):
        failed.append("H3")

    # bidPrtcptLmtYn(입찰참가제한)을 중소기업 제한으로 해석. indstrytyLmtYn은 업종 제한이라 H3에서 다룸
    if tender.get("bidPrtcptLmtYn") == "Y" and not company.get("중소기업여부"):
        failed.append("H4")

    if tender.get("rgnLmtBidLocplcJdgmBssNm"):
        if company.get("소재지") not in region_limits.get(tender["bidNtceNo"], []):
            failed.append("H5")

    budget, rng = tender.get("budget"), company.get("참여가능금액", {})
    if budget is not None and not (rng.get("최소", 0) <= budget <= rng.get("최대", math.inf)):
        failed.append("H6")
    return failed


def _keywords(company: dict) -> set[str]:
    kws = set(company.get("역량키워드", []))
    for h in company.get("과거실적", []):
        kws.update(h.get("키워드", []))
    return kws


def _tender_text(tender: dict) -> str:
    return " ".join(str(tender.get(k, "")) for k in ("bidNtceNm", "purchsObjPrdctList", "pubPrcrmntClsfcNm", "pubPrcrmntMidClsfcNm"))


def soft_score(tender: dict, company: dict, weights=None) -> dict:
    """0-1 항목별 점수 + 0-100 총점."""
    w1, w2, w3, w4 = weights or config.SCORE_WEIGHTS
    biz = company.get("업종", {})

    s1 = 1.0 if tender.get("pubPrcrmntClsfcNo") in biz.get("세부분류번호", []) else \
        0.3 if tender.get("pubPrcrmntLrgClsfcNm") in biz.get("대분류", []) else 0.0

    text = _tender_text(tender)
    hits = sorted(k for k in _keywords(company) if k and k in text)
    s2 = min(1.0, len(hits) / 3)

    amounts = [h["계약금액"] for h in company.get("과거실적", []) if h.get("계약금액")]
    budget = tender.get("budget")
    if budget and amounts:
        avg = sum(amounts) / len(amounts)
        s3 = max(0.0, 1 - abs(math.log(budget / avg)) / math.log(4))  # 평균 실적 대비 4배 차이면 0
    else:
        s3 = 0.5  # 예산 정보 없음 → 중립

    s4 = 1.0 if tender.get("cntrctCnclsMthdNm") in company.get("선호계약방식", []) else 0.0

    total = (s1 * w1 + s2 * w2 + s3 * w3 + s4 * w4) * 100
    return {"S1": round(s1, 3), "S2": round(s2, 3), "S3": round(s3, 3), "S4": round(s4, 3),
            "total": round(total, 1), "keyword_hits": hits}


def won(n: int | None) -> str:
    return "확인 필요" if n is None else f"{n / 10000:,.0f}만 원"


def rule_reasons(tender: dict, company: dict, score: dict) -> list[dict]:
    """룰 근거 — 모두 실제 필드 값을 출처로 단다."""
    reasons = []
    if score["S1"] == 1.0:
        reasons.append({"text": "공고 세부분류가 업체 업종 세부분류와 일치", "source_type": "field",
                        "field": "tender.pubPrcrmntClsfcNo", "value": tender["pubPrcrmntClsfcNo"]})
    elif score["S1"] > 0:
        reasons.append({"text": "공고 대분류가 업체 업종과 일치 (세부분류는 다름)", "source_type": "field",
                        "field": "tender.pubPrcrmntLrgClsfcNm", "value": tender["pubPrcrmntLrgClsfcNm"]})
    for field, label in (("bidNtceNm", "공고명"), ("purchsObjPrdctList", "구매대상 품목")):
        hits = [k for k in score["keyword_hits"] if k in str(tender.get(field, ""))]
        if hits:
            reasons.append({"text": f"{label}에 업체 역량 키워드 {len(hits)}개 포함", "source_type": "field",
                            "field": f"tender.{field}", "value": ", ".join(hits)})
            break
    if tender.get("budget"):
        reasons.append({"text": f"예산 {won(tender['budget'])} (실적 규모 유사도 {score['S3']:.2f})", "source_type": "field",
                        "field": "tender.asignBdgtAmt", "value": str(tender["budget"])})
    else:
        reasons.append({"text": "예산 정보 없음 — 확인 필요", "source_type": "field", "field": "tender.asignBdgtAmt", "value": ""})
    if score["S4"]:
        reasons.append({"text": "업체 선호 계약방식과 일치", "source_type": "field",
                        "field": "tender.cntrctCnclsMthdNm", "value": tender["cntrctCnclsMthdNm"]})
    return reasons


def recommend_by_rules(tenders: list[dict], company: dict, region_limits: dict, top_k: int | None = None) -> dict:
    """500건 → 하드 필터 → 소프트 점수 상위 K건. 화면 퍼널 숫자와 제외 사유 집계도 함께 돌려준다."""
    top_k = top_k or config.TOP_K
    passed, excluded = [], []
    for t in tenders:
        failed = hard_filter(t, company, region_limits)
        if failed:
            excluded.append({"bidNtceNo": t["bidNtceNo"], "bidNtceNm": t.get("bidNtceNm", ""), "rules": failed})
        else:
            passed.append(t)

    scored = [(t, soft_score(t, company)) for t in passed]
    scored.sort(key=lambda x: (-x[1]["total"], x[0]["bidNtceNo"]))  # 동점은 공고번호 순

    top = [{
        "rank": i + 1, "bidNtceNo": t["bidNtceNo"], "bidNtceNm": t.get("bidNtceNm", ""),
        "budget": t.get("budget"), "soft_score": s["total"], "scores": {k: s[k] for k in ("S1", "S2", "S3", "S4")},
        "reasons": rule_reasons(t, company, s), "source": "rule",
    } for i, (t, s) in enumerate(scored[:top_k])]

    # 퍼널·집계: 규칙마다 걸린 공고 수 (한 공고가 여러 규칙에 걸리면 각각 집계)
    rule_counts = {r: sum(r in e["rules"] for e in excluded) for r in RULES}
    return {
        "funnel": {"total": len(tenders), "pdf": sum(1 for t in tenders if t["pdfs"]),
                   "hard_pass": len(passed), "top": len(top)},
        "rule_counts": rule_counts,
        "excluded": excluded,
        "top": top,
    }


_SPLIT = re.compile(r"[\s,·/()\[\]^]+")


def tokenize(text: str) -> set[str]:
    return {w for w in _SPLIT.split(text) if len(w) >= 2}
