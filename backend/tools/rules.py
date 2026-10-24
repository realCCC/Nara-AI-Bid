"""하드 필터(H1, H3-H6)와 소프트 점수(S1-S4). LLM 미사용 → 항상 같은 결과."""
import math
import re

from backend import config
from backend.tools.interview import interview_filter, region_bonus

RULES = {
    "H1": "첨부 PDF 없음",
    "H3": "업종 대분류·세부분류 불일치",
    "H4": "참가 제한 공고인데 업체가 중소기업 아님",
    "H5": "지역 제한 허용 지역 밖 (또는 매핑표에 없음)",
    "H6": "참여 가능 금액 범위 밖",
    "I1": "인터뷰 조건 밖 (사업규모·지역·일정)",
}
RULE_LABELS = {"H1": "H1 PDF 보유", "H3": "H3 업종", "H4": "H4 참가 제한", "H5": "H5 지역 제한", "H6": "H6 예산",
               "I1": "I1 인터뷰 조건"}
SCORE_ITEMS = {"S1": "업종 일치", "S2": "역량 키워드", "S3": "예산 규모", "S4": "계약방식"}


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


def soft_score(tender: dict, company: dict, weights=None, semantic: dict | None = None) -> dict:
    """0-1 항목별 점수 + 0-100 총점. semantic(키워드↔공고 의미 유사도)이 있으면 S2 = max(문자열 일치, 의미 점수)."""
    w1, w2, w3, w4 = weights or config.SCORE_WEIGHTS
    biz = company.get("업종", {})

    s1 = 1.0 if tender.get("pubPrcrmntClsfcNo") in biz.get("세부분류번호", []) else \
        0.3 if tender.get("pubPrcrmntLrgClsfcNm") in biz.get("대분류", []) else 0.0

    text = _tender_text(tender)
    hits = sorted(k for k in _keywords(company) if k and k in text)
    s2 = max(min(1.0, len(hits) / 3), semantic["score"] if semantic else 0.0)

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
            "total": round(total, 1), "keyword_hits": hits, "semantic": semantic}


def score_items(tender: dict, company: dict, score: dict, weights=None, base_weights=None,
                expanded: tuple = (), s2_interview: bool = False) -> list[dict]:
    """점수표 행: 항목 | 점수(0-1) | 가중치 | 반영 점수(0-100 기준) | 산출 이유.
    가중치가 기본값과 다르거나, S2에 인터뷰 확장어·주력 분야가 쓰였으면 interview=True."""
    weights = tuple(weights or config.SCORE_WEIGHTS)
    base_weights = tuple(base_weights or weights)
    biz = company.get("업종", {})
    s1 = "세부분류 일치" if score["S1"] == 1.0 else "대분류만 일치 (0.3)" if score["S1"] > 0 else "업종 불일치"
    amounts = [h["계약금액"] for h in company.get("과거실적", []) if h.get("계약금액")]
    if tender.get("budget") and amounts:
        s3 = f"예산 {won(tender['budget'])} / 평균 실적 {won(int(sum(amounts) / len(amounts)))} (4배 차이면 0)"
    else:
        s3 = "예산 또는 실적 정보 없음 → 중립 0.5"
    hits = score["keyword_hits"]
    reasons = {
        "S1": f"{s1} (공고 {tender.get('pubPrcrmntClsfcNo', '-')} · {tender.get('pubPrcrmntLrgClsfcNm', '-')})"
              if biz else s1,
        "S2": _s2_reason(hits, score.get("semantic"), set(expanded)),
        "S3": s3,
        "S4": f"계약방식 {tender.get('cntrctCnclsMthdNm') or '-'} "
              + ("— 선호 방식과 일치" if score["S4"] else "— 선호 방식 아님"),
    }
    return [{"key": k, "item": SCORE_ITEMS[k], "score": score[k], "weight": w, "base_weight": b,
             "points": round(score[k] * w * 100, 1), "reason": reasons[k],
             "interview": abs(w - b) > 1e-9 or (k == "S2" and (s2_interview or bool(expanded)))}
            for k, w, b in zip(SCORE_ITEMS, weights, base_weights)]


def _s2_reason(hits: list[str], semantic: dict | None, expanded: set[str]) -> str:
    labels = [f"{h}(확장어)" if h in expanded else h for h in hits]
    text = f"문자열 일치 {len(hits)}개 / 3개면 만점" + (f": {', '.join(labels)}" if hits else "")
    if semantic:
        text += f" · 의미 유사(BGE-M3, 가까운 키워드 '{semantic['keyword']}') cos {semantic['cos']:.2f} → {semantic['score']:.2f}"
    return text + (" (큰 값 사용)" if semantic else "")


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


def recommend_by_rules(tenders: list[dict], company: dict, region_limits: dict, top_k: int | None = None,
                       prefs: dict | None = None, semantic: dict[str, dict] | None = None) -> dict:
    """500건 → 하드 필터 → 소프트 점수 상위 K건. 화면 퍼널 숫자와 제외 사유 집계도 함께 돌려준다.
    prefs(인터뷰 결과, interview.to_prefs)가 있으면 I1 필터·가중치·지역 가점을, semantic(공고번호 → 의미 유사도)이 있으면 S2에 반영한다."""
    top_k = top_k or config.TOP_K
    weights = prefs["weights"] if prefs else None
    passed, excluded = [], []
    for t in tenders:
        failed = hard_filter(t, company, region_limits)
        if prefs and interview_filter(t, company, prefs):
            failed.append("I1")
        if failed:
            excluded.append({"bidNtceNo": t["bidNtceNo"], "bidNtceNm": t.get("bidNtceNm", ""), "rules": failed})
        else:
            passed.append(t)

    scored = []
    for t in passed:
        s = soft_score(t, company, weights, (semantic or {}).get(t["bidNtceNo"]))
        expanded = tuple(e["term"] for e in (prefs or {}).get("expanded", []))
        items = score_items(t, company, s, weights, prefs and prefs["base_weights"], expanded,
                            s2_interview=bool(prefs and prefs.get("focus")))  # 주력 분야 → 의미 키워드가 바뀜
        if prefs and (bonus := region_bonus(t, company, prefs)):
            items.append(bonus)
            s["total"] = round(s["total"] + bonus["points"], 1)
        scored.append((t, s, items))
    scored.sort(key=lambda x: (-x[1]["total"], x[0]["bidNtceNo"]))  # 동점은 공고번호 순

    top = [{
        "rank": i + 1, "bidNtceNo": t["bidNtceNo"], "bidNtceNm": t.get("bidNtceNm", ""),
        "budget": t.get("budget"), "soft_score": s["total"], "scores": {k: s[k] for k in ("S1", "S2", "S3", "S4")},
        "score_detail": {"items": items, "total": s["total"]},
        "reasons": rule_reasons(t, company, s), "source": "rule",
    } for i, (t, s, items) in enumerate(scored[:top_k])]

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
