"""선택형 인터뷰(추천 전 HITL): 객관식 답변 → 추가 필터(I1)·가중치·지역 가점·키워드 의미 확장. 답이 없거나 "all"이면 기본값.

업체별 문항 2개(주력 분야·유사 표현)는 LLM이 역량 키워드로 만들고, 키가 없으면 과거 실적으로 분야만 만든다.
"""
import json
import math
from datetime import datetime

from backend import config

SCORE_KEYS = ("S1", "S2", "S3", "S4")
PRIORITY_BOOST = 0.15  # 중요기준으로 고른 항목 가중치 +0.15, 나머지는 비례 감소 (합 1 유지)
REGION_BONUS = 5.0     # 우리 지역 발주처 가점 (0-100점 기준)

QUESTIONS = [
    {"id": "budget", "title": "사업 규모", "desc": "예산 범위 밖 공고는 제외합니다.",
     "options": [{"value": "all", "label": "전체"}, {"value": "lt100", "label": "1억 원 미만"},
                 {"value": "100_300", "label": "1억~3억 원"}, {"value": "gt300", "label": "3억 원 이상"}]},
    {"id": "priority", "title": "가장 중요한 기준", "desc": "고른 항목의 점수 가중치를 높입니다.",
     "options": [{"value": "all", "label": "전체 (기본 가중치)"}, {"value": "S1", "label": "업종 일치"},
                 {"value": "S2", "label": "역량 키워드"}, {"value": "S3", "label": "예산 규모(실적 유사)"},
                 {"value": "S4", "label": "선호 계약방식"}]},
    {"id": "region", "title": "지역", "desc": "발주처 이름에 업체 소재지가 들어간 공고를 기준으로 합니다.",
     "options": [{"value": "all", "label": "전체"}, {"value": "bonus", "label": f"우리 지역 발주처 가점 (+{REGION_BONUS:g}점)"},
                 {"value": "only", "label": "우리 지역 발주처만"}]},
    {"id": "schedule", "title": "일정 (공고~마감 준비 기간)", "desc": "마감일이 없는 공고는 통과시킵니다.",
     "options": [{"value": "all", "label": "전체"}, {"value": "7", "label": "7일 이상"}, {"value": "14", "label": "14일 이상"}]},
]
BUDGET_RANGES = {"lt100": (0, 100_000_000), "100_300": (100_000_000, 300_000_000), "gt300": (300_000_000, math.inf)}
REGION_NAMES = {
    "서울": ["서울특별시"], "부산": ["부산광역시"], "대구": ["대구광역시"], "인천": ["인천광역시"],
    "광주": ["광주광역시", "전남광주"], "대전": ["대전광역시"], "울산": ["울산광역시"], "세종": ["세종특별자치시"],
    "경기": ["경기도"], "강원": ["강원특별자치도", "강원도"], "충북": ["충청북도"], "충남": ["충청남도"],
    "전북": ["전북특별자치도", "전라북도"], "전남": ["전라남도", "전남광주"], "경북": ["경상북도"], "경남": ["경상남도"],
    "제주": ["제주특별자치도"],
}


_SUGGEST_CACHE: dict[str, dict | None] = {}
_BROAD = {"사업", "용역", "서비스", "운영", "지원", "관리"}


def _fallback_focus(company: dict) -> list[dict]:
    """LLM 없음: 과거 실적마다 실적 키워드를 하나의 분야로."""
    out, seen = [], set()
    for h in company.get("과거실적", []):
        kws = tuple(k for k in h.get("키워드", []) if k)
        if kws and kws not in seen:
            seen.add(kws)
            out.append({"label": "·".join(kws), "keywords": list(kws), "query": f"{h['프로젝트명']}와 비슷한 용역"})
    return out[:4]


def _clean_suggest(raw: dict | None, company: dict) -> dict:
    """LLM 제안 검증: 분야 키워드는 업체 키워드 안에서만, 확장어는 업체 키워드와 겹치지 않는 짧은 명사만."""
    own = set(company.get("역량키워드", []))
    focus = []
    for f in (raw or {}).get("focus", [])[:4]:
        kws = [k for k in f.get("keywords", []) if k in own]
        kws = [k for k in kws if k not in _BROAD] or kws  # '운영' 같은 넓은 키워드가 분야 의미를 흐리지 않게
        if kws and f.get("label") and f.get("query"):
            focus.append({"label": f["label"][:20], "keywords": kws, "query": f["query"]})
    expansions, used = [], set(own)
    for e in (raw or {}).get("expansions", []):
        if e.get("keyword") not in own:
            continue
        terms = [s.replace(" ", "") for s in e.get("similar", [])]  # 문자열 일치(S2)에 쓰려면 공백 없는 단어
        terms = list(dict.fromkeys(t for t in terms if 2 <= len(t) <= 10 and t not in used | _BROAD))
        used.update(terms)
        if terms:
            expansions.append({"keyword": e["keyword"], "similar": terms[:3]})
    return {"focus": focus, "expansions": expansions}


def keyword_suggestions(company: dict) -> dict:
    key = json.dumps([company.get("업체ID"), company.get("역량키워드"), company.get("과거실적")], ensure_ascii=False)
    if key not in _SUGGEST_CACHE:
        raw = None
        if config.llm_enabled():
            from backend.tools import llm

            raw = llm.suggest_interview(company)
        s = _clean_suggest(raw, company)
        if not s["focus"]:
            s["focus"] = _fallback_focus(company)
        s["source"] = "llm" if raw else "fallback"
        _SUGGEST_CACHE[key] = s
    return _SUGGEST_CACHE[key]


def build_questions(company: dict) -> list[dict]:
    """업체별 문항(주력 분야·유사 표현) + 공통 문항."""
    s, qs = keyword_suggestions(company), []
    if s["focus"]:
        qs.append({"id": "focus", "title": "이번에 찾는 주력 분야", "type": "single",
                   "desc": "고른 분야의 키워드 의미로 검색 문장과 의미 유사도(S2)를 맞춥니다.",
                   "options": [{"value": "all", "label": "전체 (모든 역량 키워드)"}] +
                              [{"value": f"f{i}", "label": f["label"], "keywords": f["keywords"], "query": f["query"]}
                               for i, f in enumerate(s["focus"])]})
    if s["expansions"]:
        qs.append({"id": "expand", "title": "함께 찾을 유사 표현", "type": "multi",
                   "desc": "업체 키워드와 같은 뜻으로 쓰이는 표현을 고르면 키워드 일치·검색에 함께 씁니다. 고르지 않으면 원래 키워드만.",
                   "options": [{"value": t, "label": t, "group": e["keyword"]} for e in s["expansions"] for t in e["similar"]]})
    return qs + QUESTIONS


def clean_answers(answers: dict | None, questions: list[dict] | None = None) -> dict:
    """모르는 문항·값, 건너뛴 문항(None/"all"/빈 목록)은 버린다. multi 문항은 목록."""
    qs = {q["id"]: q for q in (questions or QUESTIONS)}
    out = {}
    for k, v in (answers or {}).items():
        if k not in qs:
            continue
        valid = {o["value"] for o in qs[k]["options"]}
        if qs[k].get("type") == "multi":
            picked = list(dict.fromkeys(x for x in (v if isinstance(v, list) else [v]) if x in valid))
            if picked:
                out[k] = picked
        elif isinstance(v, str) and v in valid and v != "all":
            out[k] = v
    return out


def boost_weights(base: tuple, key: str) -> tuple:
    i = SCORE_KEYS.index(key)
    new = min(1.0, base[i] + PRIORITY_BOOST)
    scale = (1 - new) / (1 - base[i]) if base[i] < 1 else 0
    return tuple(round(new if j == i else w * scale, 4) for j, w in enumerate(base))


def to_prefs(answers: dict | None, base_weights: tuple | None = None, questions: list[dict] | None = None) -> dict:
    """인터뷰 답변 → 룰 단계 입력. weights는 soft_score에, 나머지는 interview_filter·region_bonus·키워드 확장에 쓴다."""
    qs = questions or QUESTIONS
    a = clean_answers(answers, qs)
    base = tuple(base_weights or config.SCORE_WEIGHTS)
    opts = {q["id"]: {o["value"]: o for o in q["options"]} for q in qs}
    focus = opts["focus"][a["focus"]] if "focus" in a else None
    return {
        "answers": a,
        "focus": focus and {k: focus[k] for k in ("label", "keywords", "query")},
        "expanded": [{"term": t, "group": opts["expand"][t]["group"]} for t in a.get("expand", [])],
        "base_weights": base,
        "weights": boost_weights(base, a["priority"]) if "priority" in a else base,
        "priority": a.get("priority"),
        "budget": BUDGET_RANGES.get(a.get("budget")),
        "region": a.get("region"),
        "min_prep_days": int(a["schedule"]) if "schedule" in a else None,
    }


def expand_company(company: dict, prefs: dict | None) -> dict:
    """키워드 일치(S2)·BM25·PDF 청크 검색용 프로필: 역량 키워드 + 고른 유사 표현. (LLM 판단에는 원래 프로필을 쓴다)"""
    terms = [e["term"] for e in (prefs or {}).get("expanded", [])]
    return {**company, "역량키워드": [*company.get("역량키워드", []), *terms]} if terms else company


def semantic_keywords(company: dict, prefs: dict | None) -> list[str]:
    """의미 유사도(BGE-M3) 기준 키워드: 주력 분야를 골랐으면 그 분야 키워드(+그 키워드의 유사 표현), 아니면 전체."""
    prefs = prefs or {}
    expanded = prefs.get("expanded", [])
    if prefs.get("focus"):
        base = prefs["focus"]["keywords"]
        return [*base, *(e["term"] for e in expanded if e["group"] in base)]
    return [*company.get("역량키워드", []), *(e["term"] for e in expanded)]


def region_match(tender: dict, company: dict) -> bool:
    names = REGION_NAMES.get(company.get("소재지", ""), [company.get("소재지") or "\0"])
    insts = f"{tender.get('dminsttNm', '')} {tender.get('ntceInsttNm', '')}"
    return any(n in insts for n in names)


def prep_days(tender: dict) -> int | None:
    try:
        start = datetime.strptime(tender["bidNtceDt"][:16], "%Y-%m-%d %H:%M")
        end = datetime.strptime(tender["bidClseDt"][:16], "%Y-%m-%d %H:%M")
    except (KeyError, TypeError, ValueError):
        return None
    return (end - start).days


def interview_filter(tender: dict, company: dict, prefs: dict) -> bool:
    """True면 인터뷰 조건 밖 (I1)."""
    if prefs.get("budget") and tender.get("budget") is not None:
        lo, hi = prefs["budget"]
        if not lo <= tender["budget"] < hi:
            return True
    if prefs.get("region") == "only" and not region_match(tender, company):
        return True
    if prefs.get("min_prep_days"):
        days = prep_days(tender)
        if days is not None and days < prefs["min_prep_days"]:
            return True
    return False


def region_bonus(tender: dict, company: dict, prefs: dict) -> dict | None:
    if prefs.get("region") != "bonus" or not region_match(tender, company):
        return None
    return {"key": "B1", "item": "지역 가점", "score": None, "weight": None, "base_weight": None,
            "points": REGION_BONUS, "interview": True,
            "reason": f"발주처({tender.get('dminsttNm', '')})가 업체 소재지({company.get('소재지')}) 지역"}
