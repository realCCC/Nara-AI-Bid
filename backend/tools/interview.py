"""선택형 인터뷰(추천 전 HITL): LLM이 업체정보를 읽고 문항·선택지를 만든다 → 답변 → 추가 필터(I1)·가중치·지역 가점·키워드 의미 확장.

선택지마다 effect(정해진 효과 어휘)를 붙이고, 코드가 effect만 검증·적용한다. LLM이 없거나 출력이 검증을 못 넘으면 기본 문항.
마지막 문항은 자유 입력(note): LLM이 같은 effect로 해석한다(있으면 선택 답보다 우선).
인터뷰 뒤에는 핵심 사업(이번에 가장 원하는 사업)과 역량 맞춤 검색 프롬프트를 담당자가 확인하고, 그 값이 검색·순위·검증의 최우선 기준이 된다.
답이 없거나 "all"이면 기본값.
"""
import json
import math
from datetime import datetime

from backend import config
from backend.tools import keyword_profile

SCORE_KEYS = ("S1", "S2", "S3", "S4")
SCORE_NAMES = {"S1": "업종 일치", "S2": "역량 키워드", "S3": "예산 규모(실적 유사)", "S4": "선호 계약방식"}
PRIORITY_BOOST = 0.15  # 중요기준으로 고른 항목 가중치 +0.15, 나머지는 비례 감소 (합 1 유지)
REGION_BONUS = 5.0     # 우리 지역 발주처 가점 (0-100점 기준)
NOTE_GROUP = "직접 입력"  # 자유 입력에서 나온 키워드의 그룹 (주력 분야와 상관없이 의미 매칭에 쓴다)
NOTE_MAX = 500
MASTER_KEY_MAX, MASTER_PROMPT_MAX, MAX_MASTERS = 30, 300, 5
MAX_QUESTIONS, MAX_OPTIONS = 6, 6

ALL = {"value": "all", "label": "전체 (기본값)", "effect": {}}
NOTE_QUESTION = {"id": "note", "title": "추가로 알려주실 내용", "type": "text", "source": "default",
                 "desc": "찾는 사업·피하고 싶은 사업·예산·지역 등을 자유롭게 적어 주세요. 해석된 조건은 위 선택보다 우선합니다.",
                 "placeholder": "예) 올해는 AI 교육 위주로 찾고 있고, 시설 공사·물품 구매는 빼 주세요. 2억 이하가 좋아요."}

DEFAULT_QUESTIONS = [
    {"id": "budget", "title": "사업 규모", "desc": "예산 범위 밖 공고는 제외합니다.",
     "options": [ALL, {"value": "lt100", "label": "1억 원 미만", "effect": {"budget": [0, 100_000_000]}},
                 {"value": "100_300", "label": "1억~3억 원", "effect": {"budget": [100_000_000, 300_000_000]}},
                 {"value": "gt300", "label": "3억 원 이상", "effect": {"budget": [300_000_000, None]}}]},
    {"id": "priority", "title": "가장 중요한 기준", "desc": "고른 항목의 점수 가중치를 높입니다.",
     "options": [ALL, *({"value": k, "label": v, "effect": {"priority": k}} for k, v in SCORE_NAMES.items())]},
    {"id": "region", "title": "지역", "desc": "발주처 이름에 업체 소재지가 들어간 공고를 기준으로 합니다.",
     "options": [ALL, {"value": "bonus", "label": f"우리 지역 발주처 가점 (+{REGION_BONUS:g}점)", "effect": {"region": "bonus"}},
                 {"value": "only", "label": "우리 지역 발주처만", "effect": {"region": "only"}}]},
    {"id": "schedule", "title": "일정 (공고~마감 준비 기간)", "desc": "마감일이 없는 공고는 통과시킵니다.",
     "options": [ALL, {"value": "7", "label": "7일 이상", "effect": {"min_prep_days": 7}},
                 {"value": "14", "label": "14일 이상", "effect": {"min_prep_days": 14}}]},
]
QUESTIONS = [{**q, "type": "single", "source": "default"} for q in DEFAULT_QUESTIONS]
REGION_NAMES = {
    "서울": ["서울특별시"], "부산": ["부산광역시"], "대구": ["대구광역시"], "인천": ["인천광역시"],
    "광주": ["광주광역시", "전남광주"], "대전": ["대전광역시"], "울산": ["울산광역시"], "세종": ["세종특별자치시"],
    "경기": ["경기도"], "강원": ["강원특별자치도", "강원도"], "충북": ["충청북도"], "충남": ["충청남도"],
    "전북": ["전북특별자치도", "전라북도"], "전남": ["전라남도", "전남광주"], "경북": ["경상북도"], "경남": ["경상남도"],
    "제주": ["제주특별자치도"],
}


_QUESTION_CACHE: dict[str, list[dict]] = {}
_BROAD = {"사업", "용역", "서비스", "운영", "지원", "관리"}


def _term(s) -> str | None:
    """문자열 일치(S2)·공고명 제외에 쓸 수 있는 공백 없는 2-10자 단어."""
    t = str(s or "").replace(" ", "")
    return t if 2 <= len(t) <= 10 and t not in _BROAD else None


def clean_effect(raw: dict | None, company: dict) -> dict:
    """LLM effect 검증: 아는 키·범위 안의 값만 남긴다. 분야 키워드는 업체 키워드 안에서만, 확장어는 업체 키워드와 겹치지 않게."""
    raw, own, out = raw or {}, set(company.get("역량키워드", [])), {}
    lo, hi = raw.get("budget_min_won"), raw.get("budget_max_won")
    if "budget" in raw:  # 내부 형식 [lo, hi]
        lo, hi = (list(raw["budget"]) + [None, None])[:2]
    lo = lo if isinstance(lo, (int, float)) and lo > 0 else 0
    hi = hi if isinstance(hi, (int, float)) and hi > 0 else None
    if (lo or hi) and (hi is None or lo < hi):
        out["budget"] = [int(lo), int(hi) if hi else None]
    if raw.get("priority") in SCORE_KEYS:
        out["priority"] = raw["priority"]
    if raw.get("region") in ("bonus", "only"):
        out["region"] = raw["region"]
    days = raw.get("min_prep_days")
    if isinstance(days, int) and 1 <= days <= 60:
        out["min_prep_days"] = days
    focus = raw.get("focus") or {"keywords": raw.get("focus_keywords"), "query": raw.get("focus_query"),
                                 "label": raw.get("focus_label")}
    kws = [k for k in focus.get("keywords") or [] if k in own]
    kws = [k for k in kws if k not in _BROAD] or kws  # '운영' 같은 넓은 키워드가 분야 의미를 흐리지 않게
    if kws and (focus.get("query") or "").strip():
        out["focus"] = {"label": (focus.get("label") or "·".join(kws))[:20], "keywords": kws, "query": focus["query"].strip()}
    group = raw.get("expand_group")
    expand = raw.get("expand") or [{"term": t, "group": group} for t in raw.get("expand_terms") or []]
    expand = [{"term": t, "group": e.get("group")} for e in expand if (t := _term(e.get("term"))) and t not in own
              and (e.get("group") in own or e.get("group") == NOTE_GROUP)]
    if expand:
        out["expand"] = list({e["term"]: e for e in expand}.values())
    exclude = list(dict.fromkeys(t for x in raw.get("exclude") or raw.get("exclude_terms") or [] if (t := _term(x))))
    if exclude:
        out["exclude"] = exclude
    return out


def effect_text(effect: dict) -> str:
    """선택지가 실제로 하는 일 (화면 툴팁)."""
    parts = []
    if b := effect.get("budget"):
        parts.append(f"예산 {b[0] / 1e8:g}억~{'' if b[1] is None else f'{b[1] / 1e8:g}억'} 밖 제외")
    if p := effect.get("priority"):
        parts.append(f"{SCORE_NAMES[p]} 가중치 +{PRIORITY_BOOST:g}")
    if r := effect.get("region"):
        parts.append("우리 지역 발주처만" if r == "only" else f"우리 지역 발주처 +{REGION_BONUS:g}점")
    if d := effect.get("min_prep_days"):
        parts.append(f"준비 기간 {d}일 미만 제외")
    if f := effect.get("focus"):
        parts.append(f"주력 분야 키워드 {', '.join(f['keywords'])}")
    if e := effect.get("expand"):
        parts.append(f"확장어 {', '.join(x['term'] for x in e)}")
    if x := effect.get("exclude"):
        parts.append(f"공고명에 {', '.join(x)} 있으면 제외")
    return " · ".join(parts)


def _clean_questions(raw: dict | None, company: dict) -> list[dict]:
    """LLM 문항 검증: effect가 빈 선택지는 버리고, 선택지가 모자란 문항도 버린다. single은 맨 앞에 '전체'를 붙인다.
    한 문항에는 가장 많이 쓰인 효과 종류만 남긴다 (예산 문항에 가중치가 섞이는 것 방지)."""
    out = []
    for q in (raw or {}).get("questions", [])[:MAX_QUESTIONS]:
        kind = q.get("type") if q.get("type") in ("single", "multi") else "single"
        effs = [(o, clean_effect(o.get("effect"), company)) for o in q.get("options", [])[:MAX_OPTIONS]]
        keys = [k for _, e in effs for k in e]
        main = max(keys, key=keys.count) if keys else None
        opts = []
        for o, eff in effs:
            eff = {main: eff[main]} if main in eff else {}
            if eff and (o.get("label") or "").strip():
                opts.append({"value": f"o{len(opts)}", "label": o["label"].strip()[:30], "effect": eff,
                             "hint": effect_text(eff), "group": (eff.get("expand") or [{}])[0].get("group")})
        if len(opts) >= (1 if kind == "multi" else 2) and (q.get("title") or "").strip():
            out.append({"id": f"q{len(out)}", "title": q["title"].strip()[:40], "desc": (q.get("desc") or "").strip()[:80],
                        "type": kind, "source": "llm", "options": opts if kind == "multi" else [ALL, *opts]})
    return out


def _fallback_focus(company: dict) -> list[dict]:
    """LLM 없음: 과거 실적마다 실적 키워드를 하나의 분야로."""
    out, seen = [], set()
    for h in company.get("과거실적", []):
        kws = tuple(k for k in h.get("키워드", []) if k)
        if kws and kws not in seen:
            seen.add(kws)
            eff = {"focus": {"label": "·".join(kws), "keywords": list(kws), "query": f"{h['프로젝트명']}와 비슷한 용역"}}
            out.append({"value": f"f{len(out)}", "label": "·".join(kws), "effect": eff, "hint": effect_text(eff)})
    return out[:4]


def _default_questions(company: dict) -> list[dict]:
    focus = _fallback_focus(company)
    head = [{"id": "focus", "title": "이번에 찾는 주력 분야", "type": "single", "source": "default",
             "desc": "고른 분야의 키워드 의미로 검색 문장과 의미 유사도(S2)를 맞춥니다.", "options": [ALL, *focus]}] if focus else []
    return head + QUESTIONS


def build_questions(company: dict) -> list[dict]:
    """LLM이 업체정보로 만든 문항(실패 시 기본 문항) + 마지막 자유 입력 문항."""
    key = json.dumps([company.get(k) for k in ("업체ID", "업종", "소재지", "참여가능금액", "역량키워드", "선호계약방식", "과거실적")],
                     ensure_ascii=False)
    if key not in _QUESTION_CACHE:
        qs = []
        if config.llm_enabled():
            from backend.tools import llm

            qs = _clean_questions(llm.generate_interview(company), company)
        _QUESTION_CACHE[key] = qs if len(qs) >= 2 else _default_questions(company)
    return [*_QUESTION_CACHE[key], NOTE_QUESTION]


def clean_answers(answers: dict | None, questions: list[dict] | None = None) -> dict:
    """모르는 문항·값, 건너뛴 문항(None/"all"/빈 목록·빈 문자열)은 버린다. multi 문항은 목록, text 문항은 문자열."""
    qs = {q["id"]: q for q in (questions or QUESTIONS)}
    out = {}
    for k, v in (answers or {}).items():
        if k not in qs:
            continue
        if qs[k].get("type") == "text":
            if isinstance(v, str) and v.strip():
                out[k] = v.strip()[:NOTE_MAX]
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


def merge_effects(effects: list[dict]) -> dict:
    """뒤에 오는 effect가 앞을 덮는다 (자유 입력 해석을 맨 뒤에). 확장어·제외어는 합친다."""
    out: dict = {}
    for e in effects:
        for k, v in e.items():
            if k in ("expand", "exclude"):
                out[k] = list({(x["term"] if k == "expand" else x): x for x in [*out.get(k, []), *v]}.values())
            else:
                out[k] = v
    return out


def to_prefs(answers: dict | None, base_weights: tuple | None = None, questions: list[dict] | None = None,
             note_effect: dict | None = None) -> dict:
    """인터뷰 답변(+자유 입력 해석) → 룰 단계 입력. weights는 soft_score에, 나머지는 interview_filter·region_bonus·키워드 확장에 쓴다."""
    qs = questions or QUESTIONS
    a = clean_answers(answers, qs)
    base = tuple(base_weights or config.SCORE_WEIGHTS)
    opts = {q["id"]: {o["value"]: o for o in q.get("options", [])} for q in qs}
    picked = [opts[k][v].get("effect", {}) for k, vs in a.items() if k in opts and opts[k]
              for v in (vs if isinstance(vs, list) else [vs])]
    eff = merge_effects([*picked, note_effect or {}])
    budget = eff.get("budget")
    return {
        "answers": a,
        "note": a.get("note"),
        "note_effect": note_effect or None,
        "focus": eff.get("focus"),
        "expanded": eff.get("expand", []),
        "exclude": eff.get("exclude", []),
        "base_weights": base,
        "weights": boost_weights(base, eff["priority"]) if "priority" in eff else base,
        "priority": eff.get("priority"),
        "budget": budget and (budget[0], math.inf if budget[1] is None else budget[1]),
        "region": eff.get("region"),
        "min_prep_days": eff.get("min_prep_days"),
    }


def parse_note(note: str | None, company: dict) -> dict:
    """자유 입력 → effect. LLM이 없거나 실패하면 빈 effect (원문은 검색 문장에만 붙는다).
    찾고 싶은 단어가 업체 키워드면 주력 분야로, 아니면 확장어로."""
    if not note or not config.llm_enabled():
        return {}
    from backend.tools import llm

    raw = llm.parse_note(note, company)
    if not raw:
        return {}
    own = set(company.get("역량키워드", []))
    include = raw.get("include_terms") or []
    raw = {**raw, "expand": [{"term": t, "group": NOTE_GROUP} for t in include if t not in own]}
    if focus := [t for t in include if t in own]:
        raw["focus"] = {"label": "·".join(focus), "keywords": focus, "query": note}
    return clean_effect(raw, company)


def answer_lines(questions: list[dict], prefs: dict) -> list[str]:
    """고른 답 → '문항: 선택지' 줄. 내부 값이 아니라 담당자가 본 문구로 적는다."""
    out = []
    for q in questions:
        v = prefs["answers"].get(q["id"])
        if v is None:
            continue
        if q.get("type") == "text":
            out.append(f"{q['title']}: {v}")
        else:
            labels = {o["value"]: o["label"] for o in q["options"]}
            out.append(f"{q['title']}: {', '.join(labels[x] for x in (v if isinstance(v, list) else [v]))}")
    return out


def clean_master_key(text) -> str:
    return " ".join(text.split())[:MASTER_KEY_MAX] if isinstance(text, str) else ""


def clean_master_prompt(text) -> str:
    return " ".join(text.split())[:MASTER_PROMPT_MAX] if isinstance(text, str) else ""


def template_master_prompt(company: dict, key: str) -> str:
    """LLM 없음·실패: 핵심 사업 + 역량 키워드 (지어낼 여지가 없다)."""
    keywords = [k for k in company.get("역량키워드", []) if k not in _BROAD][:4]
    return clean_master_prompt(f"{key} 관련 공공 용역 사업" + (f" 역량: {', '.join(keywords)}" if keywords else ""))


def _anchored(key: str, prompt: str) -> str:
    """검색 프롬프트에 핵심 사업의 주요 단어가 빠지면 키를 앞에 붙인다 (핵심 사업이 검색 중심이어야 한다)."""
    return prompt if all(w in prompt for w in key.split() if w not in _BROAD) else clean_master_prompt(f"{key} {prompt}")


def _clean_masters(raw: list[dict] | None) -> list[dict]:
    out: dict[str, dict] = {}
    for c in (raw or [])[:MAX_MASTERS]:
        key, prompt = clean_master_key(c.get("key")), clean_master_prompt(c.get("prompt"))
        if key and prompt and key not in out:
            out[key] = {"key": key, "reason": str(c.get("reason") or "").strip()[:80], "prompt": _anchored(key, prompt)}
    return list(out.values())


def _fallback_masters(company: dict, prefs: dict) -> list[dict]:
    """인터뷰 주력 분야 → 과거 실적 키워드 → 역량 키워드 순으로 후보."""
    focus = prefs.get("focus") or {}
    past = [k for h in company.get("과거실적", []) for k in h.get("키워드", [])]
    keys = [k for k in dict.fromkeys([*focus.get("keywords", []), *past, *company.get("역량키워드", [])]) if k not in _BROAD]
    return [{"key": k, "reason": "업체 역량·실적 키워드", "prompt": template_master_prompt(company, k)}
            for k in map(clean_master_key, keys[:MAX_MASTERS]) if k]


def build_masters(company: dict, questions: list[dict], prefs: dict) -> dict:
    """업체 역량 + 인터뷰 답변 → 핵심 사업 후보와 후보별 검색 프롬프트. source: llm | template."""
    if config.llm_enabled():
        from backend.tools import llm

        cands = _clean_masters(llm.master_candidates(company, answer_lines(questions, prefs)))
        if len(cands) >= 2:
            return {"candidates": cands, "source": "llm"}
    return {"candidates": _fallback_masters(company, prefs), "source": "template"}


def master_prompt(company: dict, key: str, answers: list[str]) -> str:
    """담당자가 직접 적은 핵심 사업 → 역량에 맞춘 검색 프롬프트."""
    prompt = ""
    if config.llm_enabled():
        from backend.tools import llm

        prompt = clean_master_prompt(llm.master_search_prompt(company, key, answers))
    return _anchored(key, prompt) if prompt else template_master_prompt(company, key)


def resolve_master(draft: dict, picked: dict | None, company: dict) -> dict | None:
    """담당자 확인 값 → 확정 핵심 사업. 비우면 첫 후보, 후보도 없으면 None (기존 검색 경로)."""
    picked, cands = picked or {}, draft["candidates"]
    key = clean_master_key(picked.get("key")) or (cands[0]["key"] if cands else "")
    if not key:
        return None
    cand = next((c for c in cands if c["key"] == key), None)
    base = cand["prompt"] if cand else template_master_prompt(company, key)
    prompt = clean_master_prompt(picked.get("prompt")) or base
    return {"key": key, "prompt": prompt, "source": draft["source"] if cand else "custom", "edited": prompt != base}


def with_master(prefs: dict, master: dict | None, lines: list[str]) -> dict:
    """확정 핵심 사업을 prefs에 넣는다. 공백 없는 키는 S2 문자열 일치·PDF 청크 검색에도 쓴다."""
    term = _term(master["key"]) if master else None
    expanded = [*prefs["expanded"], {"term": term, "group": NOTE_GROUP}] if term else prefs["expanded"]
    return {**prefs, "master": master, "answer_lines": lines, "expanded": expanded}


def with_note(company: dict, prefs: dict | None) -> dict:
    """LLM 판단(RAG 순위·검증)용 프로필: 확정 핵심 사업·검색 프롬프트, 고른 답, 자유 입력 원문을 붙인다."""
    prefs, extra = prefs or {}, {}
    if master := prefs.get("master"):
        extra |= {"핵심사업": master["key"], "검색프롬프트": master["prompt"]}
    if prefs.get("answer_lines"):
        extra["인터뷰답변"] = prefs["answer_lines"]
    if prefs.get("note"):
        extra["추가요청"] = prefs["note"]
    return {**company, **extra} if extra else company


def expand_company(company: dict, prefs: dict | None) -> dict:
    """키워드 일치(S2)·BM25·PDF 청크 검색용 프로필: 역량 키워드 + 고른 유사 표현 + 키워드 의미 프로필의 유사어('유사어').
    (LLM 판단에는 원래 프로필을 쓴다)"""
    terms = [e["term"] for e in (prefs or {}).get("expanded", [])]
    out = {**company, "역량키워드": [*company.get("역량키워드", []), *terms]} if terms else company
    syn = keyword_profile.synonyms(company)
    return {**out, "유사어": syn} if syn else out


def semantic_keywords(company: dict, prefs: dict | None) -> list[str]:
    """의미 유사도(BGE-M3) 기준 키워드: 주력 분야를 골랐으면 그 분야 키워드(+그 키워드의 유사 표현·직접 입력), 아니면 전체."""
    prefs = prefs or {}
    expanded = prefs.get("expanded", [])
    if prefs.get("focus"):
        base = prefs["focus"]["keywords"]
        return [*base, *(e["term"] for e in expanded if e["group"] in base or e["group"] == NOTE_GROUP)]
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
    name = (tender.get("bidNtceNm") or "").replace(" ", "")
    return any(x in name for x in prefs.get("exclude", []))


def region_bonus(tender: dict, company: dict, prefs: dict) -> dict | None:
    if prefs.get("region") != "bonus" or not region_match(tender, company):
        return None
    return {"key": "B1", "item": "지역 가점", "score": None, "weight": None, "base_weight": None,
            "points": REGION_BONUS, "interview": True,
            "reason": f"발주처({tender.get('dminsttNm', '')})가 업체 소재지({company.get('소재지')}) 지역"}
