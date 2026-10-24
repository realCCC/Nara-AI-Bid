"""역량키워드 의미 프로필: LLM이 키워드마다 설명문·유사어·가상 공고명을 만든다 (업체당 1회, 캐시).

설명문·가상 공고명 → S2 의미 매칭 임베딩 문장 (단어 하나보다 의미가 또렷하다)
유사어 → S2 문자열 일치(같은 키워드의 유사어는 1개로 센다)·BM25 검색 문장·PDF 청크 검색
LLM이 없거나 실패하면 빈 프로필 → 키워드 그대로 (기존 동작).
"""
import json

from backend import config

_CACHE: dict[str, dict] = {}
_BROAD = {"사업", "용역", "서비스", "운영", "지원", "관리"}
MAX_SYNONYMS = 4


def _key(company: dict) -> str:
    return json.dumps([company.get(k) for k in ("업체ID", "업종", "역량키워드", "과거실적")], ensure_ascii=False)


def clean(raw: dict | None, company: dict) -> dict:
    """키워드는 업체 키워드 안에서만, 유사어는 공백 없는 2-10자 단어로 업체 키워드·넓은 단어·다른 키워드 유사어와 겹치지 않게."""
    own = list(dict.fromkeys(company.get("역량키워드", [])))
    used, out = set(own) | _BROAD, []
    for k in (raw or {}).get("keywords", []):
        kw = k.get("keyword")
        if kw not in own or any(x["keyword"] == kw for x in out):
            continue
        syns = []
        for s in k.get("synonyms") or []:
            t = str(s).replace(" ", "")
            if 2 <= len(t) <= 10 and t not in used:
                used.add(t)
                syns.append(t)
        out.append({"keyword": kw, "description": (k.get("description") or "").strip()[:100],
                    "synonyms": syns[:MAX_SYNONYMS], "sample_title": (k.get("sample_title") or "").strip()[:80]})
    return {"source": "llm" if out else "none", "keywords": out}


def build(company: dict) -> dict:
    key = _key(company)
    if key not in _CACHE:
        raw = None
        if config.llm_enabled() and company.get("역량키워드"):
            from backend.tools import llm

            raw = llm.keyword_profile(company)
        _CACHE[key] = clean(raw, company)
    return _CACHE[key]


def synonyms(company: dict) -> dict[str, list[str]]:
    """키워드 → 유사어 (유사어가 있는 키워드만)."""
    return {k["keyword"]: k["synonyms"] for k in build(company)["keywords"] if k["synonyms"]}


def embed_texts(company: dict) -> dict[str, str]:
    """키워드 → 임베딩 문장. '행사: 공공기관 기념식·축제 운영 대행 예) ○○시 시민의 날 기념행사 운영 용역'."""
    out = {}
    for k in build(company)["keywords"]:
        text = f"{k['keyword']}: {k['description']}" if k["description"] else k["keyword"]
        out[k["keyword"]] = f"{text} 예) {k['sample_title']}" if k["sample_title"] else text
    return out
