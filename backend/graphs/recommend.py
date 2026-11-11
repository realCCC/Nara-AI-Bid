"""추천 Worker (서브그래프): 룰 필터·점수 → 하이브리드 검색(dense+BM25+룰 순위 RRF) → 리랭크 → 상위 K건.

Supervisor가 이 서브그래프 하나를 Worker로 호출한다. 순수 함수(rule_stage 등)는 run_demo·테스트에서도 쓴다.
"""
import hashlib
import json

from langgraph.graph import END, START, StateGraph

from backend import config as settings
from backend.tools import keyword_profile, llm
from backend.tools.interview import expand_company, interview_filter, semantic_keywords
from backend.tools.loaders import load_region_limits, tenders_by_no
from backend.tools.rules import hard_filter, recommend_by_rules
from backend.tools.search import hybrid_search, keyword_similarity, master_first, rerank, template_query, tender_doc


def rule_stage(tenders: list[dict], company: dict, region_limits: dict, prefs: dict | None = None) -> dict:
    """하드 필터 통과분 전체를 점수순으로 (검색 단계 후보).
    S2는 역량 키워드(+유사어) + 인터뷰 확장어 문자열 일치와, 임베딩이 되면 키워드 의미 문장↔공고 의미 유사도 중 큰 값."""
    eff = expand_company(company, prefs)
    passed = {t["bidNtceNo"]: tender_doc(t) for t in tenders
              if not hard_filter(t, eff, region_limits) and not (prefs and interview_filter(t, eff, prefs))}
    kws = semantic_keywords(company, prefs)
    texts = keyword_profile.embed_texts(company)
    sims, reason = keyword_similarity(kws, passed, corpus=[tender_doc(t) for t in tenders],  # 백분위 기준: 전체 공고
                                      texts=texts)
    result = recommend_by_rules(tenders, eff, region_limits, top_k=max(len(tenders), 1), prefs=prefs, semantic=sims)
    profile = keyword_profile.build(company)
    return {"candidates": result["top"],
            "semantic": {"used": sims is not None, "keywords": kws, "reason": reason, "profile": profile["source"],
                         "synonyms": keyword_profile.synonyms(company),
                         "texts": {k: texts[k] for k in kws if k in texts}},
            "filter_log": {"funnel": result["funnel"], "rule_counts": result["rule_counts"], "excluded": result["excluded"]}}


_QUERY_CACHE: dict[str, str] | None = None  # 프로필 해시 → LLM 검색 문장 (QUERY_CACHE 파일에 저장)


def _query_cache() -> dict[str, str]:
    """같은 프로필은 같은 문장. 메모리에만 두면 프로세스마다 LLM이 문장을 새로 만들어 후보가 바뀐다
    (실측 C001: 두 실행의 문장이 달라 정답 '계양아라온 워터축제'가 2위 → RRF 16위로 빠짐) → 파일에 고정."""
    global _QUERY_CACHE
    if _QUERY_CACHE is None:
        try:
            _QUERY_CACHE = {k: v["query"] for k, v in json.loads(settings.QUERY_CACHE.read_text(encoding="utf-8")).items()}
        except (OSError, ValueError, KeyError, TypeError):
            _QUERY_CACHE = {}
    return _QUERY_CACHE


def _save_query(key: str, company: dict, query: str) -> None:
    cache = _query_cache()
    cache[key] = query
    try:
        old = json.loads(settings.QUERY_CACHE.read_text(encoding="utf-8")) if settings.QUERY_CACHE.exists() else {}
        old[key] = {"업체ID": company.get("업체ID"), "query": query}
        settings.QUERY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        settings.QUERY_CACHE.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
    except (OSError, ValueError) as e:  # 저장 실패해도 이번 프로세스에서는 메모리 캐시로 고정
        import logging
        logging.getLogger(__name__).warning("검색 문장 캐시 저장 실패: %s", e)


def make_query(company: dict, prefs: dict | None = None) -> tuple[str, str]:
    """검색 문장. 마스터키가 있으면 담당자가 확인한 검색 프롬프트를 그대로 쓴다.
    없으면 업체정보 → 검색 문장: 주력 분야를 골랐으면 그 분야 문장, LLM이 없거나 실패하면 템플릿.
    고른 유사 표현과 자유 입력 원문은 뒤에 붙인다.
    (키워드 의미 프로필 유사어는 붙이지 않는다 — 20여 개를 붙이면 dense·리랭커 초점이 흐려져 실측에서 상위 10건이 나빠졌다)"""
    prefs = prefs or {}
    if prefs.get("master"):
        return prefs["master"]["prompt"], "master"
    terms = [e["term"] for e in prefs.get("expanded", [])]
    tail = (f" 관련 표현: {', '.join(terms)}" if terms else "") + (f" 요청: {prefs['note']}" if prefs.get("note") else "")
    if prefs.get("focus"):
        return f"{prefs['focus']['query']} 핵심 키워드: {', '.join(prefs['focus']['keywords'])}{tail}", "interview"
    if not settings.llm_enabled():
        return template_query(company) + tail, "template"
    key = hashlib.sha1(json.dumps(company, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cache = _query_cache()
    if key not in cache and (q := llm.search_query(company)):
        _save_query(key, company, q)
    return (cache[key] + tail, "llm") if key in cache else (template_query(company) + tail, "template")


def search_stage(candidates: list[dict], tenders: dict[str, dict], company: dict, prefs: dict | None = None) -> dict:
    query, source = make_query(company, prefs)
    docs = {c["bidNtceNo"]: tender_doc(tenders[c["bidNtceNo"]]) for c in candidates}
    result = hybrid_search(docs, query, rule_order=[c["bidNtceNo"] for c in candidates]) if docs else \
        {"path": "keyword_fallback", "reason": "후보 없음", "query": query, "order": [], "scores": {}, "lists": []}
    if master := (prefs or {}).get("master"):
        result = master_first(result, docs, master["key"])
    return {**result, "query_source": source}


def rerank_stage(search: dict, candidates: list[dict], tenders: dict[str, dict], top_k: int | None = None) -> dict:
    top_k = top_k or settings.TOP_K
    docs = {c["bidNtceNo"]: tender_doc(tenders[c["bidNtceNo"]]) for c in candidates}
    result = rerank(search, docs) if candidates else {**search, "rerank": "skipped", "rerank_reason": "후보 없음"}
    if candidates and search.get("master"):  # 리랭커가 섞어도 마스터키 일치 공고가 앞에 오게
        result = master_first(result, docs, search["master"]["key"])
    by_no = {c["bidNtceNo"]: c for c in candidates}
    pre = {no: r for r, no in enumerate(search.get("order", []), 1)}  # 리랭크 전(하이브리드 검색) 순위
    # journey: 단계별 순위 (룰 점수 → 하이브리드 검색 → 리랭크). RAG·검증 단계가 이어서 채운다 → 화면에서 단계 효과를 보여 준다
    top = [{**by_no[no], "rank": i, "search": result["scores"][no],
            "journey": {"rule": result["scores"][no].get("rule_rank"), "search": pre.get(no), "rerank": i}}
           for i, no in enumerate(result["order"][:top_k], 1)]
    meta = {k: result.get(k) for k in ("path", "reason", "query", "query_source", "lists", "rerank", "rerank_reason",
                                       "semantic", "master", "moves")}
    return {"top": top, "search": meta}


def run_pipeline(tenders: list[dict], company: dict, region_limits: dict, prefs: dict | None = None) -> dict:
    """그래프 없이 Worker 전체 실행 (scripts/run_demo.py --pipeline)."""
    by_no = {t["bidNtceNo"]: t for t in tenders}
    stage = rule_stage(tenders, company, region_limits, prefs)
    out = rerank_stage(search_stage(stage["candidates"], by_no, company, prefs), stage["candidates"], by_no)
    stage["filter_log"]["funnel"]["top"] = len(out["top"])
    return {"top": out["top"], "search": {**out["search"], "semantic": stage["semantic"]}, "filter_log": stage["filter_log"]}


# ---------- 서브그래프 노드 (State 키는 ProposalState와 공유) ----------
def filter_and_score(state):
    stage = rule_stage(list(tenders_by_no(state.get("tenders_path")).values()), state["company_profile"],
                       load_region_limits(), state.get("prefs"))
    return {"candidates": stage["candidates"], "filter_log": stage["filter_log"], "search": {"semantic": stage["semantic"]}}


def search(state):
    result = search_stage(state["candidates"], tenders_by_no(state.get("tenders_path")), state["company_profile"],
                          state.get("prefs"))
    return {"search": {**result, "semantic": state["search"]["semantic"]}}


def rerank_top(state):
    out = rerank_stage(state["search"], state["candidates"], tenders_by_no(state.get("tenders_path")))
    log = {**state["filter_log"], "funnel": {**state["filter_log"]["funnel"], "top": len(out["top"])}}
    return {"top10": out["top"], "search": out["search"], "filter_log": log}


def build_recommend_worker(state_schema):
    b = StateGraph(state_schema)
    b.add_node("filter_and_score", filter_and_score)
    b.add_node("search", search)
    b.add_node("rerank", rerank_top)
    b.add_edge(START, "filter_and_score")
    b.add_edge("filter_and_score", "search")
    b.add_edge("search", "rerank")
    b.add_edge("rerank", END)
    return b.compile()
