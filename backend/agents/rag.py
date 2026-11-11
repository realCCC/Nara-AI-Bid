"""RAG 순위 Worker: 상위 10건 공고서 청크를 읽고 LLM이 순위·근거를 매긴다. 실패하면 룰 순위."""
from backend import config as settings
from backend.agents.state import ProposalState, tenders
from backend.tools import activity, llm
from backend.tools.evidence import check_evidence_sources
from backend.tools.interview import expand_company, with_note
from backend.tools.pdf import tender_chunks
from backend.tools.retrieval import retrieve_for_company
from backend.tools.rules import rule_reasons, soft_score


def _with_rule(base: dict) -> dict:
    """응답 영역 분리: rule(점수표·룰 근거) / llm(LLM 근거·플래그, 없으면 None)."""
    return {**base, "rule": {"soft_score": base["soft_score"], "score_detail": base.get("score_detail"),
                             "reasons": base["reasons"]}, "llm": None}


def rag_recommend(state: ProposalState):
    company, top, by_no_t = state["company_profile"], state["top10"], tenders(state)
    if not top:
        return {"recommendations": [], "rag_chunks": {}, "rag_status": "no_candidates"}
    recs = [_with_rule(c) for c in top]
    if not settings.llm_enabled():
        return {"recommendations": recs, "rag_chunks": {}, "rag_status": "rule_only"}
    search_profile = expand_company(company, state.get("prefs"))  # 청크 검색에는 인터뷰 확장어도 쓴다
    query = (state.get("search") or {}).get("query")  # 공고 검색에 쓴 검색 문장 → 과업 내용 청크를 dense로
    chunks, paths = {}, {}
    with activity.step("RAG", f"공고서 근거 검색 중 · 공고 {len(top)}건") as done:
        for c in top:
            no = c["bidNtceNo"]
            chunks[no], paths[no] = retrieve_for_company(tender_chunks(by_no_t[no]), search_profile, query=query, tender_no=no)
        used = list(paths.values())
        hybrid, bm25 = used.count("hybrid"), used.count("bm25")
        done |= {"message": f"근거 청크 {sum(map(len, chunks.values()))}개 검색 · 벡터+BM25 {hybrid}건, BM25만 {bm25}건",
                 "level": "warn" if bm25 else "info"}
    chunk_search = {"paths": paths, "query": query}
    judged = with_note(company, state.get("prefs"))  # 마스터키·인터뷰 답도 순위 판단·근거 대조에
    ranked = llm.rag_rank(judged, top, chunks, by_no_t)
    if ranked is None:  # 출력 검증 실패 → 룰 순위로 대체
        return {"recommendations": recs, "rag_chunks": chunks, "rag_status": "rule_fallback", "rag_chunk_search": chunk_search}
    by_no = {c["bidNtceNo"]: c for c in recs}
    out = []
    for i, r in enumerate(ranked, 1):
        base, t = by_no[r["bidNtceNo"]], by_no_t[r["bidNtceNo"]]
        llm_reasons = check_evidence_sources(r["reasons"], t, judged, chunks[r["bidNtceNo"]])
        shown = list(llm_reasons)
        if len(shown) < 2:  # 출처 확인된 근거가 2개 미만이면 룰 근거 보충
            shown += rule_reasons(t, company, soft_score(t, company))[: 3 - len(shown)]
        out.append({**base, "rank": i, "reasons": shown, "source": "rag", "journey": {**base.get("journey", {}), "rag": i},
                    "qualification_flags": r.get("qualification_flags", []),
                    "pdf_evidence": bool(chunks[r["bidNtceNo"]]),
                    "llm": {"rank": i, "reasons": llm_reasons, "dropped": len(r["reasons"]) - len(llm_reasons),
                            "supplemented": len(shown) - len(llm_reasons),
                            "qualification_flags": r.get("qualification_flags", []),
                            "pdf_evidence": bool(chunks[r["bidNtceNo"]])}})
    return {"recommendations": out, "rag_chunks": chunks, "rag_status": "llm", "rag_chunk_search": chunk_search}
