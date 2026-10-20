"""SC-001 → HITL ① → SC-002 그래프. Supervisor는 규칙 기반(비어 있는 단계로 분기)."""
import logging
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app import config as settings
from app.db import repository as repo
from app.tools import llm, proposal
from app.tools.evidence import check_evidence_sources
from app.tools.loaders import load_region_limits, tenders_by_no
from app.tools.pdf import tender_chunks
from app.tools.retrieval import retrieve, retrieve_for_company
from app.tools.rules import recommend_by_rules, rule_reasons, soft_score

log = logging.getLogger(__name__)
MAX_ROUTES = 12


class ProposalState(TypedDict, total=False):
    company_profile: dict
    tenders_path: str | None
    filter_log: dict
    top10: list[dict]
    rag_chunks: dict[str, list[dict]]
    recommendations: list[dict]
    rag_status: str
    selected_tender_id: str | None
    criteria: dict
    proposal_html: str
    routes: int


def _thread_id(config: RunnableConfig) -> str | None:
    return (config or {}).get("configurable", {}).get("thread_id")


def _save(config, state, status, **kw):
    """SQLite 쓰기 실패는 로그만 남기고 그래프는 계속 (이력 저장은 보조 기능)."""
    tid = _thread_id(config)
    if not tid:
        return
    try:
        repo.save_run(tid, state["company_profile"]["업체ID"], status, **kw)
    except Exception as e:
        log.warning("이력 저장 실패: %s", e)


def _tenders(state) -> dict[str, dict]:
    return tenders_by_no(state.get("tenders_path"))


def filter_and_score(state: ProposalState):
    result = recommend_by_rules(list(_tenders(state).values()), state["company_profile"], load_region_limits())
    return {"top10": result["top"],
            "filter_log": {"funnel": result["funnel"], "rule_counts": result["rule_counts"], "excluded": result["excluded"]}}


def rag_recommend(state: ProposalState, config: RunnableConfig):
    company, top, tenders = state["company_profile"], state["top10"], _tenders(state)
    if not top:
        return {"recommendations": [], "rag_chunks": {}, "rag_status": "no_candidates"}
    if not settings.llm_enabled():
        recs, status, chunks = top, "rule_only", {}
    else:
        chunks = {c["bidNtceNo"]: retrieve_for_company(tender_chunks(tenders[c["bidNtceNo"]]), company) for c in top}
        ranked = llm.rag_rank(company, top, chunks, tenders)
        if ranked is None:  # 출력 검증 실패 → 룰 순위로 대체
            recs, status = top, "rule_fallback"
        else:
            by_no = {c["bidNtceNo"]: c for c in top}
            recs = []
            for i, r in enumerate(ranked, 1):
                base, t = by_no[r["bidNtceNo"]], tenders[r["bidNtceNo"]]
                verified = check_evidence_sources(r["reasons"], t, company, chunks[r["bidNtceNo"]])
                if len(verified) < 2:  # 출처 확인된 근거가 2개 미만이면 룰 근거 보충
                    verified += rule_reasons(t, company, soft_score(t, company))[: 3 - len(verified)]
                recs.append({**base, "rank": i, "reasons": verified, "source": "rag",
                             "qualification_flags": r.get("qualification_flags", []),
                             "pdf_evidence": bool(chunks[r["bidNtceNo"]])})
            status = "rag"
    _save(config, state, "recommended", recommendations=recs)
    return {"recommendations": recs, "rag_chunks": chunks, "rag_status": status}


def select_tender(state: ProposalState, config: RunnableConfig):  # HITL ①
    tender_id = interrupt({"recommendations": state["recommendations"]})
    if tender_id not in {r["bidNtceNo"] for r in state["recommendations"]}:
        raise ValueError(f"추천 목록에 없는 공고입니다: {tender_id}")
    _save(config, state, "selected", selection=tender_id)
    return {"selected_tender_id": tender_id}


def extract_criteria(state: ProposalState):
    tender = _tenders(state)[state["selected_tender_id"]]
    extracted = None
    if settings.llm_enabled():
        chunks = tender_chunks(tender)
        hits = retrieve(chunks, "평가 항목 배점 기술 능력 평가 과업 기간 계약 기간", k=8)
        extracted = llm.extract_criteria(tender, hits)
    return {"criteria": proposal.normalize_criteria(extracted, tender)}


def build_proposal_template(state: ProposalState, config: RunnableConfig):
    tender = _tenders(state)[state["selected_tender_id"]]
    html = proposal.assemble_proposal(state["company_profile"], tender, state["criteria"])
    _save(config, state, "done", proposal={"bidNtceNo": tender["bidNtceNo"], "criteria": state["criteria"], "html": html})
    return {"proposal_html": html}


WORKERS = {
    "filter_and_score": filter_and_score,
    "rag_recommend": rag_recommend,
    "select_tender": select_tender,
    "extract_criteria": extract_criteria,
    "build_proposal_template": build_proposal_template,
}


def supervisor(state: ProposalState):
    return {"routes": state.get("routes", 0) + 1}


def route(state: ProposalState) -> str:  # 규칙 기반: 비어 있는 단계로
    if state.get("routes", 0) > MAX_ROUTES:
        return END
    if "top10" not in state:
        return "filter_and_score"
    if "recommendations" not in state:
        return "rag_recommend"
    if not state["recommendations"]:
        return END
    if not state.get("selected_tender_id"):
        return "select_tender"
    if "criteria" not in state:
        return "extract_criteria"
    if not state.get("proposal_html"):
        return "build_proposal_template"
    return END


def build_graph(checkpointer=None):
    builder = StateGraph(ProposalState)
    builder.add_node("supervisor", supervisor)
    for name, fn in WORKERS.items():
        builder.add_node(name, fn)
        builder.add_edge(name, "supervisor")  # 작업 후 Supervisor로 복귀
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route, [*WORKERS, END])
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()  # langgraph dev 진입점 (서버가 checkpointer 주입)
