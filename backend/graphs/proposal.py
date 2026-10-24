"""인터뷰(HITL ⓪) → SC-001 → HITL ① → SC-002 그래프. Supervisor는 규칙 기반(비어 있는 단계로 분기).

Supervisor → 추천 Worker 서브그래프(필터 → 검색 → 리랭크) → RAG 순위·근거 → 검증 서브에이전트(상위 N건) → 선택 → 초안
"""
import logging
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from backend import config as settings
from backend.db import repository as repo
from backend.graphs.recommend import build_recommend_worker
from backend.tools import keyword_profile, llm, proposal
from backend.tools.evidence import check_evidence_sources
from backend.tools.interview import build_questions, clean_answers, expand_company, parse_note, to_prefs, with_note
from backend.tools.loaders import tenders_by_no
from backend.tools.pdf import tender_chunks
from backend.tools.retrieval import retrieve, retrieve_for_company
from backend.tools.rules import rule_reasons, soft_score

log = logging.getLogger(__name__)
MAX_ROUTES = 12


class ProposalState(TypedDict, total=False):
    company_profile: dict
    tenders_path: str | None
    interview_questions: list[dict]
    interview_answers: dict
    prefs: dict
    candidates: list[dict]
    search: dict
    filter_log: dict
    top10: list[dict]
    rag_chunks: dict[str, list[dict]]
    recommendations: list[dict]
    rag_status: str
    verification: dict
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


def prepare_interview(state: ProposalState):
    """업체정보로 문항을 미리 만든다(LLM). interrupt 재개 때 노드가 다시 실행돼도 LLM을 또 부르지 않도록 분리.
    키워드 의미 프로필도 여기서 미리 만들어 둔다 (추천 단계는 캐시를 쓴다)."""
    keyword_profile.build(state["company_profile"])
    return {"interview_questions": build_questions(state["company_profile"])}


def interview(state: ProposalState):  # HITL ⓪: 객관식 + 자유 입력 인터뷰. 입력에 interview_answers가 있으면 묻지 않음
    questions = state["interview_questions"]
    answers = state.get("interview_answers")
    if answers is None:
        # 재개 값은 {"answers": {...}} — 빈 dict를 그대로 넘기면 LangGraph가 인터럽트 ID 매핑으로 해석한다
        answers = (interrupt({"questions": questions}) or {}).get("answers")
    answers = answers if isinstance(answers, dict) else {}
    note = clean_answers(answers, questions).get("note")
    prefs = to_prefs(answers, questions=questions, note_effect=parse_note(note, state["company_profile"]))
    return {"interview_answers": prefs["answers"], "prefs": prefs}


def _with_rule(base: dict) -> dict:
    """응답 영역 분리: rule(점수표·룰 근거) / llm(LLM 근거·플래그, 없으면 None)."""
    return {**base, "rule": {"soft_score": base["soft_score"], "score_detail": base.get("score_detail"),
                             "reasons": base["reasons"]}, "llm": None}


def rag_recommend(state: ProposalState):
    company, top, tenders = state["company_profile"], state["top10"], _tenders(state)
    if not top:
        return {"recommendations": [], "rag_chunks": {}, "rag_status": "no_candidates"}
    recs = [_with_rule(c) for c in top]
    if not settings.llm_enabled():
        return {"recommendations": recs, "rag_chunks": {}, "rag_status": "rule_only"}
    search_profile = expand_company(company, state.get("prefs"))  # 청크 검색에는 인터뷰 확장어도 쓴다
    chunks = {c["bidNtceNo"]: retrieve_for_company(tender_chunks(tenders[c["bidNtceNo"]]), search_profile) for c in top}
    ranked = llm.rag_rank(with_note(company, state.get("prefs")), top, chunks, tenders)  # 자유 입력 원문도 판단에
    if ranked is None:  # 출력 검증 실패 → 룰 순위로 대체
        return {"recommendations": recs, "rag_chunks": chunks, "rag_status": "rule_fallback"}
    by_no = {c["bidNtceNo"]: c for c in recs}
    out = []
    for i, r in enumerate(ranked, 1):
        base, t = by_no[r["bidNtceNo"]], tenders[r["bidNtceNo"]]
        llm_reasons = check_evidence_sources(r["reasons"], t, company, chunks[r["bidNtceNo"]])
        shown = list(llm_reasons)
        if len(shown) < 2:  # 출처 확인된 근거가 2개 미만이면 룰 근거 보충
            shown += rule_reasons(t, company, soft_score(t, company))[: 3 - len(shown)]
        out.append({**base, "rank": i, "reasons": shown, "source": "rag",
                    "qualification_flags": r.get("qualification_flags", []),
                    "pdf_evidence": bool(chunks[r["bidNtceNo"]]),
                    "llm": {"rank": i, "reasons": llm_reasons, "dropped": len(r["reasons"]) - len(llm_reasons),
                            "supplemented": len(shown) - len(llm_reasons),
                            "qualification_flags": r.get("qualification_flags", []),
                            "pdf_evidence": bool(chunks[r["bidNtceNo"]])}})
    return {"recommendations": out, "rag_chunks": chunks, "rag_status": "llm"}


def apply_verification(recs: list[dict], result: dict | None, reason: str | None, n: int) -> tuple[list[dict], dict]:
    """검증 결과 반영. 실패(result None)면 원순위 그대로 + 사유."""
    head = recs[:n]
    if result is None:
        return recs, {"status": "skipped", "reason": reason, "checked": len(head), "changed": False}
    by_no, items = {c["bidNtceNo"]: c for c in head}, {i["bidNtceNo"]: i for i in result["items"]}
    new_head = [{**by_no[no], "verify": {"fit_score": items[no]["fit_score"], "comment": items[no]["comment"],
                                         "rank_before": by_no[no]["rank"]}} for no in result["order"]]
    out = [{**r, "rank": i} for i, r in enumerate(new_head + recs[n:], 1)]
    changed = [c["bidNtceNo"] for c in head] != result["order"]
    return out, {"status": "passed", "reason": None, "checked": len(head), "changed": changed}


def verify_recommendations(state: ProposalState, config: RunnableConfig):  # 검증 서브에이전트
    recs, n = state["recommendations"], max(1, min(settings.VERIFY_TOP_N, 5))
    if not settings.llm_enabled():
        result, reason = None, "LLM 키 없음"
    else:
        try:
            result, reason = llm.verify_ranking(with_note(state["company_profile"], state.get("prefs")), recs[:n],
                                                _tenders(state))
        except Exception as e:  # 서브에이전트 예외는 그래프를 멈추지 않는다
            result, reason = None, f"예외: {type(e).__name__}"
    recs, verification = apply_verification(recs, result, reason, n)
    _save(config, state, "recommended", recommendations=recs)
    return {"recommendations": recs, "verification": verification}


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
    "prepare_interview": prepare_interview,
    "interview": interview,
    "recommend_worker": build_recommend_worker(ProposalState),
    "rag_recommend": rag_recommend,
    "verify_recommendations": verify_recommendations,
    "select_tender": select_tender,
    "extract_criteria": extract_criteria,
    "build_proposal_template": build_proposal_template,
}


def supervisor(state: ProposalState):
    return {"routes": state.get("routes", 0) + 1}


def route(state: ProposalState) -> str:  # 규칙 기반: 비어 있는 단계로
    if state.get("routes", 0) > MAX_ROUTES:
        return END
    if "interview_questions" not in state:
        return "prepare_interview"
    if "prefs" not in state:
        return "interview"
    if "top10" not in state:
        return "recommend_worker"
    if "recommendations" not in state:
        return "rag_recommend"
    if not state["recommendations"]:
        return END
    if "verification" not in state:
        return "verify_recommendations"
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
