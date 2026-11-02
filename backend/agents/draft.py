"""제안서 초안 Worker (SC-002): 평가 기준·과업 범위 추출 → 섹션별 LLM 초안(업체 문서 검색) → 초안 HTML 조립."""
from langchain_core.runnables import RunnableConfig

from backend import config as settings
from backend.agents.state import ProposalState, save, tenders
from backend.tools import checklist, llm, proposal, proposal_writer
from backend.tools.pdf import tender_chunks
from backend.tools.retrieval import retrieve

CRITERIA_QUERY = "평가 항목 배점 기술 능력 평가 세부 평가 기준 과업 기간 계약 기간"
TASK_QUERY = "과업 내용 과업 범위 과업 지시서 주요 과업 수행 내용 사업 목적"


def extract_criteria(state: ProposalState):
    tender = tenders(state)[state["selected_tender_id"]]
    extracted, chunks = None, []
    if settings.llm_enabled():
        chunks = tender_chunks(tender)
        no = tender["bidNtceNo"]
        hits = retrieve(chunks, CRITERIA_QUERY, k=8, tender_no=no)
        seen = {(c["file"], c["page"], c["text"]) for c in hits}
        hits += [c for c in retrieve(chunks, TASK_QUERY, k=5, tender_no=no) if (c["file"], c["page"], c["text"]) not in seen]
        extracted = llm.extract_criteria(tender, hits)
    return {"criteria": proposal.normalize_criteria(extracted, tender, chunks)}


def build_proposal_template(state: ProposalState, config: RunnableConfig):
    tender = tenders(state)[state["selected_tender_id"]]
    rec = next((r for r in state.get("recommendations", []) if r["bidNtceNo"] == tender["bidNtceNo"]), {})
    ai = proposal_writer.write(state["company_profile"], tender, state["criteria"])  # LLM 없으면 None → [확인 필요]
    # 담당자가 확인한 체크리스트가 있으면 그대로 (다시 만들면 LLM 재호출로 항목이 달라질 수 있음)
    checks = state.get("proposal_checks") or checklist.build(tender, rec)["items"]
    html = proposal.assemble_proposal(state["company_profile"], tender, state["criteria"], rec=rec, checks=checks, ai=ai)
    save(config, state, "done", proposal={"bidNtceNo": tender["bidNtceNo"], "criteria": state["criteria"], "html": html})
    return {"proposal_html": html, "proposal_ai": ai, "proposal_checks": checks}
