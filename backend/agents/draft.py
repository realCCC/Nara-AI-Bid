"""제안서 초안 Worker (SC-002): 평가 기준 추출 → 초안 HTML 조립."""
from langchain_core.runnables import RunnableConfig

from backend import config as settings
from backend.agents.state import ProposalState, save, tenders
from backend.tools import llm, proposal
from backend.tools.pdf import tender_chunks
from backend.tools.retrieval import retrieve


def extract_criteria(state: ProposalState):
    tender = tenders(state)[state["selected_tender_id"]]
    extracted = None
    if settings.llm_enabled():
        chunks = tender_chunks(tender)
        hits = retrieve(chunks, "평가 항목 배점 기술 능력 평가 과업 기간 계약 기간", k=8, tender_no=tender["bidNtceNo"])
        extracted = llm.extract_criteria(tender, hits)
    return {"criteria": proposal.normalize_criteria(extracted, tender)}


def build_proposal_template(state: ProposalState, config: RunnableConfig):
    tender = tenders(state)[state["selected_tender_id"]]
    html = proposal.assemble_proposal(state["company_profile"], tender, state["criteria"])
    save(config, state, "done", proposal={"bidNtceNo": tender["bidNtceNo"], "criteria": state["criteria"], "html": html})
    return {"proposal_html": html}
