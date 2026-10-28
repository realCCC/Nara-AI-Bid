"""공고 선택 Worker (HITL ①): 추천 목록을 보여 주고 사용자가 고른 공고를 받는다."""
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from backend.agents.state import ProposalState, save


def select_tender(state: ProposalState, config: RunnableConfig):
    tender_id = interrupt({"recommendations": state["recommendations"]})
    if tender_id not in {r["bidNtceNo"] for r in state["recommendations"]}:
        raise ValueError(f"추천 목록에 없는 공고입니다: {tender_id}")
    save(config, state, "selected", selection=tender_id)
    return {"selected_tender_id": tender_id}
