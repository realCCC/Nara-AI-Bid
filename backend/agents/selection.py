"""공고 선택 Worker (HITL ①): 추천 목록을 보여 주고 사용자가 고른 공고를 받는다."""
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from backend.agents.state import ProposalState, save


def select_tender(state: ProposalState, config: RunnableConfig):
    # 재개 값: 공고번호, 또는 {"bidNtceNo", "checks"(담당자가 확인한 체크리스트 항목)} — 초안에 확인한 항목 그대로 싣는다
    picked = interrupt({"recommendations": state["recommendations"]})
    tender_id, checks = (picked.get("bidNtceNo"), picked.get("checks")) if isinstance(picked, dict) else (picked, None)
    if tender_id not in {r["bidNtceNo"] for r in state["recommendations"]}:
        raise ValueError(f"추천 목록에 없는 공고입니다: {tender_id}")
    save(config, state, "selected", selection=tender_id)
    return {"selected_tender_id": tender_id, **({"proposal_checks": checks} if checks is not None else {})}
