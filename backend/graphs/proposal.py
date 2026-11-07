"""인터뷰(HITL ⓪) → 마스터키·검색 프롬프트 확인(HITL) → SC-001 → HITL ① → SC-002 그래프. Supervisor는 규칙 기반(비어 있는 단계로 분기).

Supervisor → 추천 Worker 서브그래프(필터 → 검색 → 리랭크) → RAG 순위·근거 → 검증 서브에이전트(상위 N건) → 선택 → 초안
Worker 노드는 backend/agents/에 에이전트별 파일로 둔다. 이 파일은 Supervisor와 그래프 조립만.
"""
from langgraph.graph import END, START, StateGraph

from backend.agents.draft import build_proposal_template, extract_criteria
from backend.agents.interview import interview, prepare_interview, prepare_master, review_master
from backend.agents.rag import rag_recommend
from backend.agents.selection import select_tender
from backend.agents.state import ProposalState
from backend.agents.verify import verify_recommendations
from backend.graphs.recommend import build_recommend_worker

MAX_ROUTES = 12

WORKERS = {
    "prepare_interview": prepare_interview,
    "interview": interview,
    "prepare_master": prepare_master,
    "review_master": review_master,
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
    if "master_draft" not in state:
        return "prepare_master"
    if "master" not in state["prefs"]:
        return "review_master"
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
