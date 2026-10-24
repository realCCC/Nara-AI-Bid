"""PoC 서버: 화면(frontend/index.html) + API. 그래프는 프로세스 내 checkpointer로 HITL에서 멈췄다 재개한다.

실행: uv run uvicorn backend.server:app --port 8000  → http://localhost:8000
"""
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from backend import config
from backend.db import repository as repo
from backend.graphs.proposal import build_graph
from backend.tools import checklist
from backend.tools.loaders import tenders_by_no
from backend.tools.proposal import to_download_html
from backend.tools.rules import RULE_LABELS, RULES

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="나라장터 AI 입찰 어시스턴트 PoC")
graph = build_graph(InMemorySaver())
repo.init_db()

# PoC 간이 로그인: 서버 메모리에 토큰 → 업체 ID (만료·권한 없음, 4번 문서 4-2)
SESSIONS: dict[str, str] = {}


class LoginIn(BaseModel):
    company_id: str
    password: str


class InterviewIn(BaseModel):
    token: str
    thread_id: str
    answers: dict[str, str | list[str] | None] = {}  # 건너뛴 문항은 생략·null·빈 목록 → 기본값 "전체", note는 자유 입력


class SelectIn(BaseModel):
    token: str
    thread_id: str
    bidNtceNo: str
    checks: list[str] = []  # 담당자가 체크한 체크리스트 항목 id (전부 있어야 선택 확정)


class ChecklistIn(BaseModel):
    token: str
    thread_id: str
    bidNtceNo: str


# 담당자 확인 체크리스트: (업체 ID, 공고번호) → 만든 체크리스트. 같은 공고를 새 작업으로 다시 골라도 그대로 쓴다
CHECKLISTS: dict[tuple[str, str], dict] = {}


def _company(token: str) -> dict:
    cid = SESSIONS.get(token)
    profile = repo.load_company_profile(cid) if cid else None
    if not profile:
        raise HTTPException(401, "로그인이 필요합니다.")
    return profile


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/status")
def status():
    return {"llm": config.llm_enabled(), "companies": repo.count_companies(), "embedding": config.EMBEDDING_PROVIDER}


@app.post("/api/login")
def login(body: LoginIn):
    result = repo.login(body.company_id.strip(), body.password)
    if not result["ok"]:
        raise HTTPException(401, result["error"])
    token = uuid.uuid4().hex
    SESSIONS[token] = result["profile"]["업체ID"]
    return {"token": token, "profile": result["profile"], "code_names": _code_names()}


def _code_names() -> dict[str, str]:
    """조달 세부분류번호 → 세부분류명 (공고 데이터의 pubPrcrmntClsfcNo·Nm). 업체 업종·추천 근거 출처를 이름으로 보여 주는 데 쓴다."""
    return {t["pubPrcrmntClsfcNo"]: t["pubPrcrmntClsfcNm"] for t in tenders_by_no().values()
            if t.get("pubPrcrmntClsfcNo") and t.get("pubPrcrmntClsfcNm")}


@app.post("/api/logout")
def logout(token: str):
    SESSIONS.pop(token, None)
    return {"ok": True}


def _owned(token: str, thread_id: str) -> dict:
    company = _company(token)
    snapshot = graph.get_state(_cfg(thread_id))
    if not snapshot.values or snapshot.values["company_profile"]["업체ID"] != company["업체ID"]:
        raise HTTPException(404, "작업을 찾을 수 없습니다. 추천을 다시 실행하세요.")
    return company


@app.post("/api/recommend")
def recommend(token: str):
    """추천 시작 → 인터뷰(HITL ⓪)에서 멈추고 객관식 문항을 돌려준다."""
    company = _company(token)
    thread_id = uuid.uuid4().hex
    graph.invoke({"company_profile": company}, _cfg(thread_id))
    tasks = graph.get_state(_cfg(thread_id)).tasks
    questions = next((i.value["questions"] for t in tasks for i in t.interrupts), [])
    return {"thread_id": thread_id, "questions": questions}


@app.post("/api/interview")
def answer_interview(body: InterviewIn):
    """인터뷰 답변으로 재개 → 추천 Worker·RAG·검증 → HITL ①에서 멈춤."""
    _owned(body.token, body.thread_id)
    if graph.get_state(_cfg(body.thread_id)).next != ("interview",):
        raise HTTPException(409, "이미 답변한 인터뷰입니다. 추천을 다시 실행하세요.")
    state = graph.invoke(Command(resume={"answers": {k: v for k, v in body.answers.items() if v}}), _cfg(body.thread_id))
    log = state["filter_log"]
    prefs = state["prefs"]
    return {
        "thread_id": body.thread_id,
        "funnel": log["funnel"],
        "rule_counts": [{"rule": RULE_LABELS[r], "desc": RULES[r], "count": n} for r, n in log["rule_counts"].items()],
        "interview": {"answers": prefs["answers"], "weights": prefs["weights"], "base_weights": prefs["base_weights"],
                      "focus": prefs["focus"], "expanded": prefs["expanded"], "exclude": prefs["exclude"],
                      "note": prefs["note"], "note_effect": prefs["note_effect"]},
        "search": state.get("search"),
        "rag_status": state.get("rag_status"),
        "verification": state.get("verification"),
        "recommendations": state.get("recommendations", []),
    }


@app.post("/api/checklist")
def make_checklist(body: ChecklistIn):
    """HITL ①: 고른 공고의 확인 항목 (공고 필드 + AI가 공고서에서 뽑은 조건 + AI 자격 의심 사항)."""
    company = _owned(body.token, body.thread_id)
    recs = graph.get_state(_cfg(body.thread_id)).values.get("recommendations") or []
    rec = next((r for r in recs if r["bidNtceNo"] == body.bidNtceNo), None)
    if rec is None:
        raise HTTPException(400, f"추천 목록에 없는 공고입니다: {body.bidNtceNo}")
    result = checklist.build(tenders_by_no()[body.bidNtceNo], rec)
    CHECKLISTS[(company["업체ID"], body.bidNtceNo)] = result
    return result


@app.post("/api/select")
def select(body: SelectIn):
    company = _owned(body.token, body.thread_id)
    made = CHECKLISTS.get((company["업체ID"], body.bidNtceNo))
    if made is None:
        raise HTTPException(400, "담당자 확인 체크리스트를 먼저 확인하세요.")
    missing = [i["id"] for i in made["items"] if i["id"] not in set(body.checks)]
    if missing:
        raise HTTPException(400, f"확인하지 않은 항목이 {len(missing)}개 있습니다.")
    try:
        state = graph.invoke(Command(resume=body.bidNtceNo), _cfg(body.thread_id))
    except ValueError as e:
        raise HTTPException(400, str(e))
    repo.save_checklist(body.thread_id, body.bidNtceNo, made)
    tender = tenders_by_no()[body.bidNtceNo]
    return {"criteria": state["criteria"], "proposal_html": state["proposal_html"],
            "tender": {"bidNtceNo": tender["bidNtceNo"], "bidNtceNm": tender["bidNtceNm"], "budget": tender["budget"]}}


@app.get("/api/history")
def history(token: str):
    return {"runs": repo.list_runs(_company(token)["업체ID"])}


@app.get("/api/proposal/{thread_id}", response_class=HTMLResponse)
def download(thread_id: str, token: str):
    company = _company(token)
    run = repo.get_run(thread_id)
    if not run or run["run"]["company_id"] != company["업체ID"] or not run["proposal"]:
        raise HTTPException(404, "제안서가 없습니다.")
    tender = tenders_by_no()[run["proposal"]["bidNtceNo"]]
    html = to_download_html(run["proposal"]["proposal_html"], tender["bidNtceNm"])
    return HTMLResponse(html, headers={"Content-Disposition": f'attachment; filename="proposal_{tender["bidNtceNo"]}.html"'})
