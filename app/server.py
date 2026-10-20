"""PoC 서버: 화면(frontend/index.html) + API. 그래프는 프로세스 내 checkpointer로 HITL에서 멈췄다 재개한다.

실행: uv run uvicorn app.server:app --port 8000  → http://localhost:8000
"""
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from app import config
from app.db import repository as repo
from app.graphs.proposal import build_graph
from app.tools.loaders import tenders_by_no
from app.tools.proposal import to_download_html
from app.tools.rules import RULE_LABELS, RULES

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="나라장터 AI 입찰 어시스턴트 PoC")
graph = build_graph(InMemorySaver())
repo.init_db()

# PoC 간이 로그인: 서버 메모리에 토큰 → 업체 ID (만료·권한 없음, 4번 문서 4-2)
SESSIONS: dict[str, str] = {}


class LoginIn(BaseModel):
    company_id: str
    password: str


class SelectIn(BaseModel):
    token: str
    thread_id: str
    bidNtceNo: str


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
    return {"token": token, "profile": result["profile"]}


@app.post("/api/logout")
def logout(token: str):
    SESSIONS.pop(token, None)
    return {"ok": True}


@app.post("/api/recommend")
def recommend(token: str):
    company = _company(token)
    thread_id = uuid.uuid4().hex
    state = graph.invoke({"company_profile": company}, _cfg(thread_id))  # HITL ①에서 멈춤
    log = state["filter_log"]
    return {
        "thread_id": thread_id,
        "funnel": log["funnel"],
        "rule_counts": [{"rule": RULE_LABELS[r], "desc": RULES[r], "count": n} for r, n in log["rule_counts"].items()],
        "rag_status": state.get("rag_status"),
        "recommendations": state.get("recommendations", []),
    }


@app.post("/api/select")
def select(body: SelectIn):
    company = _company(body.token)
    snapshot = graph.get_state(_cfg(body.thread_id))
    if not snapshot.values or snapshot.values["company_profile"]["업체ID"] != company["업체ID"]:
        raise HTTPException(404, "작업을 찾을 수 없습니다. 추천을 다시 실행하세요.")
    try:
        state = graph.invoke(Command(resume=body.bidNtceNo), _cfg(body.thread_id))
    except ValueError as e:
        raise HTTPException(400, str(e))
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
