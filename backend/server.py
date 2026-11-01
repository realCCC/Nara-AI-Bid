"""PoC 서버: 화면(frontend/index.html) + API. 그래프는 프로세스 내 checkpointer로 HITL에서 멈췄다 재개한다.

실행: uv run uvicorn backend.server:app --port 8000  → http://localhost:8000
"""
import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from backend import config
from backend.db import repository as repo
from backend.graphs.proposal import build_graph
from backend.tools import checklist, proposal, proposal_chat
from backend.tools.pdf import fetch_tender_pdfs
from backend.tools.loaders import tenders_by_no
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


class ChatIn(BaseModel):
    token: str
    thread_id: str
    message: str


class ApplyIn(BaseModel):
    token: str
    thread_id: str
    keys: list[str]  # 적용할 변경안의 칸 key (직전 대화 응답의 변경안 중에서)


class ChecklistIn(BaseModel):
    token: str
    thread_id: str
    bidNtceNo: str


# 담당자 확인 체크리스트: (업체 ID, 공고번호) → 만든 체크리스트. 같은 공고를 새 작업으로 다시 골라도 그대로 쓴다
CHECKLISTS: dict[tuple[str, str], dict] = {}
# 대화형 보완: thread_id → 제안서 렌더링 재료(company·tender·criteria·rec·checks·ai) + 대화 기록 + 대기 중 변경안
PROPOSALS: dict[str, dict] = {}
CHAT_MAX = 500
# 추천 진행 상황: thread_id → 끝난 노드·단계별 숫자. 화면이 1초마다 읽어 단계 카드를 하나씩 채운다
PROGRESS: dict[str, dict] = {}


def _track(thread_id: str, node: str, update) -> None:
    """그래프 스트림의 노드 완료 이벤트를 화면용 진행 정보로 요약한다."""
    p, info = PROGRESS[thread_id], PROGRESS[thread_id]["info"]
    p["done"].append(node)
    if not isinstance(update, dict):
        return
    if node == "filter_and_score":
        f = update["filter_log"]["funnel"]
        info |= {"total": f["total"], "hard_pass": f["hard_pass"]}
    elif node == "search":
        info["path"] = update["search"].get("path")
    elif node == "rerank":
        info |= {"top": len(update["top10"]), "rerank": update["search"].get("rerank")}
    elif node == "rag_recommend":
        info |= {"rag_status": update["rag_status"], "recs": len(update["recommendations"])}


def _brief(items: list[dict], n: int = 10) -> list[dict]:
    """화면 스트리밍용 공고 요약 (중간 결과 목록)."""
    return [{"bidNtceNo": c["bidNtceNo"], "bidNtceNm": c.get("bidNtceNm"), "budget": c.get("budget"),
             "soft_score": c.get("soft_score"), "rank": c.get("rank")} for c in items[:n]]


def _step_event(node: str, update) -> dict | None:
    """노드 완료 이벤트 → 화면에 바로 보여 줄 그 단계의 중간 결과. 화면에 쓸 게 없는 노드는 None."""
    if not isinstance(update, dict):
        return None
    if node == "interview":
        return {"node": node, "conditions": len((update.get("prefs") or {}).get("answers") or {})}
    if node == "filter_and_score":
        f, sem = update["filter_log"]["funnel"], update["search"]["semantic"]
        return {"node": node, "total": f["total"], "hard_pass": f["hard_pass"], "top": _brief(update["candidates"], 5),
                "keywords": sem.get("keywords") if sem.get("used") else []}
    if node == "search":
        s, by_no = update["search"], tenders_by_no()
        top = [{"bidNtceNo": no, "bidNtceNm": by_no[no]["bidNtceNm"], "rrf": s["scores"][no]["rrf"]} for no in s["order"][:5]]
        return {"node": node, "path": s.get("path"), "query": s.get("query"), "query_source": s.get("query_source"),
                "lists": s.get("lists"), "top": top}
    if node == "rerank":
        return {"node": node, "rerank": update["search"].get("rerank"), "top": _brief(update["top10"])}
    if node == "rag_recommend":
        return {"node": node, "rag_status": update["rag_status"], "top": _brief(update["recommendations"])}
    if node == "verify_recommendations":
        return {"node": node, "verification": update["verification"], "top": _brief(update["recommendations"])}
    return None


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(jsonable_encoder(data), ensure_ascii=False)}\n\n"


def _company(token: str) -> dict:
    cid = SESSIONS.get(token)
    profile = repo.load_company_profile(cid) if cid else None
    if not profile:
        raise HTTPException(401, "로그인이 필요합니다.")
    return profile


def _cfg(thread_id: str, company: dict | None = None) -> dict:
    """그래프 실행 설정. company를 주면 LangSmith 트레이스에 업체ID 메타데이터·태그를 단다."""
    cfg = {"configurable": {"thread_id": thread_id}}
    if company:
        cfg |= {"run_name": "proposal_graph", "tags": [company["업체ID"]], "metadata": {"업체ID": company["업체ID"]}}
    return cfg


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/status")
def status():
    return {"llm": config.llm_enabled(), "companies": repo.count_companies(), "embedding": config.EMBEDDING_PROVIDER,
            "embedding_enabled": config.embedding_enabled(),
            "embedding_model": config.EMBEDDING_MODEL if config.embedding_enabled() else None,
            "chat_model": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") if config.llm_enabled() else None,
            "rerank_model": config.RERANK_MODEL if config.embedding_enabled() else None,
            "body_search": config.BODY_SEARCH}


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
    graph.invoke({"company_profile": company}, _cfg(thread_id, company))
    tasks = graph.get_state(_cfg(thread_id)).tasks
    questions = next((i.value["questions"] for t in tasks for i in t.interrupts), [])
    return {"thread_id": thread_id, "questions": questions}


def _check_interview(body: InterviewIn) -> dict:
    company = _owned(body.token, body.thread_id)
    if graph.get_state(_cfg(body.thread_id)).next != ("interview",):
        raise HTTPException(409, "이미 답변한 인터뷰입니다. 추천을 다시 실행하세요.")
    return company


def _resume_interview(body: InterviewIn, company: dict) -> Iterator[tuple[str, dict]]:
    """인터뷰 답변으로 재개 → 추천 Worker·RAG·검증 → HITL ①에서 멈춤. 노드가 끝날 때마다 (노드, 갱신값)을 낸다."""
    PROGRESS[body.thread_id] = {"company": company["업체ID"], "done": [], "info": {}}
    # invoke 대신 stream: 서브그래프(필터→검색→리랭크) 노드까지 끝날 때마다 진행 상황을 남긴다
    for _ns, chunk in graph.stream(Command(resume={"answers": {k: v for k, v in body.answers.items() if v}}),
                                   _cfg(body.thread_id, company), stream_mode="updates", subgraphs=True):
        for node, update in chunk.items():
            if not node.startswith("__"):
                _track(body.thread_id, node, update)
                yield node, update


@app.post("/api/interview")
def answer_interview(body: InterviewIn):
    for _ in _resume_interview(body, _check_interview(body)):
        pass
    return _interview_result(body.thread_id)


@app.post("/api/interview/stream")
def answer_interview_stream(body: InterviewIn):
    """/api/interview와 같은 실행을 SSE로: 단계가 끝날 때마다 step 이벤트(중간 결과), 마지막에 result 이벤트."""
    company = _check_interview(body)  # 권한·중복 오류는 스트림 시작 전에 HTTP 오류로

    def events():
        try:
            for node, update in _resume_interview(body, company):
                if (ev := _step_event(node, update)) is not None:
                    yield _sse("step", ev)
            yield _sse("result", _interview_result(body.thread_id))
        except Exception as e:  # 스트림 도중 실패 → 화면에 오류 표시
            yield _sse("error", {"detail": str(e)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _interview_result(thread_id: str) -> dict:
    state = graph.get_state(_cfg(thread_id)).values
    log = state["filter_log"]
    prefs = state["prefs"]
    return {
        "thread_id": thread_id,
        "funnel": log["funnel"],
        "rule_counts": [{"rule": RULE_LABELS[r], "desc": RULES[r], "count": n} for r, n in log["rule_counts"].items()],
        "interview": {"answers": prefs["answers"], "weights": prefs["weights"], "base_weights": prefs["base_weights"],
                      "focus": prefs["focus"], "expanded": prefs["expanded"], "exclude": prefs["exclude"],
                      "note": prefs["note"], "note_effect": prefs["note_effect"]},
        "search": state.get("search"),
        "rag_status": state.get("rag_status"),
        "rag_chunk_search": state.get("rag_chunk_search"),
        "verification": state.get("verification"),
        "recommendations": state.get("recommendations", []),
    }


@app.get("/api/progress")
def progress(token: str, thread_id: str):
    """추천 진행 상황 (끝난 노드 목록과 단계별 숫자)."""
    company = _company(token)
    p = PROGRESS.get(thread_id)
    if p is None or p["company"] != company["업체ID"]:
        return {"done": [], "info": {}}
    return {"done": p["done"], "info": p["info"]}


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
        state = graph.invoke(Command(resume=body.bidNtceNo), _cfg(body.thread_id, company))
    except ValueError as e:
        raise HTTPException(400, str(e))
    repo.save_checklist(body.thread_id, body.bidNtceNo, made)
    PROPOSALS.pop(body.thread_id, None)  # 새 초안 → 이전 대화·변경안은 버린다
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
    html = proposal.to_download_html(run["proposal"]["proposal_html"], tender["bidNtceNm"])
    return HTMLResponse(html, headers={"Content-Disposition": f'attachment; filename="proposal_{tender["bidNtceNo"]}.html"'})


# ---------- 공고서 원문 보기 ----------
def _tender_pdfs(no: str) -> list[dict]:
    tender = tenders_by_no().get(no)
    if tender is None:
        raise HTTPException(404, "공고가 없습니다.")
    names = {f["name"]: i for i, f in enumerate(tender["pdfs"], 1)}
    return [{"idx": names[f["file"]], "name": f["file"]} for f in fetch_tender_pdfs(tender)]


@app.get("/api/tender/{no}/files")
def tender_files(no: str, token: str):
    _company(token)
    return {"files": _tender_pdfs(no)}


@app.get("/api/tender/{no}/pdf/{idx}")
def tender_pdf(no: str, idx: int, token: str):
    _company(token)
    if idx not in {f["idx"] for f in _tender_pdfs(no)}:
        raise HTTPException(404, "공고서 파일이 없습니다.")
    return FileResponse(config.PDF_DIR / no / f"{idx}.pdf", media_type="application/pdf",
                        headers={"Content-Disposition": "inline"})


# ---------- 제안서 대화형 보완 ----------
def _proposal_ctx(token: str, thread_id: str) -> dict:
    company = _owned(token, thread_id)
    if thread_id not in PROPOSALS:
        v = graph.get_state(_cfg(thread_id)).values
        if not v.get("proposal_html"):
            raise HTTPException(400, "제안서 초안을 먼저 만드세요.")
        tender = tenders_by_no()[v["selected_tender_id"]]
        rec = next((r for r in v.get("recommendations", []) if r["bidNtceNo"] == tender["bidNtceNo"]), {})
        PROPOSALS[thread_id] = {"company": v["company_profile"], "tender": tender, "criteria": v["criteria"], "rec": rec,
                                "checks": v.get("proposal_checks") or [], "ai": v.get("proposal_ai") or {},
                                "chat": [], "pending": {}}
    ctx = PROPOSALS[thread_id]
    if ctx["company"]["업체ID"] != company["업체ID"]:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return ctx


def _chat_message(body: ChatIn) -> str:
    message = body.message.strip()[:CHAT_MAX]
    if not message:
        raise HTTPException(400, "요청 내용을 입력하세요.")
    return message


def _chat_finish(ctx: dict, message: str, result: dict) -> dict:
    """대화 기록에 남기고 변경안을 대기시킨 뒤 화면용 응답으로."""
    ctx["chat"] += [{"role": "user", "text": message}, {"role": "ai", "text": result["reply"]}]
    ctx["pending"] = {e["key"]: e for e in result["edits"]}
    return {"reply": result["reply"], "skipped": result["skipped"],
            "edits": [{k: e[k] for k in ("key", "label", "before", "after", "source", "grounded")} for e in result["edits"]]}


@app.post("/api/proposal/chat")
def proposal_chat_api(body: ChatIn):
    """사용자 요청 → 칸 단위 변경안 (적용 전). 변경안은 서버에 대기시키고 /api/proposal/apply로 고른 것만 반영."""
    ctx, message = _proposal_ctx(body.token, body.thread_id), _chat_message(body)
    return _chat_finish(ctx, message, proposal_chat.chat(ctx, message, ctx["chat"]))


@app.post("/api/proposal/chat/stream")
def proposal_chat_stream(body: ChatIn):
    """/api/proposal/chat과 같은 처리를 SSE로: 단계(불러오기·근거 검색·AI 작성·검증)마다 step, 마지막에 result."""
    ctx, message = _proposal_ctx(body.token, body.thread_id), _chat_message(body)  # 권한·입력 오류는 스트림 전에

    def events():
        try:
            for ev, data in proposal_chat.chat_steps(ctx, message, ctx["chat"]):
                yield _sse(ev, _chat_finish(ctx, message, data) if ev == "result" else data)
        except Exception as e:
            yield _sse("error", {"detail": str(e)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/proposal/apply")
def proposal_apply(body: ApplyIn):
    ctx = _proposal_ctx(body.token, body.thread_id)
    chosen = [ctx["pending"][k] for k in body.keys if k in ctx["pending"]]
    if not chosen:
        raise HTTPException(400, "적용할 변경안이 없습니다. 다시 요청해 주세요.")
    ctx["ai"], ctx["criteria"] = proposal_chat.apply(ctx, chosen)
    for e in chosen:
        ctx["pending"].pop(e["key"])
    html = proposal.assemble_proposal(ctx["company"], ctx["tender"], ctx["criteria"], rec=ctx["rec"],
                                      checks=ctx["checks"], ai=ctx["ai"])
    repo.save_run(body.thread_id, ctx["company"]["업체ID"], "done",
                  proposal={"bidNtceNo": ctx["tender"]["bidNtceNo"], "criteria": ctx["criteria"], "html": html})
    return {"proposal_html": html, "criteria": ctx["criteria"], "applied": [e["key"] for e in chosen]}
