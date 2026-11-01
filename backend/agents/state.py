"""Supervisor·Worker가 공유하는 그래프 State와 공용 헬퍼."""
import logging
from typing import TypedDict

from langchain_core.runnables import RunnableConfig

from backend.db import repository as repo
from backend.tools.loaders import tenders_by_no

log = logging.getLogger(__name__)


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
    rag_chunk_search: dict  # {"paths": {공고번호: hybrid|bm25|none}, "query": 검색 문장}
    verification: dict
    selected_tender_id: str | None
    criteria: dict
    proposal_html: str
    proposal_ai: dict | None   # 섹션별 LLM 초안 (대화형 보완이 칸 단위로 고친 뒤 다시 렌더링)
    proposal_checks: list[dict]
    routes: int


def _thread_id(config: RunnableConfig) -> str | None:
    return (config or {}).get("configurable", {}).get("thread_id")


def save(config, state, status, **kw):
    """SQLite 쓰기 실패는 로그만 남기고 그래프는 계속 (이력 저장은 보조 기능)."""
    tid = _thread_id(config)
    if not tid:
        return
    try:
        repo.save_run(tid, state["company_profile"]["업체ID"], status, **kw)
    except Exception as e:
        log.warning("이력 저장 실패: %s", e)


def tenders(state) -> dict[str, dict]:
    return tenders_by_no(state.get("tenders_path"))
