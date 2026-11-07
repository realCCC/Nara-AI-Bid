"""로그 팝업용 기록 (서버 메모리, 업체별로 조회). 주요 기능의 결과와 LLM·RAG·벡터DB·임베딩·리랭크 호출을 남긴다.

서버가 bind·bound로 업체를 묶으면 도구 코드는 note·step만 부른다. 묶인 업체가 없으면(스크립트·테스트) 아무것도 안 남긴다.
level: info 정상 · warn AI 대신 룰·기본값으로 대체 · error 실패
"""
import time
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from threading import Lock

MAX_LOGS = 300
_LOGS: deque[dict] = deque(maxlen=MAX_LOGS)
_LOCK = Lock()
_last_id = 0
_COMPANY: ContextVar[str | None] = ContextVar("activity_company", default=None)


def add(company_id: str, stage: str, message: str, level: str = "info") -> None:
    global _last_id
    with _LOCK:
        _last_id += 1
        _LOGS.append({"id": _last_id, "company": company_id, "time": datetime.now().strftime("%H:%M:%S"),
                      "stage": stage, "message": message, "level": level})


def bind(company_id: str) -> None:
    """이 실행 흐름의 로그를 업체에 묶는다. 그래프 작업 스레드·ContextThreadPoolExecutor에는 컨텍스트로 전달된다."""
    _COMPANY.set(company_id)


def bound(company_id: str, items: Iterable) -> Iterator:
    """스트리밍 응답은 next()마다 다른 컨텍스트에서 돌아서, 다음 값을 꺼내기 직전마다 다시 묶는다."""
    it = iter(items)
    while True:
        bind(company_id)
        try:
            item = next(it)
        except StopIteration:
            return
        yield item


def note(stage: str, message: str, level: str = "info") -> None:
    if company_id := _COMPANY.get():
        add(company_id, stage, message, level)


@contextmanager
def step(stage: str, start: str):
    """시작 줄과 결과 줄(소요 시간 포함)을 남긴다. 본문은 done["message"](·done["level"])에 결과를 적는다."""
    t0, done = time.perf_counter(), {}
    note(stage, start)
    try:
        yield done
    except Exception as e:
        note(stage, f"실패: {getattr(e, 'detail', None) or e}", "error")
        raise
    note(stage, f"{done.get('message', '완료')} ({time.perf_counter() - t0:.1f}초)", done.get("level", "info"))


def since(company_id: str, after: int = 0) -> list[dict]:
    with _LOCK:
        return [{k: v for k, v in e.items() if k != "company"} for e in _LOGS
                if e["company"] == company_id and e["id"] > after]


def node_log(node: str, update) -> tuple[str, str, str] | None:
    """추천 그래프 노드 결과 → (기능, 메시지, level). 로그로 남기지 않는 노드는 None."""
    if not isinstance(update, dict):
        return None
    if node == "review_master":
        prefs = update["prefs"]
        master = prefs["master"]
        if not master:
            return "마스터키", f"마스터키 없이 진행 · 인터뷰 답 {len(prefs['answers'])}개", "warn"
        kind = "담당자 수정" if master["edited"] else "자동 작성"
        return "마스터키", f"'{master['key']}' 확정 · 검색 프롬프트 {kind} · 인터뷰 답 {len(prefs['answers'])}개", "info"
    if node == "filter_and_score":
        f = update["filter_log"]["funnel"]
        return "자격 필터", f"공고 {f['total']}건 중 {f['hard_pass']}건 통과", "info"
    if node == "search":
        s = update["search"]
        hits = f" · 마스터키 일치 {s['master']['hits']}건 우선" if s.get("master") else ""
        if s.get("path") == "semantic":
            return "공고 검색", f"의미 검색 완료 (메모리 벡터+BM25){hits}", "info"
        return "공고 검색", f"키워드 검색으로 대체{hits} ({s.get('reason') or '사유 없음'})", "warn"
    if node == "rerank":
        s, n = update["search"], len(update["top10"])
        if s.get("rerank") == "applied":
            return "후보 선별", f"상위 {n}건 선별 · 리랭크 적용", "info"
        return "후보 선별", f"상위 {n}건 선별 · 리랭크 생략 ({s.get('rerank_reason') or '사유 없음'})", "warn"
    if node == "rag_recommend":
        n = len(update["recommendations"])
        if update["rag_status"] == "llm":
            return "AI 순위", f"{n}건 순위·근거 작성", "info"
        if update["rag_status"] == "no_candidates":
            return "AI 순위", "후보 공고 없음", "warn"
        return "AI 순위", f"룰 순위로 대체 ({update['rag_status']})", "warn"
    if node == "verify_recommendations":
        v = update["verification"]
        if v["status"] == "passed":
            return ("추천 검증", f"상위 {v['checked']}건 재평가 · {'순위 조정' if v['changed'] else '순위 유지'}"
                    f" · 자격 의심 {v['flagged']}건", "info")
        return "추천 검증", f"검증 생략 ({v.get('reason') or '사유 없음'})", "warn"
    return None
