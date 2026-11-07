"""로그 팝업: 업체별 조회·이어 읽기·노드 결과 요약."""
import contextvars

import pytest

from backend.db import vector_store
from backend.tools import activity, llm, retrieval


def test_logs_are_per_company_and_resume_after_id():
    activity.add("LOG-A", "로그인", "완료")
    activity.add("LOG-B", "로그인", "완료")
    activity.add("LOG-A", "AI 순위", "실패", "error")
    logs = activity.since("LOG-A")
    assert [e["stage"] for e in logs] == ["로그인", "AI 순위"] and logs[1]["level"] == "error"
    assert [e["stage"] for e in activity.since("LOG-A", logs[0]["id"])] == ["AI 순위"]
    assert "company" not in logs[0]


def test_note_needs_bound_company_and_follows_context():
    activity.note("LLM", "묶인 업체 없음")  # 스크립트·테스트에서는 남기지 않는다
    ctx = contextvars.copy_context()

    def run():
        activity.bind("LOG-D")
        with activity.step("LLM", "호출 중") as done:
            contextvars.copy_context().run(activity.note, "벡터DB", "조회")  # 작업 스레드로 복사된 컨텍스트
            done["message"] = "응답 완료"

    ctx.run(run)
    logs = activity.since("LOG-D")
    assert [e["stage"] for e in logs] == ["LLM", "벡터DB", "LLM"]
    assert logs[0]["message"] == "호출 중" and logs[2]["message"].startswith("응답 완료 (")


def test_bound_rebinds_each_next_in_fresh_context():
    def gen():
        yield activity._COMPANY.get()
        yield activity._COMPANY.get()

    it = activity.bound("LOG-E", gen())
    seen = [contextvars.copy_context().run(next, it) for _ in range(2)]  # 스트리밍처럼 매번 새 컨텍스트
    assert seen == ["LOG-E", "LOG-E"]


def test_step_logs_error_and_reraises():
    def run():
        activity.bind("LOG-F")
        with activity.step("RAG", "검색 중"):
            raise ValueError("문제")

    with pytest.raises(ValueError):
        contextvars.copy_context().run(run)
    assert activity.since("LOG-F")[-1]["level"] == "error"


def test_llm_call_is_logged_with_retry_and_failure(monkeypatch):
    class Fake:
        calls = 0

        def with_structured_output(self, schema):
            return self

        def with_config(self, **kw):
            return self

        def invoke(self, prompt):
            Fake.calls += 1
            if Fake.calls == 1:
                raise RuntimeError("일시 오류")
            return "ok"

    monkeypatch.setattr(llm, "get_llm", lambda: Fake())

    def run():
        activity.bind("LOG-G")
        return llm._invoke(llm.RankResult, "p")

    assert contextvars.copy_context().run(run) == "ok"
    logs = activity.since("LOG-G")
    assert [e["level"] for e in logs] == ["info", "warn", "info"] and "순위·근거 작성" in logs[0]["message"]


def test_vector_db_failure_is_logged_once_per_retry_window(monkeypatch):
    monkeypatch.setattr(vector_store, "_CONN", None)
    monkeypatch.setattr(vector_store, "_ERROR", None)
    monkeypatch.setattr(vector_store.config, "PGVECTOR_DSN", "postgresql://x:x@127.0.0.1:1/x")

    def run():
        activity.bind("LOG-H")
        for _ in range(3):
            with pytest.raises(RuntimeError):
                vector_store.load("1", ["k"])

    contextvars.copy_context().run(run)
    logs = activity.since("LOG-H")
    assert len(logs) == 1 and logs[0]["stage"] == "벡터DB" and logs[0]["level"] == "warn"


def test_single_text_embedding_is_not_logged(monkeypatch):
    monkeypatch.setattr(retrieval, "_vectors", lambda texts: texts)

    def run():
        activity.bind("LOG-I")
        retrieval.encode(["검색 문장"])
        retrieval.encode(["청크 1", "청크 2"])

    contextvars.copy_context().run(run)
    logs = activity.since("LOG-I")
    assert len(logs) == 2 and "텍스트 2개" in logs[0]["message"]


def test_buffer_keeps_latest_only():
    for i in range(activity.MAX_LOGS + 5):
        activity.add("LOG-C", "반복", str(i))
    logs = activity.since("LOG-C")
    assert len(logs) == activity.MAX_LOGS and logs[-1]["message"] == str(activity.MAX_LOGS + 4)


def test_fallbacks_are_warnings():
    assert activity.node_log("search", {"search": {"path": "keyword_fallback", "reason": "임베딩 미사용"}})[2] == "warn"
    assert activity.node_log("rerank", {"search": {"rerank": "skipped"}, "top10": []})[2] == "warn"
    assert activity.node_log("rag_recommend", {"recommendations": [], "rag_status": "rule_fallback"})[2] == "warn"
    assert activity.node_log("verify_recommendations", {"verification": {"status": "skipped", "reason": "x"}})[2] == "warn"


def test_success_summaries():
    stage, msg, level = activity.node_log("search", {"search": {"path": "semantic", "master": {"key": "축제", "hits": 7}}})
    assert (stage, level) == ("공고 검색", "info") and "일치 7건" in msg
    assert "1건" in activity.node_log("rag_recommend", {"recommendations": [{}], "rag_status": "llm"})[1]
    assert activity.node_log("supervisor", {}) is None and activity.node_log("search", None) is None
