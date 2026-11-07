"""SQLite: ① 로그인 성공·실패 ② 프로필 로드(해시 제외) ③ runs 저장·조회 + 그래프 E2E (룰 모드)."""
import pytest

from backend.db import repository as repo
from scripts.seed_companies import seed
from backend import config


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "test.db"
    ok, failed = seed(config.COMPANIES_DIR, path)
    assert ok >= 1 and not failed
    return path


def test_login(db):
    assert repo.login("C001", "1", db)["ok"]
    assert repo.login("C001", "wrong", db)["error"] == "비밀번호가 일치하지 않습니다."
    assert repo.login("C999", "1", db)["error"] == "존재하지 않는 업체 ID입니다."


def test_login_empty_db(tmp_path):
    repo.init_db(tmp_path / "empty.db")
    assert "미적재" in repo.login("C001", "1", tmp_path / "empty.db")["error"]


def test_profile_has_no_password(db):
    p = repo.load_company_profile("C001", db)
    assert p["회사명"] and p["업종"]["대분류"]
    assert "비밀번호" not in p and "password_hash" not in p


def test_seed_is_idempotent(db):
    n = repo.count_companies(db)
    seed(config.COMPANIES_DIR, db)
    assert repo.count_companies(db) == n


def test_runs(db):
    recs = [{"rank": 1, "bidNtceNo": "T1", "bidNtceNm": "공고1", "soft_score": 90, "reasons": [], "source": "rule"}]
    repo.save_run("th1", "C001", "recommended", recommendations=recs, db_path=db)
    repo.save_run("th1", "C001", "selected", selection="T1", db_path=db)
    repo.save_run("th2", "C002", "recommended", recommendations=recs, db_path=db)
    runs = repo.list_runs("C001", db)
    assert [(r["thread_id"], r["status"], r["title"]) for r in runs] == [("th1", "selected", "공고1")]
    assert repo.get_run("th1", db)["selection"]["bidNtceNo"] == "T1"


def test_graph_e2e_rule_mode(db, monkeypatch):
    """로그인 → 인터뷰(HITL ⓪) → 추천(HITL ①에서 멈춤) → 선택 재개 → 초안 + 이력 저장."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from backend.graphs.proposal import build_graph

    monkeypatch.setattr(config, "SQLITE_PATH", db)
    monkeypatch.setattr(config, "llm_enabled", lambda: False)
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "keyword")  # .env가 bge여도 모델 로드 없이
    graph = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "e2e"}}
    profile = repo.login("C001", "1", db)["profile"]

    graph.invoke({"company_profile": profile}, cfg)
    assert graph.get_state(cfg).next == ("interview",)
    graph.invoke(Command(resume={"answers": {}}), cfg)  # 전부 건너뜀 → 기본값
    assert graph.get_state(cfg).next == ("review_master",)
    state = graph.invoke(Command(resume={"key": "", "prompt": ""}), cfg)  # 비우면 첫 후보
    assert len(state["recommendations"]) == 10 and state["rag_status"] == "rule_only"
    assert state["verification"] == {"status": "skipped", "reason": "LLM 키 없음", "checked": 5, "changed": False,
                                     "flagged": 0, "issues_dropped": 0}
    assert state["search"]["path"] == "keyword_fallback" and state["recommendations"][0]["llm"] is None
    assert graph.get_state(cfg).next == ("select_tender",)

    pick = state["recommendations"][0]["bidNtceNo"]
    confirmed = [{"id": "c0", "category": "참가자격", "text": "담당자가 확인한 테스트 항목", "origin": "field"}]
    state = graph.invoke(Command(resume={"bidNtceNo": pick, "checks": confirmed}), cfg)
    assert state["selected_tender_id"] == pick
    assert state["proposal_checks"] == confirmed  # 확인한 체크리스트를 다시 만들지 않고 그대로 초안에
    assert "제안서 초안" in state["proposal_html"] and "[확인 필요]" in state["proposal_html"]
    assert repo.list_runs("C001", db)[0]["status"] == "done"
