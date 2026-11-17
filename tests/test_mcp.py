"""MCP 서버 도구 (uv sync --extra mcp 일 때만). 데이터·PDF 추출은 가짜로 바꿔 끼워 도구 입출력만 본다."""
import pytest

pytest.importorskip("mcp")

from backend import config  # noqa: E402
from backend import mcp_server as srv  # noqa: E402

COMPANY = {"소재지": "서울", "중소기업여부": True, "업종": {"대분류": ["행사"], "세부분류번호": []},
           "참여가능금액": {"최소": 0, "최대": 100}}
TENDER = {"bidNtceNo": "T1", "bidNtceNm": "축제 운영", "budget": 50, "pubPrcrmntLrgClsfcNm": "행사",
          "rgnLmtBidLocplcJdgmBssNm": "부산", "pdfs": [{"name": "공고.pdf"}]}
CHUNKS = ({"file": "공고.pdf", "page": 2, "pages": [2, 3], "section": "2. 입찰참가자격",
           "text": "가. 소프트웨어사업자로 신고한 업체 나. 공고일 기준 서울특별시에 주된 영업소를 둔 업체",
           "search_text": "입찰참가자격 소프트웨어사업자"},
          {"file": "공고.pdf", "page": 5, "pages": [5], "section": "5. 제출서류",
           "text": "제안서 10부와 가격 입찰서를 제출 마감 일시까지 제출", "search_text": "제출서류 제안서"})


@pytest.fixture(autouse=True)
def fake_data(monkeypatch):
    monkeypatch.setattr(srv, "tenders_by_no", lambda: {"T1": TENDER})
    monkeypatch.setattr(srv, "_region_limits", lambda: {"T1": ["부산"]})
    monkeypatch.setattr(srv, "_chunks", lambda pdf: ("공고.pdf", CHUNKS))
    monkeypatch.setattr(config, "CHUNK_EMBEDDING", False)


def test_tools_registered():
    import asyncio

    names = {t.name for t in asyncio.run(srv.mcp.list_tools())}
    assert names == {"check_eligibility", "extract_requirements", "verify_quote"}


def test_check_eligibility_uses_given_profile():
    out = srv.check_eligibility(COMPANY, "T1")
    assert out["passed"] is False and [r["rule"] for r in out["failed_rules"]] == ["H5"]
    assert srv.check_eligibility({**COMPANY, "소재지": "부산"}, "T1")["passed"] is True


def test_unknown_tender_raises_tool_error():
    with pytest.raises(srv.ToolError, match="없는 공고번호"):
        srv.check_eligibility(COMPANY, "X")


def test_verify_quote_ignores_spaces_and_checks_page():
    out = srv.verify_quote("서울특별시에 주된  영업소를 둔 업체", "공고.pdf")
    assert out["found"] is True and out["pages"] == [2]
    assert srv.verify_quote("서울특별시에 주된 영업소", "공고.pdf", page=3)["found"] is True  # 걸친 페이지도 통과
    assert srv.verify_quote("서울특별시에 주된 영업소", "공고.pdf", page=5)["found"] is False
    assert srv.verify_quote("부산광역시에 주된 영업소", "공고.pdf")["found"] is False


def test_verify_quote_ellipsis_parts_must_share_chunk():
    assert srv.verify_quote("소프트웨어사업자로 … 주된 영업소", "공고.pdf")["found"] is True
    assert srv.verify_quote("소프트웨어사업자로 … 제안서 10부", "공고.pdf")["found"] is False
    with pytest.raises(srv.ToolError, match="너무 짧"):
        srv.verify_quote("업체", "공고.pdf")


def test_extract_requirements_drops_unverified_quotes(monkeypatch):
    from backend.tools import llm

    monkeypatch.setattr(config, "llm_enabled", lambda: True)
    monkeypatch.setattr(llm, "extract_checklist", lambda t, c: {"items": [
        {"category": "참가자격", "text": "소프트웨어사업자 신고", "file": "x.pdf", "page": 2, "quote": "소프트웨어사업자로 신고한 업체"},
        {"category": "실적요건", "text": "최근 3년 실적", "file": "공고.pdf", "page": 2, "quote": "최근 3년간 유사 실적"}]})
    out = srv.extract_requirements("공고.pdf")
    assert out["source"] == "llm" and [i["category"] for i in out["items"]] == ["참가자격"]
    assert out["dropped_unverified"] == 1


def test_extract_requirements_without_llm_returns_clauses(monkeypatch):
    monkeypatch.setattr(config, "llm_enabled", lambda: False)
    out = srv.extract_requirements("공고.pdf")
    assert out["source"] == "clauses" and out["items"] == [] and out["clauses"]
