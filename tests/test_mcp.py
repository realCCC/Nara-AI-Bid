"""MCP 서버 도구 (uv sync --extra mcp 일 때만). 데이터·규칙은 가짜로 바꿔 끼워 도구 입출력만 본다."""
import pytest

pytest.importorskip("mcp")

from backend import mcp_server as srv  # noqa: E402

COMPANY = {"업체ID": "C001", "회사명": "테스트", "소재지": "서울", "중소기업여부": True, "역량키워드": ["행사"],
           "업종": {"대분류": ["행사"], "세부분류번호": []}, "참여가능금액": {"최소": 0, "최대": 100}}
TENDER = {"bidNtceNo": "T1", "bidNtceNm": "축제 운영", "budget": 50, "pubPrcrmntLrgClsfcNm": "행사",
          "rgnLmtBidLocplcJdgmBssNm": "부산", "pdfs": [{"name": "공고.pdf"}]}


@pytest.fixture(autouse=True)
def fake_data(monkeypatch):
    monkeypatch.setattr(srv, "_companies", lambda: {"C001": COMPANY})
    monkeypatch.setattr(srv, "tenders_by_no", lambda: {"T1": TENDER})
    monkeypatch.setattr(srv, "_region_limits", lambda: {"T1": ["부산"]})


def test_tools_registered():
    import asyncio

    names = {t.name for t in asyncio.run(srv.mcp.list_tools())}
    assert names == {"list_companies", "recommend_tenders", "check_qualification", "get_tender", "search_evidence",
                     "get_checklist"}


def test_check_qualification_reports_failed_rule():
    out = srv.check_qualification("c001", "T1")  # 업체ID 대소문자 무시
    assert out["passed"] is False and [r["rule"] for r in out["failed_rules"]] == ["H5"]


def test_unknown_ids_raise_tool_error():
    with pytest.raises(srv.ToolError, match="없는 업체ID"):
        srv.check_qualification("C999", "T1")
    with pytest.raises(srv.ToolError, match="없는 공고번호"):
        srv.get_tender("X")


def test_get_tender_and_list_companies():
    assert srv.get_tender("T1")["pdfs"] == ["공고.pdf"]
    assert srv.list_companies()[0]["업체ID"] == "C001"
