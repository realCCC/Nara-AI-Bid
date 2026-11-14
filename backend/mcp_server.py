"""MCP 서버: 공고 추천·자격 확인·공고서 근거 검색을 MCP 도구로 연다.

웹 화면(FastAPI + LangGraph)과 같은 함수를 쓰므로, Claude Desktop 같은 MCP 클라이언트에서도
"C003 업체가 낼 만한 공고 찾아줘" → recommend_tenders, "이 공고 참가 자격 돼?" → check_qualification 처럼 부를 수 있다.
판단(대화·요약)은 클라이언트 LLM이 하고, 이 서버는 공고 데이터와 규칙·검색 결과만 돌려준다.

실행: uv sync --extra mcp  →  uv run --extra mcp python -m backend.mcp_server   (stdio)
"""
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from backend import config
from backend.graphs.recommend import run_pipeline
from backend.tools import checklist
from backend.tools.loaders import load_company_file, load_region_limits, tenders_by_no
from backend.tools.pdf import tender_chunks
from backend.tools.retrieval import retrieve
from backend.tools.rules import RULES, hard_filter, tender_info

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)  # stdio 전송: stdout은 MCP 메시지 전용

mcp = MCPServer("nara-bid-agent", instructions=(
    "나라장터 용역 공고(2026-05) 추천 도구. 업체ID(C001~C010)로 맞춤 공고를 찾고, 공고번호로 참가 자격·공고서 근거·"
    "확인 체크리스트를 조회한다. 결과의 근거(필드 값·공고서 인용)를 그대로 보여 주고, 최종 입찰 판단은 사용자에게 맡긴다."))

TENDER_FIELDS = ("bidNtceNo", "bidNtceNm", "budget", "pubPrcrmntLrgClsfcNm", "pubPrcrmntClsfcNm", "cntrctCnclsMthdNm",
                 "bidPrtcptLmtYn", "rgnLmtBidLocplcJdgmBssNm", "bidClseDt", "dminsttNm")


@lru_cache
def _companies() -> dict[str, dict]:
    out = {}
    for p in sorted(Path(config.COMPANIES_DIR).glob("*.json")):
        c = load_company_file(p)
        out[c["업체ID"]] = c
    return out


@lru_cache
def _region_limits() -> dict:
    return load_region_limits()


def _company(company_id: str) -> dict:
    c = _companies().get(company_id.strip().upper())
    if c is None:
        raise ToolError(f"없는 업체ID: {company_id} (가능: {', '.join(_companies())})")
    return c


def _tender(tender_no: str) -> dict:
    t = tenders_by_no().get(tender_no.strip())
    if t is None:
        raise ToolError(f"없는 공고번호: {tender_no}")
    return t


@mcp.tool()
def list_companies() -> list[dict]:
    """등록된 업체 목록 (업체ID·회사명·소재지·역량 키워드)."""
    return [{"업체ID": cid, "회사명": c.get("회사명"), "소재지": c.get("소재지"), "역량키워드": c.get("역량키워드", [])}
            for cid, c in _companies().items()]


@mcp.tool()
def recommend_tenders(company_id: str, top_k: int = 5) -> dict:
    """업체에 맞는 공고 추천: 자격 필터(H1~H6) → 의미 검색(BM25+임베딩+RRF) → 리랭크. LLM 순위·검증 전 단계 결과.

    company_id: 업체ID (예: C001). top_k: 돌려줄 공고 수 (1~10).
    """
    company = _company(company_id)
    out = run_pipeline(list(tenders_by_no().values()), company, _region_limits())
    top = [{"rank": i, "bidNtceNo": r["bidNtceNo"], "bidNtceNm": r["bidNtceNm"], "budget": r.get("budget"),
            "soft_score": r["soft_score"], "info": r.get("info"), "reasons": [x.get("text") for x in r.get("reasons", [])]}
           for i, r in enumerate(out["top"][:max(1, min(top_k, 10))], 1)]
    funnel = out["filter_log"]["funnel"]
    return {"company": company.get("회사명"), "search_query": out["search"].get("query"),
            "funnel": {"전체": funnel["total"], "자격 필터 통과": funnel["hard_pass"]}, "recommendations": top}


@mcp.tool()
def check_qualification(company_id: str, tender_no: str) -> dict:
    """업체가 공고의 참가 자격 규칙(PDF 보유·업종·참가 제한·지역 제한·예산)을 통과하는지. 걸린 규칙과 사유를 돌려준다."""
    company, tender = _company(company_id), _tender(tender_no)
    failed = hard_filter(tender, company, _region_limits())
    return {"bidNtceNo": tender["bidNtceNo"], "bidNtceNm": tender.get("bidNtceNm"), "passed": not failed,
            "failed_rules": [{"rule": r, "reason": RULES[r]} for r in failed],
            "note": "규칙으로 확인한 결과입니다. 공고서 본문의 면허·인증·실적 조건은 get_checklist로 확인하세요."}


@mcp.tool()
def get_tender(tender_no: str) -> dict:
    """공고 상세 (공고명·예산·업종·계약방식·참가 제한·지역 제한·마감·수요기관)."""
    t = _tender(tender_no)
    return {**{f: t.get(f) for f in TENDER_FIELDS}, "summary": tender_info(t), "pdfs": [p.get("name") for p in t["pdfs"]]}


@mcp.tool()
def search_evidence(tender_no: str, query: str, k: int = 3) -> list[dict]:
    """공고서 PDF에서 질의와 관련된 조항 청크 k개 (파일·페이지·섹션·원문). 근거 인용에 쓴다."""
    chunks = tender_chunks(_tender(tender_no))
    return [{"file": c["file"], "page": c["page"], "section": c.get("section"), "text": c["text"]}
            for c in retrieve(chunks, query, k=max(1, min(k, 8)), tender_no=tender_no)]


@mcp.tool()
def get_checklist(tender_no: str) -> list[dict]:
    """입찰 참가 전 확인 체크리스트 (공고 조건 + 공고서에서 뽑은 자격·서류·일정). LLM 키가 없으면 공고 필드 항목만."""
    return [{k: i.get(k) for k in ("category", "text", "origin")} for i in checklist.build(_tender(tender_no))["items"]]


if __name__ == "__main__":
    mcp.run()
