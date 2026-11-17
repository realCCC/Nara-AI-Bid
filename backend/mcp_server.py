"""MCP 서버: 누구나 자기 업체 정보·공고서로 쓸 수 있는 입찰 도구 3개.

- check_eligibility: 업체 프로필(JSON)을 직접 넘겨 나라장터 공고의 참가 자격 규칙(H1~H6)을 판정
- extract_requirements: 공고서 PDF(경로·URL)에서 참가자격·실적·제출서류·일정을 원문 인용과 함께 추출
- verify_quote: 인용문이 공고서 PDF 원문에 실제로 있는지 대조 (다른 AI가 쓴 제안서·요약의 환각 검사)

웹 화면(FastAPI + LangGraph)과 같은 규칙·청킹·원문 대조 함수를 쓴다. 판단(대화·요약)은 클라이언트 LLM이 한다.

실행: uv sync --extra mcp  →  uv run --extra mcp python -m backend.mcp_server   (stdio)
"""
import hashlib
import logging
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from backend import config
from backend.tools.checklist import QUERY as REQUIREMENT_QUERY
from backend.tools.evidence import check_reason
from backend.tools.loaders import load_region_limits, tenders_by_no
from backend.tools.pdf import chunk_pages, parse_pdf
from backend.tools.retrieval import retrieve
from backend.tools.rules import RULES, hard_filter, tender_info

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)  # stdio 전송: stdout은 MCP 메시지 전용

mcp = MCPServer("nara-bid-agent", instructions=(
    "입찰 준비 도구. check_eligibility는 업체 프로필을 넘겨 나라장터 공고(2026-05 스냅샷)의 참가 자격 규칙을 판정하고, "
    "extract_requirements는 아무 공고서 PDF에서 참가 요건을 원문 인용과 함께 뽑으며, verify_quote는 인용문이 원문에 "
    "있는지 대조한다. 결과의 근거(규칙 사유·원문 인용·쪽수)를 그대로 보여 주고, 최종 입찰 판단은 사용자에게 맡긴다."))

PDF_DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "nara-bid-mcp"
MAX_PDF_BYTES = 30 * 1024 * 1024


@lru_cache
def _region_limits() -> dict:
    return load_region_limits()


def _tender(tender_no: str) -> dict:
    t = tenders_by_no().get(tender_no.strip())
    if t is None:
        raise ToolError(f"없는 공고번호: {tender_no} (2026-05 나라장터 용역 공고 스냅샷 기준)")
    return t


def _pdf_path(pdf: str) -> Path:
    """로컬 경로는 그대로, http(s) URL은 임시 폴더에 받아 둔다 (같은 URL은 다시 받지 않음)."""
    pdf = pdf.strip()
    if re.match(r"https?://", pdf, re.IGNORECASE):
        path = PDF_DOWNLOAD_DIR / f"{hashlib.sha1(pdf.encode()).hexdigest()[:16]}.pdf"
        if not path.exists():
            try:
                r = httpx.get(pdf, timeout=30, follow_redirects=True)
                r.raise_for_status()
            except httpx.HTTPError as e:
                raise ToolError(f"PDF 다운로드 실패: {e}") from e
            if not r.content.startswith(b"%PDF"):
                raise ToolError("URL 응답이 PDF가 아닙니다")
            if len(r.content) > MAX_PDF_BYTES:
                raise ToolError("PDF가 30MB를 넘습니다")
            PDF_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
        return path
    path = Path(pdf).expanduser()
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise ToolError(f"PDF 파일을 찾을 수 없습니다: {pdf}")
    return path


@lru_cache(maxsize=16)
def _chunks(pdf: str) -> tuple[str, tuple[dict, ...]]:
    """(파일명, 조항 단위 청크). 같은 PDF를 여러 도구가 연달아 불러도 한 번만 추출한다."""
    path = _pdf_path(pdf)
    try:
        chunks = chunk_pages(path.name, parse_pdf(path))
    except Exception as e:
        raise ToolError(f"PDF 텍스트 추출 실패: {e}") from e
    if not chunks:
        raise ToolError("PDF에서 텍스트를 찾지 못했습니다 (스캔 이미지 PDF일 수 있음)")
    return path.name, tuple(chunks)


@mcp.tool()
def check_eligibility(company: dict, tender_no: str) -> dict:
    """업체 프로필로 나라장터 공고의 참가 자격 규칙(PDF 보유·업종·중소기업 제한·지역 제한·예산)을 판정한다.

    company 예: {"소재지": "서울", "중소기업여부": true, "업종": {"대분류": ["행사"], "세부분류번호": ["8011159901"]},
                 "참여가능금액": {"최소": 0, "최대": 300000000}}  — 없는 항목은 해당 규칙에서 불리하게 판정될 수 있다.
    tender_no: 나라장터 공고번호 (2026-05 용역 공고 스냅샷).
    """
    tender = _tender(tender_no)
    failed = hard_filter(tender, company, _region_limits())
    return {"bidNtceNo": tender["bidNtceNo"], "bidNtceNm": tender.get("bidNtceNm"), "tender": tender_info(tender),
            "passed": not failed, "failed_rules": [{"rule": r, "reason": RULES[r]} for r in failed],
            "note": "공고 필드로 확인한 규칙 결과입니다. 공고서 본문의 면허·인증·실적 조건은 extract_requirements로 확인하세요."}


@mcp.tool()
def extract_requirements(pdf: str) -> dict:
    """공고서 PDF에서 입찰 참가 전 확인할 요건(참가자격·실적요건·제출서류·일정·기타)을 뽑는다.

    pdf: 로컬 파일 경로 또는 http(s) URL. 항목마다 원문 인용·쪽수를 붙이고, 인용이 원문에 없는 항목은 버린다.
    LLM을 쓸 수 없으면 요건 관련 조항 원문만 돌려준다.
    """
    name, chunks = _chunks(pdf)
    chunks = list(chunks)
    hits = retrieve(chunks, REQUIREMENT_QUERY, k=8)
    clauses = [{"page": c["page"], "section": c.get("section"), "text": c["text"]} for c in hits]
    if not config.llm_enabled():
        return {"file": name, "source": "clauses", "items": [], "clauses": clauses}
    from backend.tools import llm

    raw = llm.extract_checklist({"bidNtceNm": name}, hits) or {}
    items, dropped = [], 0
    for it in raw.get("items", []):
        ev = {"source_type": "pdf", "file": name, "page": it.get("page"), "quote": it.get("quote")}
        if (it.get("text") or "").strip() and check_reason(ev, {}, {}, chunks):
            items.append({"category": it.get("category") or "기타", "text": it["text"].strip(),
                          "page": it.get("page"), "quote": it.get("quote")})
        else:
            dropped += 1
    return {"file": name, "source": "llm" if items else "clauses", "items": items, "dropped_unverified": dropped,
            "clauses": [] if items else clauses}


@mcp.tool()
def verify_quote(quote: str, pdf: str, page: int | None = None) -> dict:
    """인용문이 공고서 PDF 원문에 실제로 있는지 대조한다. 공백 차이는 무시하고, 줄임표(... …)로 줄인 인용은
    조각마다 같은 조항에 있어야 통과한다.

    quote: 확인할 인용문. pdf: 로컬 파일 경로 또는 http(s) URL. page: 적힌 쪽수 (주면 그 쪽만 확인).
    """
    parts = _quote_parts(quote)
    if sum(map(len, parts)) < 4:
        raise ToolError("인용문이 너무 짧습니다 (공백 제외 4자 이상)")
    name, chunks = _chunks(pdf)
    hits = [c for c in chunks if (page is None or page in c.get("pages", (c["page"],)))
            and all(p in _norm(c["text"]) for p in parts)]
    return {"file": name, "found": bool(hits), "pages": sorted({c["page"] for c in hits}),
            "matches": [{"page": c["page"], "pages": c.get("pages"), "section": c.get("section"), "text": c["text"]}
                        for c in hits[:3]]}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _quote_parts(quote: str) -> list[str]:
    return [p for p in (_norm(s) for s in re.split(r"\.{2,}|…", quote or "")) if p]


if __name__ == "__main__":
    mcp.run()
