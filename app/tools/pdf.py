"""공고서 PDF 다운로드(캐시)·텍스트 추출·청킹."""
import logging
import re
from pathlib import Path

import httpx

from app import config

log = logging.getLogger(__name__)
CHUNK_SIZE = 500


def fetch_tender_pdfs(tender: dict, pdf_dir: Path | None = None) -> list[dict]:
    """첨부 PDF를 data/pdfs/<공고번호>/ 에 저장. 이미 있으면 다시 받지 않음. 실패한 파일은 건너뜀."""
    out = Path(pdf_dir or config.PDF_DIR) / tender["bidNtceNo"]
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, f in enumerate(tender["pdfs"], 1):
        path = out / f"{i}.pdf"
        if not path.exists() and config.FETCH_PDFS:
            try:
                r = httpx.get(f["url"], timeout=30, follow_redirects=True)
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    raise ValueError("PDF 형식 아님")
                path.write_bytes(r.content)
            except Exception as e:  # 다운로드 실패 → 해당 파일은 룰 근거만
                log.warning("PDF 다운로드 실패 %s %s: %s", tender["bidNtceNo"], f["name"], e)
                continue
        if path.exists():
            saved.append({"file": f["name"], "path": path})
    return saved


def parse_pdf(path: Path) -> list[tuple[int, str]]:
    import pymupdf

    with pymupdf.open(path) as doc:
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


def chunk_pages(file: str, pages: list[tuple[int, str]], size: int = CHUNK_SIZE) -> list[dict]:
    """페이지 단위로 자른 뒤 size자씩. 출처(file·page)를 청크마다 유지."""
    chunks = []
    for page, text in pages:
        text = re.sub(r"\s+", " ", text).strip()
        for start in range(0, len(text), size):
            piece = text[start:start + size]
            if len(piece) > 30:
                chunks.append({"file": file, "page": page, "text": piece})
    return chunks


def tender_chunks(tender: dict) -> list[dict]:
    chunks = []
    for f in fetch_tender_pdfs(tender):
        try:
            chunks += chunk_pages(f["file"], parse_pdf(f["path"]))
        except Exception as e:
            log.warning("PDF 추출 실패 %s: %s", f["path"], e)
    return chunks
