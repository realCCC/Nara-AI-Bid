"""업체 문서(회사소개서·실적 설명 등) 청킹·벡터 검색 — 제안서 초안 LLM 작성의 업체 근거.

<COMPANY_DOCS_DIR>/<업체ID>/*.txt|*.md|*.pdf 를 공고서와 같은 조항 청킹으로 나눈다.
벡터는 공고 청크와 같은 pgvector 테이블에 'company:<업체ID>' 키로 저장한다 (필요할 때 없는 청크만 계산, 문서가 바뀌면 갱신).
임베딩·DB를 못 쓰면 BM25만. 문서가 없으면 빈 목록 → 업체 JSON만 근거로 쓴다.
"""
import logging
import re
from pathlib import Path

from backend import config
from backend.tools import activity
from backend.tools.pdf import chunk_pages, parse_pdf

log = logging.getLogger(__name__)

SUFFIXES = (".txt", ".md", ".pdf")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")


def _text_pages(path: Path) -> list[tuple[int, str]]:
    """txt/md → 1쪽짜리 페이지. 마크다운 제목(# …)은 섹션 기호(□)로 바꿔 조항 청킹이 섹션으로 인식하게 한다."""
    lines = [_MD_HEADING.sub("□ ", line) if _MD_HEADING.match(line) else line
             for line in path.read_text(encoding="utf-8").splitlines()]
    return [(1, "\n".join(lines))]


def doc_dir(company: dict) -> Path:
    return config.COMPANY_DOCS_DIR / str(company.get("업체ID", ""))


def load_chunks(company: dict) -> list[dict]:
    """업체 문서 청크. 청크마다 file(파일명)·page·section·text."""
    d = doc_dir(company)
    if not company.get("업체ID") or not d.is_dir():
        return []
    chunks = []
    for f in sorted(p for p in d.iterdir() if p.suffix.lower() in SUFFIXES):
        try:
            pages = parse_pdf(f) if f.suffix.lower() == ".pdf" else _text_pages(f)
            chunks += chunk_pages(f.name, pages)
        except Exception as e:
            log.warning("업체 문서 읽기 실패 %s: %s", f, e)
    return chunks


def vectors(company: dict, chunks: list[dict]):
    """청크 벡터 (pgvector 캐시). 임베딩·DB를 못 쓰면 None → BM25만."""
    if not chunks:
        return None
    from backend.tools.retrieval import chunk_vectors

    try:
        return chunk_vectors(f"company:{company['업체ID']}", chunks)
    except Exception as e:
        log.info("업체 문서 임베딩 미사용: %s", e)
        return None


def search(chunks: list[dict], query: str, k: int = 4, vecs=None) -> list[dict]:
    """BM25 + dense(vecs가 있으면) RRF 상위 k."""
    from backend.tools.retrieval import hybrid_retrieve

    hits = hybrid_retrieve(chunks, query, k, vecs) if chunks else []
    if chunks:
        activity.note("RAG", f"업체 문서 청크 {len(hits)}개 검색 ({'벡터+BM25' if vecs is not None else 'BM25'})")
    return hits
