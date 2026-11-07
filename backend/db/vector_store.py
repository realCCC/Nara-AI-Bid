"""pgvector 청크 벡터 저장소: 공고서 PDF 청크 벡터 + 메타데이터(공고번호·파일·페이지·섹션·원문).

DB는 docker compose up -d 로 띄운다 (uv sync --extra rag 필요).
연결·설치 실패는 예외 → 호출부(retrieval)가 BM25(키워드)만으로 검색한다.
"""
import logging
import time
from pathlib import Path

from backend import config
from backend.tools import activity

log = logging.getLogger(__name__)
SCHEMA = Path(__file__).with_name("vector_schema.sql")
RETRY_SEC = 30  # 연결 실패 후 이 시간 동안은 바로 실패 (공고마다 연결 대기 방지), 지나면 다시 시도
_CONN = None
_ERROR: tuple[float, str] | None = None  # (실패 시각, 사유)


def _conn():
    global _CONN, _ERROR
    if _CONN is not None and not _CONN.closed:
        return _CONN
    if _ERROR and time.monotonic() - _ERROR[0] < RETRY_SEC:
        raise RuntimeError(_ERROR[1])
    try:
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(config.PGVECTOR_DSN, autocommit=True, connect_timeout=3)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)  # vector 타입 ↔ numpy 배열
        conn.execute(SCHEMA.read_text(encoding="utf-8"))
    except Exception as e:  # 미설치·DB 미기동
        _ERROR = (time.monotonic(), f"pgvector 사용 불가: {e}")
        log.warning("%s (%d초 뒤 다시 시도)", _ERROR[1], RETRY_SEC)
        reason = (str(e).splitlines() or [type(e).__name__])[0][:100]
        activity.note("벡터DB", f"연결 실패 → BM25로 대체, {RETRY_SEC}초 뒤 재시도 ({reason})", "warn")
        raise RuntimeError(_ERROR[1]) from e
    _CONN, _ERROR = conn, None
    activity.note("벡터DB", "pgvector 연결 성공")
    return conn


def load(tender_no: str, keys: list[str]) -> dict:
    """공고 하나의 저장된 청크 벡터 {text_hash: 벡터}."""
    rows = _conn().execute(
        "SELECT text_hash, embedding FROM tender_chunks WHERE bid_ntce_no = %s AND model = %s AND text_hash = ANY(%s)",
        (tender_no, config.EMBEDDING_MODEL, list(keys))).fetchall()
    activity.note("벡터DB", f"조회 · {tender_no} 저장된 벡터 {len(rows)}/{len(keys)}개")
    return {k: v.to_numpy() for k, v in rows}  # pgvector.Vector → numpy


def save(tender_no: str, rows: list[tuple[str, dict, object]], keep: list[str]):
    """rows(text_hash, 청크, 벡터)를 넣고, keep(지금 청크)에 없는 이 공고의 옛 청크는 지운다 (청킹이 바뀐 경우)."""
    conn = _conn()
    with conn.transaction():
        conn.execute("DELETE FROM tender_chunks WHERE bid_ntce_no = %s AND model = %s AND NOT (text_hash = ANY(%s))",
                     (tender_no, config.EMBEDDING_MODEL, list(keep)))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO tender_chunks (bid_ntce_no, model, text_hash, file, page, pages, section, text, search_text,"
                " embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (bid_ntce_no, model, text_hash) DO NOTHING",
                [(tender_no, config.EMBEDDING_MODEL, k, c.get("file"), c.get("page"), c.get("pages"), c.get("section"),
                  c["text"], c.get("search_text") or c["text"], v) for k, c, v in rows])
    activity.note("벡터DB", f"저장 · {tender_no} 벡터 {len(rows)}개")


def tender_scores(query_vec, tender_nos: list[str], top: int = 3) -> dict[str, float]:
    """공고별 본문 점수 {공고번호: 질의↔청크 코사인 상위 top개 평균}. 청크가 없는 공고는 빠진다.
    벡터를 가져오지 않고 DB 안에서 계산한다 (후보 수백 건 × 청크 수십 개)."""
    rows = _conn().execute(
        "SELECT bid_ntce_no, avg(score) FROM ("
        " SELECT bid_ntce_no, 1 - (embedding <=> %s) AS score,"
        "  row_number() OVER (PARTITION BY bid_ntce_no ORDER BY embedding <=> %s) AS rn"
        " FROM tender_chunks WHERE model = %s AND bid_ntce_no = ANY(%s)) x"
        " WHERE rn <= %s GROUP BY bid_ntce_no",
        (query_vec, query_vec, config.EMBEDDING_MODEL, list(tender_nos), top)).fetchall()
    activity.note("벡터DB", f"본문 점수 조회 · 공고 {len(tender_nos)}건 중 {len(rows)}건")
    return {no: float(s) for no, s in rows}


def search(query_vec, k: int = 5, tender_no: str | None = None, section: str | None = None) -> list[dict]:
    """DB 안에서 코사인 유사도 상위 k 청크 (HNSW). 공고·섹션으로 좁힐 수 있다 — 여러 공고를 가로지르는 검색용."""
    where, params = ["model = %s"], [config.EMBEDDING_MODEL]
    if tender_no:
        where.append("bid_ntce_no = %s")
        params.append(tender_no)
    if section:
        where.append("section = %s")
        params.append(section)
    rows = _conn().execute(
        "SELECT bid_ntce_no, file, page, pages, section, text, search_text, 1 - (embedding <=> %s) AS score"
        f" FROM tender_chunks WHERE {' AND '.join(where)} ORDER BY embedding <=> %s LIMIT %s",
        (query_vec, *params, query_vec, k)).fetchall()
    activity.note("벡터DB", f"유사 청크 조회 · {len(rows)}개")
    cols = ("bidNtceNo", "file", "page", "pages", "section", "text", "search_text", "score")
    return [dict(zip(cols, r)) | {"score": round(float(r[-1]), 4)} for r in rows]
