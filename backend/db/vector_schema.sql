-- 공고서 PDF 청크 벡터 (VECTOR_STORE=pgvector). 연결 시 backend/db/vector_store.py가 실행한다.
CREATE TABLE IF NOT EXISTS tender_chunks (
    id          BIGSERIAL PRIMARY KEY,
    bid_ntce_no TEXT NOT NULL,         -- 공고번호
    model       TEXT NOT NULL,         -- 임베딩 모델 (모델이 바뀌면 이전 벡터와 섞이지 않게)
    text_hash   TEXT NOT NULL,         -- sha1(모델 + search_text), npz 캐시 키와 같음
    file        TEXT,
    page        INTEGER,
    pages       INTEGER[],
    section     TEXT,                  -- 조항 제목 (입찰참가자격, 평가 방법 ...)
    text        TEXT NOT NULL,         -- 원문 (출처 대조용)
    search_text TEXT NOT NULL,         -- 임베딩한 문장 ([섹션] 원문)
    embedding   vector(1024) NOT NULL, -- BGE-M3 정규화 벡터
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (bid_ntce_no, model, text_hash)
);

-- 코사인 거리(<=>) 근사 최근접 검색
CREATE INDEX IF NOT EXISTS tender_chunks_embedding_idx ON tender_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS tender_chunks_section_idx ON tender_chunks (section);
