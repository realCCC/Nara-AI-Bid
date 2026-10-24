"""청크 검색. 기본은 키워드 겹침(설치 불필요), EMBEDDING_PROVIDER=bge + CHUNK_EMBEDDING=1이면 BGE-M3 유사도.

CPU에서는 공고서 청크(10건 × 약 15청크) 임베딩이 수 분 걸려 기본은 키워드 겹침 (인터뷰 확장어는 반영됨)."""
from functools import lru_cache

from backend import config
from backend.tools.rules import tokenize

# 고정 질의 (3번 문서 SC-001 RAG 추천)
FIXED_QUERIES = ("입찰 참가자격 제한", "과업 내용 범위 수행", "평가 방법 배점 기술 가격")


@lru_cache
def _bge():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def retrieve(chunks: list[dict], query: str, k: int = 3) -> list[dict]:
    if not chunks:
        return []
    if config.EMBEDDING_PROVIDER == "bge" and config.CHUNK_EMBEDDING:
        from backend.tools.search import embed  # 벡터 캐시: 같은 청크를 질의마다 다시 계산하지 않음

        scores = (embed([c["text"] for c in chunks]) @ embed([query])[0]).tolist()
    else:
        q = tokenize(query)
        scores = [len(q & tokenize(c["text"])) for c in chunks]
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [chunks[i] for i in order[:k] if scores[i] > 0]


def retrieve_for_company(chunks: list[dict], company: dict, k: int = 3) -> list[dict]:
    """업체 역량 + 고정 질의로 검색, 중복 제거."""
    kw = " ".join(company.get("역량키워드", []))
    seen, out = set(), []
    for q in FIXED_QUERIES:
        for c in retrieve(chunks, f"{kw} {q}", k):
            key = (c["file"], c["page"], c["text"][:40])
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out
