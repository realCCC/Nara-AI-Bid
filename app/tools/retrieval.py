"""청크 검색. 기본은 키워드 겹침(설치 불필요), EMBEDDING_PROVIDER=bge면 BGE-M3 유사도."""
from functools import lru_cache

from app import config
from app.tools.rules import tokenize

# 고정 질의 (3번 문서 SC-001 RAG 추천)
FIXED_QUERIES = ("입찰 참가자격 제한", "과업 내용 범위 수행", "평가 방법 배점 기술 가격")


@lru_cache
def _bge():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def retrieve(chunks: list[dict], query: str, k: int = 3) -> list[dict]:
    if not chunks:
        return []
    if config.EMBEDDING_PROVIDER == "bge":
        model = _bge()
        vecs = model.encode([query] + [c["text"] for c in chunks], normalize_embeddings=True)
        scores = (vecs[1:] @ vecs[0]).tolist()
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
