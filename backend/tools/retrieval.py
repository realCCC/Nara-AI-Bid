"""청크 검색. retrieve: 키워드 겹침(기본) 또는 BGE-M3 유사도. retrieve_for_company(RAG 근거): BM25 + BGE-M3 → RRF.

RAG 청크 벡터는 상위 공고의 청크만 계산해 CHUNK_VEC_DIR/<공고번호>.npz로 저장한다 (전체 PDF를 미리 받지 않음).
같은 공고가 다시 추천되면 읽기만 한다. 데모 전에는 scripts/warm_chunk_vectors.py로 미리 채운다.
임베딩을 못 쓰면(EMBEDDING_PROVIDER=keyword·CHUNK_EMBEDDING=0·모델 실패) BM25만."""
import hashlib
import logging
from functools import lru_cache

from backend import config
from backend.tracing import chunk_view, traced
from backend.tools.rules import tokenize

log = logging.getLogger(__name__)

# 고정 질의 (3번 문서 SC-001 RAG 추천): 자격·배점 같은 정형 용어는 BM25가, 과업 내용은 검색 문장(dense)이 맡는다
FIXED_QUERIES = ("입찰 참가자격 제한", "과업 내용 범위 수행", "평가 방법 배점 기술 가격")


@lru_cache
def _bge():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


@traced("retrieve", "retriever", big=("chunks",), outputs=chunk_view)
def search_text(chunk: dict) -> str:
    """검색·임베딩용 텍스트: 섹션 제목이 붙은 search_text (없으면 원문)."""
    return chunk.get("search_text") or chunk["text"]


def retrieve(chunks: list[dict], query: str, k: int = 3, tender_no: str | None = None) -> list[dict]:
    """질의 하나로 상위 k 청크 (검증 자격 청크·체크리스트·평가항목).
    tender_no를 주면 RAG 단계가 저장한 공고별 청크 벡터(CHUNK_VEC_DIR)를 다시 쓴다 — 같은 청크를 또 임베딩하지 않음."""
    if not chunks:
        return []
    scores = None
    if config.EMBEDDING_PROVIDER == "bge" and config.CHUNK_EMBEDDING:
        from backend.tools.search import embed

        try:
            vecs = chunk_vectors(tender_no, chunks) if tender_no else embed([search_text(c) for c in chunks])
            scores = (vecs @ embed([query])[0]).tolist()
        except Exception as e:  # 미설치·모델 로드 실패 → 키워드 겹침
            log.info("청크 임베딩 미사용: %s", e)
    if scores is None:
        q = tokenize(query)
        scores = [len(q & tokenize(search_text(c))) for c in chunks]
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [chunks[i] for i in order[:k] if scores[i] > 0]


def _chunk_key(text: str) -> str:
    return hashlib.sha1(f"{config.EMBEDDING_MODEL}\0{text}".encode("utf-8")).hexdigest()


def chunk_vectors(tender_no: str, chunks: list[dict]):
    """공고 하나의 청크 벡터 (행렬). 공고별 파일에 없는 청크만 계산해 그 파일만 다시 쓴다.
    임베딩을 못 쓰면 예외 → 호출부가 BM25만."""
    import numpy as np

    if config.EMBEDDING_PROVIDER != "bge" or not config.CHUNK_EMBEDDING:
        raise RuntimeError(f"EMBEDDING_PROVIDER={config.EMBEDDING_PROVIDER}, CHUNK_EMBEDDING={config.CHUNK_EMBEDDING}")
    path = config.CHUNK_VEC_DIR / f"{tender_no}.npz"
    cache = {}
    if path.exists():
        try:
            with np.load(path) as z:
                cache = {k: z[k] for k in z.files}
        except Exception as e:  # 깨진 파일은 새로 계산
            log.warning("청크 벡터 로드 실패 %s: %s", path, e)
    keys = [_chunk_key(search_text(c)) for c in chunks]
    todo = list(dict.fromkeys((k, search_text(c)) for k, c in zip(keys, chunks) if k not in cache))
    if todo:
        vecs = _bge().encode([t for _, t in todo], normalize_embeddings=True, batch_size=32)
        cache.update({k: v for (k, _), v in zip(todo, vecs)})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **{k: cache[k] for k in dict.fromkeys(keys)})  # 지금 청크만 저장 (청킹이 바뀌면 옛 벡터는 사라짐)
        except Exception as e:
            log.warning("청크 벡터 저장 실패 %s: %s", path, e)
    return np.array([cache[k] for k in keys])


def hybrid_retrieve(chunks: list[dict], query: str, k: int = 3, vecs=None) -> list[dict]:
    """BM25 + dense(vecs가 있으면) 순위를 RRF로 합쳐 상위 k. 청크마다 단계별 순위를 'match'로 남긴다."""
    from backend.tools.search import BM25Retriever, embed, rrf

    if not chunks:
        return []
    docs = {str(i): search_text(c) for i, c in enumerate(chunks)}
    rankings = {"bm25": [i for i, _ in BM25Retriever(docs).search(query)]}
    if vecs is not None:
        scores = (vecs @ embed([query])[0]).tolist()
        rankings["dense"] = sorted(docs, key=lambda i: -scores[int(i)])
    fused = rrf(rankings)
    rank = {name: {i: r for r, i in enumerate(order, 1)} for name, order in rankings.items()}
    top = sorted(fused, key=lambda i: (-fused[i], int(i)))[:k]
    return [{**chunks[int(i)], "match": {"bm25": rank["bm25"].get(i), "dense": rank.get("dense", {}).get(i),
                                         "rrf": round(fused[i], 5)}} for i in top]


@traced("retrieve_for_company", "retriever", big=("chunks",), outputs=lambda r: {"path": r[1], "chunks": chunk_view(r[0])})
def retrieve_for_company(chunks: list[dict], company: dict, k: int = 3, query: str | None = None,
                         tender_no: str | None = None) -> tuple[list[dict], str]:
    """업체 역량(+키워드 의미 프로필 유사어) + 고정 질의, 그리고 공고 검색에 쓴 검색 문장(query)으로 하이브리드 검색, 중복 제거.
    반환: (청크, 경로) — 경로는 hybrid(BM25+dense) / bm25(임베딩 미사용)."""
    if not chunks:
        return [], "none"
    vecs, path = None, "bm25"
    if tender_no:
        try:
            vecs, path = chunk_vectors(tender_no, chunks), "hybrid"
        except Exception as e:  # 미설치·모델 로드 실패·설정 off → BM25만
            log.info("청크 임베딩 미사용: %s", e)
    kw = " ".join([*company.get("역량키워드", []), *(s for v in company.get("유사어", {}).values() for s in v)])
    queries = [f"{kw} {q}" for q in FIXED_QUERIES] + ([query] if query else [])
    seen, out = set(), []
    for q in queries:
        for c in hybrid_retrieve(chunks, q, k, vecs):
            key = (c["file"], c["page"], c["text"][:40])
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out, path
