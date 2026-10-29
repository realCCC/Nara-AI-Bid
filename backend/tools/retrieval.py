"""청크 검색. retrieve: 키워드 겹침(기본) 또는 BGE-M3 유사도. retrieve_for_company(RAG 근거): BM25 + BGE-M3 → RRF.

RAG 청크 벡터는 상위 공고의 청크만 계산해 pgvector(backend/db/vector_store.py)에 저장한다 (전체 PDF를 미리 받지 않음).
같은 공고가 다시 추천되면 읽기만 한다. 데모 전에는 scripts/warm_chunk_vectors.py로 미리 채운다.
임베딩·DB를 못 쓰면(EMBEDDING_PROVIDER=keyword·CHUNK_EMBEDDING=0·모델 실패·DB 미기동) BM25만."""
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


@lru_cache
def _azure():
    from openai import AzureOpenAI

    return AzureOpenAI(max_retries=5)  # AZURE_OPENAI_ENDPOINT·AZURE_OPENAI_API_KEY·OPENAI_API_VERSION 환경변수


AZURE_BATCH, AZURE_WORKERS = 64, 4


def _azure_encode(texts: list[str]):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    def call(batch):
        r = _azure().embeddings.create(model=config.EMBEDDING_MODEL, input=[t or " " for t in batch],
                                       dimensions=config.EMBED_DIM)
        return [d.embedding for d in r.data]

    batches = [texts[i:i + AZURE_BATCH] for i in range(0, len(texts), AZURE_BATCH)]
    with ThreadPoolExecutor(AZURE_WORKERS) as ex:
        vecs = np.array([v for out in ex.map(call, batches) for v in out], dtype="float32")
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def encode(texts: list[str]):
    """정규화 벡터 (행렬). bge: 로컬 BGE-M3, azure: Azure OpenAI 임베딩 API (EMBED_DIM 차원)."""
    if not texts:
        raise ValueError("빈 입력")
    if config.EMBEDDING_PROVIDER == "azure":
        return _azure_encode(texts)
    return _bge().encode(texts, normalize_embeddings=True, batch_size=32)


def search_text(chunk: dict) -> str:
    """검색·임베딩용 텍스트: 섹션 제목이 붙은 search_text (없으면 원문)."""
    return chunk.get("search_text") or chunk["text"]


@traced("retrieve", "retriever", big=("chunks",), outputs=chunk_view)
def retrieve(chunks: list[dict], query: str, k: int = 3, tender_no: str | None = None) -> list[dict]:
    """질의 하나로 상위 k 청크 (검증 자격 청크·체크리스트·평가항목).
    tender_no를 주면 RAG 단계가 pgvector에 저장한 청크 벡터를 다시 쓴다 — 같은 청크를 또 임베딩하지 않음."""
    if not chunks:
        return []
    scores = None
    if config.embedding_enabled() and config.CHUNK_EMBEDDING:
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
    """공고 하나의 청크 벡터 (행렬). pgvector에 없는 청크만 계산해 메타데이터와 함께 넣는다.
    임베딩·DB를 못 쓰면 예외 → 호출부가 BM25(키워드)만."""
    import numpy as np

    from backend.db import vector_store

    if not config.embedding_enabled() or not config.CHUNK_EMBEDDING:
        raise RuntimeError(f"EMBEDDING_PROVIDER={config.EMBEDDING_PROVIDER}, CHUNK_EMBEDDING={config.CHUNK_EMBEDDING}")
    keys = [_chunk_key(search_text(c)) for c in chunks]
    cache = vector_store.load(tender_no, keys)
    todo = [(k, c) for k, c in dict(zip(keys, chunks)).items() if k not in cache]
    if todo:
        vecs = encode([search_text(c) for _, c in todo])
        cache.update({k: v for (k, _), v in zip(todo, vecs)})
        vector_store.save(tender_no, [(k, c, cache[k]) for k, c in todo], keep=keys)
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
