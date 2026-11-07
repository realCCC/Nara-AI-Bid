"""추천 검색 (인메모리, ES 미사용): 업체 → 검색 문장 → BGE-M3(dense) + BM25(kiwipiepy) → RRF(k=60) → 상위 15건 리랭크.

실행경로: semantic(dense+BM25) / keyword_fallback(BM25만 — 임베딩 미설치·실패·EMBEDDING_PROVIDER=keyword).
리랭커(bge-reranker-v2-m3)는 semantic 경로에서만 시도하고, 실패하면 생략한다.
BODY_SEARCH=1이면 semantic 경로에서 PDF 본문 순위(pgvector 청크)도 RRF에 넣는다.
"""
import hashlib
import logging
import math
from collections import Counter
from functools import lru_cache
from typing import Protocol

from backend import config
from backend.tools import activity
from backend.tracing import traced
from backend.tools.rules import tokenize

log = logging.getLogger(__name__)
RRF_K = 60
RERANK_TOP = 15
_NOUN_TAGS = ("NN", "SL", "SH", "SN", "XR")
MASTER_GENERIC = {"사업", "용역", "서비스", "운영", "지원", "관리", "기획", "대행", "지역", "행사"}  # 공고마다 흔해 사업을 가르지 못하는 말
DOC_FIELDS = ("bidNtceNm", "purchsObjPrdctList", "pubPrcrmntLrgClsfcNm", "pubPrcrmntMidClsfcNm", "pubPrcrmntClsfcNm",
              "dminsttNm")


def tender_doc(tender: dict) -> str:
    return " ".join(str(tender.get(k) or "") for k in DOC_FIELDS)


def master_match(key: str, doc: str) -> bool:
    """마스터키가 공고 문서에 있는가: 붙여 쓴 문자열 일치, 또는 일반어를 뺀 키의 명사가 모두 있음
    ('AI 교육' ↔ 'AI 활용 교육', '지역 축제 운영' ↔ '한마음축제 기획')."""
    if key.replace(" ", "").lower() in doc.replace(" ", "").lower():
        return True
    tokens = set(ko_tokens(key)) - MASTER_GENERIC
    return bool(tokens) and tokens <= set(ko_tokens(doc))


def master_first(result: dict, docs: dict[str, str], key: str) -> dict:
    """마스터키 일치 공고를 앞으로 (제외는 없다). 일치끼리·나머지끼리는 기존 순서를 지켜 역량·의미 순위를 유지한다."""
    hits = [i for i in result["order"] if master_match(key, docs[i])]
    rest = [i for i in result["order"] if i not in set(hits)]
    return {**result, "order": [*hits, *rest], "master": {"key": key, "hits": len(hits)}}


def template_query(company: dict) -> str:
    biz = ", ".join(company.get("업종", {}).get("대분류", []))
    kws = ", ".join(company.get("역량키워드", []))
    past = ", ".join(h["프로젝트명"] for h in company.get("과거실적", [])[:3])
    return f"{biz} 분야에서 {kws} 역량을 가진 업체가 수행할 용역 공고. 과거 수행 사업: {past}"


# ---------- Retriever 인터페이스 ----------
class Retriever(Protocol):
    def search(self, query: str) -> list[tuple[str, float]]:
        """(문서ID, 점수) 내림차순."""


@lru_cache
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def ko_tokens(text: str) -> list[str]:
    """kiwipiepy 명사·외국어·숫자. 미설치·실패 시 공백/기호 분리로 폴백."""
    try:
        return [t.form for t in _kiwi().tokenize(text) if t.tag.startswith(_NOUN_TAGS)]
    except Exception as e:  # ImportError 포함
        log.warning("kiwipiepy 사용 불가, 단순 분리로 대체: %s", e)
        return list(tokenize(text))


class BM25Retriever:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {i: Counter(ko_tokens(text)) for i, text in docs.items()}
        self.len = {i: sum(c.values()) for i, c in self.tf.items()}
        self.avg = (sum(self.len.values()) / len(self.len)) if self.len else 0
        df = Counter(w for c in self.tf.values() for w in c)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str) -> list[tuple[str, float]]:
        q = set(ko_tokens(query))
        out = []
        for i, tf in self.tf.items():
            s = sum(self.idf[w] * tf[w] * (self.k1 + 1) /
                    (tf[w] + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))) for w in q if w in tf)
            if s > 0:
                out.append((i, round(s, 4)))
        return sorted(out, key=lambda x: (-x[1], x[0]))


_VEC_CACHE: dict[str, object] = {}  # 문장 해시 → 벡터. 공고 문서 벡터는 EMB_CACHE 파일로도 저장
_DISK_LOADED = False


def _key(text: str) -> str:
    return hashlib.sha1(f"{config.EMBEDDING_MODEL}\0{text}".encode("utf-8")).hexdigest()


def _load_disk_cache():
    global _DISK_LOADED
    if _DISK_LOADED:
        return
    _DISK_LOADED = True
    try:
        import numpy as np

        if config.EMB_CACHE.exists():
            with np.load(config.EMB_CACHE) as z:
                _VEC_CACHE.update({k: z[k] for k in z.files})
    except Exception as e:  # 캐시가 깨졌으면 새로 계산
        log.warning("임베딩 캐시 로드 실패: %s", e)


def embed(texts: list[str], persist: bool = False):
    """정규화 벡터 (행렬). EMBEDDING_PROVIDER=bge(sentence-transformers 설치) 또는 azure(LLM 키) 필요 (아니면 예외).
    persist=True(공고 문서)면 새로 계산한 벡터를 EMB_CACHE에 저장한다."""
    import numpy as np

    if not config.embedding_enabled():
        raise RuntimeError(f"EMBEDDING_PROVIDER={config.EMBEDDING_PROVIDER}")
    from backend.tools.retrieval import encode

    _load_disk_cache()
    keys = [_key(t) for t in texts]
    todo = list(dict.fromkeys((k, t) for k, t in zip(keys, texts) if k not in _VEC_CACHE))
    if todo:
        vecs = encode([t for _, t in todo])
        for (k, _), v in zip(todo, vecs):
            _VEC_CACHE[k] = v
        if persist:
            try:
                config.EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
                np.savez(config.EMB_CACHE, **_VEC_CACHE)
            except Exception as e:
                log.warning("임베딩 캐시 저장 실패: %s", e)
    return np.array([_VEC_CACHE[k] for k in keys])


class DenseRetriever:
    """BGE-M3 코사인 유사도 (생성 시 임베딩 실패하면 예외 → 호출부가 keyword_fallback)."""

    def __init__(self, docs: dict[str, str]):
        self.ids = list(docs)
        self.vecs = embed([docs[i] for i in self.ids], persist=True)

    def search(self, query: str) -> list[tuple[str, float]]:
        scores = (self.vecs @ embed([query])[0]).tolist()
        return sorted(((i, round(s, 4)) for i, s in zip(self.ids, scores)), key=lambda x: (-x[1], x[0]))


def keyword_similarity(keywords: list[str], docs: dict[str, str], corpus: list[str] | None = None,
                       texts: dict[str, str] | None = None) -> tuple[dict[str, dict] | None, str | None]:
    """키워드 ↔ 공고 의미 유사도. 문서마다 상위 3개 키워드 코사인 평균(단일 최대값보다 노이즈가 적음 — 실측 AUC 0.84→0.86).
    texts(키워드 → 의미 프로필 문장)가 있으면 키워드 대신 그 문장을 임베딩한다.
    0-1 점수는 corpus(전체 공고) 분포의 SEM_LO_PCT~SEM_HI_PCT 백분위로 선형 변환 (업체마다 코사인 수준이 달라서).
    임베딩을 못 쓰면 (None, 사유) → S2는 문자열 일치만."""
    import numpy as np

    keywords = [k for k in dict.fromkeys(keywords) if k]
    if not keywords or not docs:
        return None, "키워드·후보 없음"
    try:
        kv = embed([(texts or {}).get(k, k) for k in keywords])
        sims = embed(list(docs.values()), persist=True) @ kv.T
        ref = embed(corpus, persist=True) @ kv.T if corpus else sims
    except Exception as e:
        return None, f"임베딩 미사용: {e}"
    top = min(3, len(keywords))
    agg = lambda m: np.sort(m, axis=1)[:, -top:].mean(axis=1)  # noqa: E731
    ref_agg = agg(ref)
    lo, hi = np.percentile(ref_agg, config.SEM_LO_PCT), np.percentile(ref_agg, config.SEM_HI_PCT)
    out = {}
    for i, row, cos in zip(docs, sims, agg(sims)):
        score = min(1.0, max(0.0, (cos - lo) / (hi - lo))) if hi > lo else 0.0
        out[i] = {"keyword": keywords[int(row.argmax())], "cos": round(float(cos), 4), "score": round(float(score), 3)}
    return out, None


@lru_cache
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def rrf(rankings: dict[str, list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: Σ 1/(k + rank)."""
    fused: dict[str, float] = {}
    for order in rankings.values():
        for r, i in enumerate(order, 1):
            fused[i] = fused.get(i, 0) + 1 / (k + r)
    return fused


def body_search(query: str, ids: list[str]) -> list[tuple[str, float]]:
    """PDF 본문 순위: pgvector에 저장된 공고 청크와 질의의 코사인 (공고마다 상위 BODY_TOP개 평균). 청크 없는 공고는 빠진다."""
    from backend.db import vector_store

    s = vector_store.tender_scores(embed([query])[0], ids, config.BODY_TOP)
    return sorted(((i, round(v, 4)) for i, v in s.items()), key=lambda x: (-x[1], x[0]))


@traced("hybrid_search", "retriever", big=("docs", "rule_order"),
        outputs=lambda r: {"path": r["path"], "reason": r["reason"], "lists": r["lists"],
                           "top10": [{"bidNtceNo": i, **r["scores"][i]} for i in r["order"][:10]]})
def hybrid_search(docs: dict[str, str], query: str, rule_order: list[str] | None = None) -> dict:
    """dense + BM25 (+ 룰 점수 순위)를 RRF로 합친다. 반환 order는 문서 전체, scores는 단계별 점수."""
    scores = {i: {"rule_rank": None, "dense": None, "bm25": None, "body": None, "rrf": None, "rerank": None}
              for i in docs}
    rankings = {}
    if rule_order:
        rankings["rule"] = rule_order
        for r, i in enumerate(rule_order, 1):
            scores[i]["rule_rank"] = r
    bm25 = BM25Retriever(docs).search(query)
    rankings["bm25"] = [i for i, _ in bm25]
    for i, s in bm25:
        scores[i]["bm25"] = s
    path, reason = "semantic", None
    try:
        dense = DenseRetriever(docs).search(query)
        rankings["dense"] = [i for i, _ in dense]
        for i, s in dense:
            scores[i]["dense"] = s
    except Exception as e:  # 미설치·모델 로드 실패·설정 off → BM25만
        path, reason = "keyword_fallback", f"임베딩 미사용: {e}"
        log.info(reason)
    if config.BODY_SEARCH and path == "semantic":
        try:
            body = body_search(query, list(docs))
            rankings["body"] = [i for i, _ in body]
            for i, s in body:
                scores[i]["body"] = s
        except Exception as e:  # DB 미기동 등 → 본문 순위 없이
            log.info("본문 순위 생략: %s", e)
    fused = rrf(rankings)
    for i, s in fused.items():
        scores[i]["rrf"] = round(s, 5)
    order = sorted(docs, key=lambda i: (-fused.get(i, 0), i))
    return {"path": path, "reason": reason, "query": query, "order": order, "scores": scores,
            "lists": list(rankings)}


@traced("rerank", big=("docs", "result"), outputs=lambda r: {"rerank": r["rerank"], "reason": r["rerank_reason"],
                                                     "top10": [{"bidNtceNo": i, **r["scores"][i]} for i in r["order"][:10]]})
def rerank(result: dict, docs: dict[str, str], top: int = RERANK_TOP) -> dict:
    """상위 top건만 cross-encoder로 재정렬. semantic이 아니거나 실패하면 생략(순서 유지)."""
    if result["path"] != "semantic":
        return {**result, "rerank": "skipped", "rerank_reason": "keyword_fallback 경로"}
    head = result["order"][:top]
    try:
        with activity.step("리랭크", f"리랭커 실행 중 · 후보 {len(head)}건") as done:
            raw = _reranker().predict([(result["query"], docs[i]) for i in head]).tolist()
            done["message"] = f"후보 {len(head)}건 재정렬 완료"
    except Exception as e:
        log.warning("리랭크 생략: %s", e)
        return {**result, "rerank": "skipped", "rerank_reason": f"리랭커 실패: {e}"}
    for i, s in zip(head, raw):
        result["scores"][i]["rerank"] = round(float(s), 4)
    # 리랭커 단독 순서는 룰 점수를 무시한다(실측: 룰 57.8점이 3위) → 리랭크 순위와 기존 RRF 순위를 다시 RRF
    by_rr = sorted(head, key=lambda i: -result["scores"][i]["rerank"])
    final = rrf({"rerank": by_rr, "fused": head})
    head = sorted(head, key=lambda i: (-final[i], by_rr.index(i)))
    return {**result, "order": head + result["order"][top:], "rerank": "applied", "rerank_reason": None}
