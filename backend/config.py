"""환경변수(.env) 읽기. 경로·모델명은 여기서만 읽는다."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(key: str, default: str) -> Path:
    p = Path(os.getenv(key) or default)
    return p if p.is_absolute() else ROOT / p


TENDERS_FILE = _path("TENDERS_FILE", "data/raw/tenders_202605.json")
COMPANIES_DIR = _path("COMPANIES_DIR", "data/companies")
# 업체 문서(회사소개서 등 txt/md/pdf): <COMPANY_DOCS_DIR>/<업체ID>/ — 제안서 초안을 LLM이 채울 때 근거로 검색
COMPANY_DOCS_DIR = _path("COMPANY_DOCS_DIR", "data/companies/docs")
REGION_LIMITS_FILE = _path("REGION_LIMITS_FILE", "data/region_limits.csv")
PDF_DIR = _path("PDF_DIR", "data/pdfs")
CHROMA_DIR = _path("CHROMA_DIR", "data/chroma")
SQLITE_PATH = _path("SQLITE_PATH", "data/app.db")

# keyword(임베딩 없음) | bge(로컬 BGE-M3) | azure(Azure OpenAI 임베딩 API, EMBEDDING_MODEL = 배포 이름)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "keyword")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
# azure 벡터 차원: pgvector 스키마 vector(1024)·HNSW 2000차원 제한 → text-embedding-3의 dimensions로 줄여 받는다
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# S2 의미 유사도: 업체 키워드↔전체 공고 코사인 분포의 SEM_LO_PCT 백분위 이하 0점, SEM_HI_PCT 이상 1점
# (BGE-M3 실측: 코사인 절대값이 업체마다 0.40~0.66으로 달라 고정 임계값 대신 업체별 백분위 사용)
SEM_LO_PCT = float(os.getenv("SEM_LO_PCT", "50"))
SEM_HI_PCT = float(os.getenv("SEM_HI_PCT", "97"))
# RAG 청크 검색에 BGE-M3 dense도 쓴다 (BM25와 RRF). 벡터는 상위 공고의 청크만 계산해 pgvector에 저장
CHUNK_EMBEDDING = os.getenv("CHUNK_EMBEDDING", "1") == "1"
# 청크 벡터 DB (docker compose up -d). 연결 실패 시 청크 검색은 BM25만
PGVECTOR_DSN = os.getenv("PGVECTOR_DSN", "postgresql://postgres:postgres@localhost:5432/ai_master")
# 공고 검색(1단계)에 PDF 본문 순위도 RRF로 넣는다 (공고마다 질의↔청크 코사인 상위 BODY_TOP개 평균).
# 청크 벡터를 미리 계산해야 함: scripts/warm_chunk_vectors.py --all. DB를 못 쓰면 본문 순위 없이 진행
BODY_SEARCH = os.getenv("BODY_SEARCH", "0") == "1"
BODY_TOP = int(os.getenv("BODY_TOP", "3"))
QUERY_CACHE = _path("QUERY_CACHE", "data/query_cache.json")  # 업체 프로필별 LLM 검색 문장 (실행마다 바뀌지 않게 고정)
VERIFY_GOLD = _path("VERIFY_GOLD", "data/eval/verify_gold.json")  # 검증 정답셋 (평가 스크립트 전용, 런타임 판정에는 쓰지 않음)
EMB_CACHE = _path("EMB_CACHE", "data/emb_cache.npz")  # 공고 문서 벡터 디스크 캐시 (CPU 443건 첫 계산 약 2-3분)
SCORE_WEIGHTS = tuple(float(w) for w in os.getenv("SCORE_WEIGHTS", "0.35,0.30,0.20,0.15").split(","))
TOP_K = int(os.getenv("TOP_K", "10"))
# 검증 서브에이전트가 재평가할 상위 건수 (1~10). 확인된 부적격은 맨 뒤로 빠지므로, 빈자리를 채우는 공고도 검증되게 상위 10건 전체
VERIFY_TOP_N = int(os.getenv("VERIFY_TOP_N", "10"))
# 검증(5단계) 결과와 4단계 순위가 어긋나면(자체 점검 불통과) 비평 프롬프트로 공고서 근거 순위(4단계)를 다시 돌리는 최대 횟수 (0~2)
VERIFY_MAX_RETRY = max(0, min(int(os.getenv("VERIFY_MAX_RETRY", "2")), 2))
# ToT: 4단계 순위안을 관점별로 몇 개 만들지 (1이면 끔, 최대 3). 5단계 검증 결과로 가장 나은 안을 고른다
TOT_BRANCHES = max(1, min(int(os.getenv("TOT_BRANCHES", "3")), 3))
FETCH_PDFS = os.getenv("FETCH_PDFS", "1") == "1"

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
AZURE_KEYS = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "OPENAI_API_VERSION", "AZURE_OPENAI_CHAT_DEPLOYMENT")


def embedding_enabled() -> bool:
    return EMBEDDING_PROVIDER in ("bge", "azure")


def llm_enabled() -> bool:
    return all(os.getenv(k) for k in AZURE_KEYS)


def chat_model_name() -> str:
    """화면 표시용 채팅 모델(배포) 이름."""
    return os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or "-"


@lru_cache
def get_llm():
    """Azure OpenAI 채팅 모델. 키가 없으면 None (호출부가 룰로 폴백)."""
    if not llm_enabled():
        return None
    from langchain_openai import AzureChatOpenAI

    # 추론 모델(gpt-5.6-luna·gpt-6 계열)은 temperature 지정을 거부한다 → LLM_TEMPERATURE를 비우면 넘기지 않음
    temp = os.getenv("LLM_TEMPERATURE", "0")
    return AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"], timeout=LLM_TIMEOUT,
                           **({"temperature": float(temp)} if temp else {}))
