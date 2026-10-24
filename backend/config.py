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
REGION_LIMITS_FILE = _path("REGION_LIMITS_FILE", "data/region_limits.csv")
PDF_DIR = _path("PDF_DIR", "data/pdfs")
CHROMA_DIR = _path("CHROMA_DIR", "data/chroma")
SQLITE_PATH = _path("SQLITE_PATH", "data/app.db")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "keyword")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
# S2 의미 유사도: 업체 키워드↔전체 공고 코사인 분포의 SEM_LO_PCT 백분위 이하 0점, SEM_HI_PCT 이상 1점
# (BGE-M3 실측: 코사인 절대값이 업체마다 0.40~0.66으로 달라 고정 임계값 대신 업체별 백분위 사용)
SEM_LO_PCT = float(os.getenv("SEM_LO_PCT", "50"))
SEM_HI_PCT = float(os.getenv("SEM_HI_PCT", "97"))
CHUNK_EMBEDDING = os.getenv("CHUNK_EMBEDDING", "0") == "1"  # PDF 청크 검색도 BGE-M3로 (CPU면 느림)
EMB_CACHE = _path("EMB_CACHE", "data/emb_cache.npz")  # 공고 문서 벡터 디스크 캐시 (CPU 443건 첫 계산 약 2-3분)
SCORE_WEIGHTS = tuple(float(w) for w in os.getenv("SCORE_WEIGHTS", "0.35,0.30,0.20,0.15").split(","))
TOP_K = int(os.getenv("TOP_K", "10"))
VERIFY_TOP_N = int(os.getenv("VERIFY_TOP_N", "5"))  # 검증 서브에이전트가 재평가할 상위 건수 (3~5)
FETCH_PDFS = os.getenv("FETCH_PDFS", "1") == "1"

AZURE_KEYS = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "OPENAI_API_VERSION", "AZURE_OPENAI_CHAT_DEPLOYMENT")


def llm_enabled() -> bool:
    return all(os.getenv(k) for k in AZURE_KEYS)


@lru_cache
def get_llm():
    """Azure OpenAI 채팅 모델. 키가 없으면 None (호출부가 룰로 폴백)."""
    if not llm_enabled():
        return None
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"], temperature=0, timeout=60)
