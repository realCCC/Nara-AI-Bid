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
SCORE_WEIGHTS = tuple(float(w) for w in os.getenv("SCORE_WEIGHTS", "0.35,0.30,0.20,0.15").split(","))
TOP_K = int(os.getenv("TOP_K", "10"))
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
