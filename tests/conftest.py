import pytest
from langsmith import tracing_context

from backend import config
from backend.tools import keyword_profile


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """.env에 Azure 키가 있어도 테스트는 실제 LLM을 부르지 않는다 (LLM 경로는 각 테스트가 monkeypatch로 켠다)."""
    monkeypatch.setattr(config, "llm_enabled", lambda: False)
    keyword_profile._CACHE.clear()
    with tracing_context(enabled=False):  # .env에 LANGSMITH_TRACING=true가 있어도 테스트는 트레이스를 보내지 않는다
        yield
    keyword_profile._CACHE.clear()
