"""Azure OpenAI 연결 확인. uv run python scripts/check_env.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

missing = [k for k in config.AZURE_KEYS if not os.getenv(k)]
if missing:
    print("누락된 환경변수:", ", ".join(missing), "→ .env에 입력하세요 (.env.example 참고)")
    sys.exit(1)
try:
    reply = config.get_llm().invoke("한 단어로 답하세요: 연결 확인").content
except Exception as e:
    print("FAIL:", type(e).__name__, e)
    sys.exit(1)
print(f"OK: {os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT']} 응답 수신 ({reply.strip()[:20]})")
