"""근거 출처 대조: 인용은 해당 파일·페이지 청크에, 필드 값은 실제 필드에 있어야 통과."""
import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def check_reason(reason: dict, tender: dict, company: dict, chunks: list[dict]) -> bool:
    if reason.get("source_type") == "pdf":
        quote = _norm(reason.get("quote"))
        if len(quote) < 4:
            return False
        return any(c["file"] == reason.get("file") and c["page"] == reason.get("page") and quote in _norm(c["text"])
                   for c in chunks)
    if reason.get("source_type") == "field":
        scope, _, name = (reason.get("field") or "").partition(".")
        src = tender if scope == "tender" else company if scope == "company" else None
        if src is None or name not in src:
            return False
        value = _norm(str(reason.get("value", "")))
        return value == "" or all(_norm(v) in _norm(str(src[name])) for v in str(reason.get("value", "")).split(","))
    return False


def check_evidence_sources(reasons: list[dict], tender: dict, company: dict, chunks: list[dict]) -> list[dict]:
    return [r for r in reasons if check_reason(r, tender, company, chunks)]
