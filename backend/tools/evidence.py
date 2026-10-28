"""근거 출처 대조: 인용은 해당 파일·페이지 청크에, 필드 값은 실제 필드에 있어야 통과."""
import re

from backend.tracing import traced


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def check_reason(reason: dict, tender: dict, company: dict, chunks: list[dict]) -> bool:
    if reason.get("source_type") == "pdf":
        # 줄임표(... …)로 줄인 인용은 조각마다 같은 청크에 있어야 통과
        quote = re.sub(r"^\s*(\([^)]*p\.\s*\d+\)\s*)?(\[[^\]]*\]\s*)?", "", reason.get("quote") or "")  # LLM이 붙인 (파일 p.쪽)·[섹션] 표시는 뗀다
        parts = [p for p in (_norm(s) for s in re.split(r"\.{2,}|…", quote)) if p]
        if sum(map(len, parts)) < 4:
            return False
        page = reason.get("page")  # 페이지를 넘어가는 청크는 걸친 페이지 어디를 적어도 통과
        return any(c["file"] == reason.get("file") and (c["page"] == page or page in c.get("pages", ()))
                   and all(p in _norm(c["text"]) for p in parts) for c in chunks)
    if reason.get("source_type") == "field":
        scope, _, name = (reason.get("field") or "").partition(".")
        src = tender if scope == "tender" else company if scope == "company" else None
        if src is None or name not in src:
            return False
        value = _norm(str(reason.get("value", "")))
        return value == "" or all(_norm(v) in _norm(str(src[name])) for v in str(reason.get("value", "")).split(","))
    return False


@traced("check_evidence_sources", big=("tender", "chunks"), outputs=lambda kept: {"kept": kept})
def check_evidence_sources(reasons: list[dict], tender: dict, company: dict, chunks: list[dict]) -> list[dict]:
    return [r for r in reasons if check_reason(r, tender, company, chunks)]
