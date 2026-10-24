"""SC-002 제안서 초안: 고정 템플릿 + 값 삽입 (LLM 미사용). 채울 수 없는 곳은 [확인 필요]."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from backend.tools.rules import tokenize, won

_env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parent.parent / "templates"),
                   autoescape=select_autoescape(["html", "j2"]))
_env.globals["won"] = won

DOWNLOAD_STYLE = """body{font-family:"Malgun Gothic",sans-serif;max-width:800px;margin:40px auto;line-height:1.6;color:#171c26}
.confirm{background:#fff4e0;color:#b45309;font-weight:700;padding:0 4px}h4{color:#2563eb;margin-top:20px}
.muted{color:#646f7f}table{border-collapse:collapse;width:100%}th,td{border:1px solid #e5e8ef;padding:6px 8px;text-align:left}"""


def fallback_criteria(tender: dict) -> dict:
    """LLM 추출 실패·키 없음: 기술·가격 평가 비율 필드만 사용 (4번 문서 6장)."""
    tech, price = tender.get("techAbltEvlRt"), tender.get("bidPrceEvlRt")
    if tech or price:
        items = [{"item": "기술 능력 평가", "score": float(tech) if tech else None, "source": "필드 · techAbltEvlRt"},
                 {"item": "가격 평가", "score": float(price) if price else None, "source": "필드 · bidPrceEvlRt"}]
    else:
        items = [{"item": "평가항목", "score": None, "source": "추출 실패"}]
    return {"items": items, "period": None, "source": "fallback"}


def normalize_criteria(extracted: dict | None, tender: dict) -> dict:
    if not extracted:
        return fallback_criteria(tender)
    items = [{"item": c["item"], "score": c.get("score"),
              "source": f"PDF · {c['file']} p.{c['page']}" if c.get("file") else "PDF"} for c in extracted["items"]]
    return {"items": items, "period": extracted.get("period"), "source": "llm"}


def _evidence(item: str, tender: dict, company: dict) -> str | None:
    """평가항목별 업체 근거 — 프로필에 있는 사실만."""
    if any(k in item for k in ("인력", "조직", "강사", "전문가")) and company.get("보유인력"):
        return " · ".join(f"{k} {v}명" for k, v in company["보유인력"].items())
    if "가격" in item:
        return None
    words = tokenize(item) | tokenize(tender.get("bidNtceNm", ""))
    related = [h for h in company.get("과거실적", [])
               if words & (set(h.get("키워드", [])) | tokenize(h.get("프로젝트명", "")))]
    if related:
        return ", ".join(f"{h['프로젝트명']} ({won(h['계약금액'])})" for h in related[:2])
    return None


def assemble_proposal(company: dict, tender: dict, criteria: dict) -> str:
    mapping = [{**c, "evidence": _evidence(c["item"], tender, company)} for c in criteria["items"]]
    return _env.get_template("proposal.html.j2").render(company=company, tender=tender, criteria=criteria, mapping=mapping)


def to_download_html(body: str, title: str) -> str:
    from html import escape

    return (f'<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>{escape(title)} 제안서 초안</title>'
            f"<style>{DOWNLOAD_STYLE}</style></head><body>{body}</body></html>")
