"""SC-002 제안서 초안: 고정 템플릿 + 값 삽입. 서술형 칸은 proposal_writer(LLM)가 채운 값(ai)을 넣고,
채울 수 없는 곳은 [확인 필요], 근거 없이 LLM이 쓴 문장은 [AI 초안·검토 필요]."""
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from backend.tools.evidence import _norm, check_reason
from backend.tools.rules import tokenize, won

_env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parent.parent / "templates"),
                   autoescape=select_autoescape(["html", "j2"]))
_env.globals["won"] = won

DOWNLOAD_STYLE = """body{font-family:"Malgun Gothic",sans-serif;max-width:800px;margin:40px auto;line-height:1.6;color:#171c26}
.confirm{background:#fff4e0;color:#b45309;font-weight:700;padding:0 4px}.ai-draft{background:#eef2ff;color:#4338ca;font-weight:700;padding:0 4px;font-size:12px}h4{color:#2563eb;margin-top:20px}
.muted{color:#646f7f}.src{color:#646f7f;font-size:12px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #e5e8ef;padding:6px 8px;text-align:left}"""


def fallback_criteria(tender: dict) -> dict:
    """LLM 추출 실패·키 없음: 기술·가격 평가 비율 필드만 사용 (4번 문서 6장)."""
    tech, price = tender.get("techAbltEvlRt"), tender.get("bidPrceEvlRt")
    if tech or price:
        items = [{"item": "기술 능력 평가", "score": float(tech) if tech else None, "source": "필드 · techAbltEvlRt"},
                 {"item": "가격 평가", "score": float(price) if price else None, "source": "필드 · bidPrceEvlRt"}]
    else:
        items = [{"item": "평가항목", "score": None, "source": "추출 실패"}]
    return {"items": items, "period": None, "tasks": [], "source": "fallback"}


# 수행 기간: "사업기간 2026. 착수일 ~ 2026.10.31." / "계약기간: 계약일로부터 3개월" / "용역기간 … 까지" (제출기간은 제외)
_PERIOD = re.compile(r"(?:과업|사업|용역|계약|수행)\s*기간\s*[:：]?\s*"
                     r"([^~]{0,30}?~\s*(?:\d{4}\s*[.년]\s*)?\d{1,2}\s*[.월]\s*\d{1,2}\s*[.일]?"
                     r"|[^.]{0,30}?\d+\s*개월(?:간)?|[^.]{0,30}?까지)")


def find_period(chunks) -> str | None:
    """LLM이 수행 기간을 못 찾았을 때 청크 원문에서 정규식으로 찾는다 (첫 번째 일치)."""
    for c in chunks:
        m = _PERIOD.search(c["text"])
        if m:
            return f"{m.group(1).strip()} ({_pdf_source(c)})"
    return None


def _pdf_source(c: dict) -> str:
    return f"PDF · {c['file']} p.{c['page']}" if c.get("file") else "PDF"


def normalize_criteria(extracted: dict | None, tender: dict, chunks: list[dict] = ()) -> dict:
    """세부항목은 청크 본문에 있는 이름만, 과업 범위는 인용이 원문에 있는 것만 남긴다."""
    if not extracted:
        return {**fallback_criteria(tender), "period": find_period(chunks)}
    body = _norm(" ".join(c["text"] for c in chunks))
    items = [{"item": c["item"], "score": c.get("score"), "source": _pdf_source(c),
              "sub_items": [x for x in c.get("sub_items") or [] if _norm(x) and _norm(x) in body]}
             for c in extracted["items"]]
    tasks = [{"text": t["text"].strip()[:120], "source": _pdf_source(t)} for t in extracted.get("tasks") or []
             if (t.get("text") or "").strip()
             and check_reason({"source_type": "pdf", **{k: t.get(k) for k in ("file", "page", "quote")}}, tender, {}, chunks)]
    return {"items": items, "period": extracted.get("period") or find_period(chunks), "tasks": tasks, "source": "llm"}


def _int(v) -> int | None:
    try:
        return int(float(v)) if v not in (None, "") else None
    except ValueError:
        return None


def overview(tender: dict, criteria: dict) -> list[tuple[str, str]]:
    """사업 개요: 공고 필드에 값이 있는 것만 (필드명이 곧 출처)."""
    budget, est = _int(tender.get("asignBdgtAmt") or tender.get("budget")), _int(tender.get("presmptPrce"))
    tech, price = tender.get("techAbltEvlRt"), tender.get("bidPrceEvlRt")
    rows = [("사업명", tender.get("bidNtceNm")),
            ("수요기관", tender.get("dminsttNm")),
            ("공고기관", tender.get("ntceInsttNm")),
            ("사업 예산(배정)", won(budget) if budget else None),
            ("추정가격", won(est) if est else None),
            ("용역 구분", tender.get("srvceDivNm")),
            ("계약 방법", tender.get("cntrctCnclsMthdNm")),
            ("낙찰 방법", tender.get("sucsfbidMthdNm")),
            ("공동수급", tender.get("cmmnSpldmdMethdNm")),
            ("평가 비율", f"기술 {tech} / 가격 {price}" if tech and price else None),
            ("과업 기간", criteria.get("period"))]
    return [(k, str(v)) for k, v in rows if v]


def bid_schedule(tender: dict) -> list[tuple[str, str]]:
    rows = [("공고 게시", "bidNtceDt"), ("입찰참가자격 등록 마감", "bidQlfctRgstDt"),
            ("공동수급협정서 마감", "cmmnSpldmdAgrmntClseDt"), ("입찰서 제출 시작", "bidBeginDt"),
            ("입찰서 제출 마감", "bidClseDt"), ("개찰", "opengDt")]
    return [(label, tender[f][:16]) for label, f in rows if tender.get(f)]


def related_history(tender: dict, company: dict) -> list[dict]:
    """공고명과 키워드가 겹치는 실적 → 회사 소개에서 '유사 실적'으로 먼저 보여 준다."""
    words = tokenize(tender.get("bidNtceNm", ""))
    return [{**h, "related": bool(words & (set(h.get("키워드", [])) | tokenize(h.get("프로젝트명", ""))))}
            for h in company.get("과거실적", [])]


def _reason_source(r: dict) -> str:
    if r.get("source_type") == "pdf":
        return _pdf_source(r)
    return f"필드 · {r['field']}" if r.get("field") else "필드"


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


def _mapping(criteria: dict, tender: dict, company: dict, ai: dict) -> list[dict]:
    """평가항목 + 업체 근거(프로필 룰 우선, 없으면 업체 문서 근거) + LLM 수행 방안·세부항목 대응."""
    out = []
    for c in criteria["items"]:
        draft = (ai.get("items") or {}).get(c["item"]) or {}
        rule = _evidence(c["item"], tender, company)
        # 대화로 고친 근거(evidence_edited)가 있으면 그것을, 아니면 프로필 룰 → 업체 문서 근거 순
        if rule and not draft.get("evidence_edited"):
            evidence = {"text": rule, "grounded": True, "sources": ["업체 프로필"]}
        else:
            evidence = draft.get("evidence")
        out.append({**c, "evidence": evidence, "plan": draft.get("plan"), "sub_drafts": draft.get("sub_items") or {}})
    return out


def assemble_proposal(company: dict, tender: dict, criteria: dict, rec: dict | None = None,
                      checks: list[dict] = (), ai: dict | None = None) -> str:
    ai = ai or {}
    mapping = _mapping(criteria, tender, company, ai)
    history = sorted(related_history(tender, company), key=lambda h: not h["related"])
    reasons = [{"text": r["text"], "source": _reason_source(r), "quote": r.get("quote")}
               for r in (rec or {}).get("reasons", [])]
    staff = company.get("보유인력") or {}
    return _env.get_template("proposal.html.j2").render(
        company=company, tender=tender, criteria=criteria, mapping=mapping, overview=overview(tender, criteria),
        schedule=bid_schedule(tender), history=history, reasons=reasons, checks=list(checks),
        staff=staff, staff_total=sum(staff.values()), ai=ai)


def to_download_html(body: str, title: str) -> str:
    from html import escape

    return (f'<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>{escape(title)} 제안서 초안</title>'
            f"<style>{DOWNLOAD_STYLE}</style></head><body>{body}</body></html>")
