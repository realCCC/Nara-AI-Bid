"""SC-002 제안서 초안 LLM 작성: 섹션별로 업체 문서를 검색해 템플릿의 [확인 필요] 칸을 채운다.

섹션(사업 이해 / 회사 소개 / 평가항목별 수행 방안 / 일정·조직)마다 업체 문서 청크를 따로 검색해 LLM을 한 번씩 부른다.
업체 프로필(JSON)은 짧아서 매번 통째로 넣는다.

근거 검증 (코드가 판정):
- 인용(cites)이 전부 업체 문서 청크 원문에 있으면 grounded → 출처 파일 표시
- 인용이 없거나 하나라도 원문에 없으면 AI 초안 → [AI 초안·검토 필요] 표시
- 사실형 칸(인증·면허, PM, 평가항목별 업체 근거)은 grounded일 때만 채운다. 아니면 None → [확인 필요]
LLM이 없거나 섹션 호출이 실패하면 그 섹션은 비어 있고 템플릿이 [확인 필요]로 남긴다.
"""
import logging

from backend import config
from backend.tools import company_docs, llm
from backend.tools.evidence import _norm, _norm_file

log = logging.getLogger(__name__)

STEPS = ("착수", "준비·기획", "수행", "결과 보고")
MAX_ITEMS = 8   # 평가항목 LLM 호출 상한 (넘는 항목은 [확인 필요])
WORKERS = 4
DOC_K = 4       # 섹션마다 넣을 업체 문서 청크 수

COMPANY_QUERY = "보유 인증 면허 등록 비전 차별화 강점"
PLAN_QUERY = "총괄 PM 책임자 핵심 인력 경력 수행 방법론 단계 일정"


def fill(draft: dict | None, chunks: list[dict]) -> dict | None:
    """LLM Draft → {"text", "grounded", "sources"(표시용 출처 문자열)}. 인용이 전부 업체 문서 원문에 있어야 grounded."""
    text = ((draft or {}).get("text") or "").strip()
    if not text:
        return None
    body: dict[str, str] = {}
    for c in chunks:
        body[_norm_file(c["file"])] = body.get(_norm_file(c["file"]), "") + _norm(c["text"])
    cites = (draft or {}).get("cites") or []
    valid = [c for c in cites if _norm(c.get("quote")) and _norm(c.get("quote")) in body.get(_norm_file(c.get("file")), "")]
    grounded = bool(cites) and len(valid) == len(cites)
    files = list(dict.fromkeys(f"업체 문서 · {c['file']}" for c in valid)) if grounded else []
    return {"text": text, "grounded": grounded, "sources": files}


def fact(draft: dict | None, chunks: list[dict]) -> dict | None:
    """사실형 칸: grounded일 때만."""
    f = fill(draft, chunks)
    return f if f and f["grounded"] else None


def _item_query(item: dict, tender: dict) -> str:
    return " ".join([item["item"], *(item.get("sub_items") or []), tender.get("bidNtceNm", "")])


def _understanding_query(tender: dict, criteria: dict) -> str:
    return " ".join([tender.get("bidNtceNm", ""), "사업 목적 추진 전략 유사 실적",
                     *(t["text"] for t in (criteria.get("tasks") or [])[:3])])


def _clean_item(raw: dict | None, item: dict, chunks: list[dict]) -> dict | None:
    if not raw:
        return None
    allowed = set(item.get("sub_items") or [])
    subs = {s["name"]: fill(s.get("text"), chunks) for s in raw.get("sub_items") or [] if s.get("name") in allowed}
    return {"plan": fill(raw.get("plan"), chunks), "evidence": fact(raw.get("evidence"), chunks),
            "sub_items": {k: v for k, v in subs.items() if v}}


def _clean_plan(raw: dict | None, company: dict, chunks: list[dict]) -> dict:
    if not raw:
        return {"phases": {}, "staffing": {}, "pm": None}
    phases = {p["step"]: {"content": (p.get("content") or "").strip(), "period": (p.get("period") or "").strip() or None}
              for p in raw.get("phases") or [] if p.get("step") in STEPS and (p.get("content") or "").strip()}
    staff = company.get("보유인력") or {}
    staffing = {}
    for s in raw.get("staffing") or []:
        role, n = s.get("role"), s.get("count")
        if role in staff and role not in staffing:
            ok_n = n if isinstance(n, int) and 0 < n <= staff[role] else None  # 보유 인원 초과·0 이하 → 비움
            staffing[role] = {"count": ok_n, "duty": (s.get("duty") or "").strip() or None}
    return {"phases": phases, "staffing": staffing, "pm": fact(raw.get("pm"), chunks)}


def write(company: dict, tender: dict, criteria: dict) -> dict | None:
    """섹션별 LLM 초안. LLM 키가 없으면 None (템플릿 기본 동작)."""
    if not config.llm_enabled():
        return None
    from langchain_core.runnables.config import ContextThreadPoolExecutor  # 스레드에서도 LangSmith 트리 유지

    chunks = company_docs.load_chunks(company)
    vecs = company_docs.vectors(company, chunks)

    def docs(q: str) -> list[dict]:
        return company_docs.search(chunks, q, DOC_K, vecs)

    items = [i for i in criteria.get("items") or [] if i.get("item") and "가격" not in i["item"]][:MAX_ITEMS]
    with ContextThreadPoolExecutor(WORKERS) as ex:
        f_und = ex.submit(lambda: llm.draft_understanding(tender, criteria, company, docs(_understanding_query(tender, criteria))))
        f_com = ex.submit(lambda: llm.draft_company(tender, criteria, company, docs(COMPANY_QUERY)))
        f_plan = ex.submit(lambda: llm.draft_plan(tender, criteria, company, docs(PLAN_QUERY)))
        f_items = {i["item"]: ex.submit(lambda i=i: llm.draft_item(i, tender, criteria, company, docs(_item_query(i, tender))))
                   for i in items}

        def get(f):
            try:
                return f.result()
            except Exception as e:  # 섹션 하나 실패 → 그 섹션만 [확인 필요]
                log.warning("제안서 섹션 작성 실패: %s", e)
                return None

        und, com, plan = get(f_und), get(f_com), get(f_plan)
        item_raw = {k: get(f) for k, f in f_items.items()}

    by_name = {i["item"]: i for i in items}
    return {
        "understanding": fill((und or {}).get("strategy"), chunks),
        "certifications": fact((com or {}).get("certifications"), chunks),
        "vision": fill((com or {}).get("vision"), chunks),
        "items": {k: v for k, raw in item_raw.items() if (v := _clean_item(raw, by_name[k], chunks))},
        **_clean_plan(plan, company, chunks),
        "doc_files": list(dict.fromkeys(c["file"] for c in chunks)),
        "doc_search": "hybrid" if vecs is not None else ("bm25" if chunks else "none"),
    }
