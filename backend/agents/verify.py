"""검증 서브에이전트: 상위 N건을 참가 조건(필드·자격 청크) 기준으로 다시 확인. 실패해도 원순위로 계속.

추천 단계는 업체 역량 쪽 청크를 보고 순위를 매긴다. 검증은 자격 쪽 청크를 따로 검색해 넘겨,
앞 단계가 놓친 불충족 조건(issues)을 찾는다. issues는 출처 대조를 통과한 것만 남기고,
남은 issue가 있는 공고는 상위 N건 안에서 뒤로 보낸다.
"""
import json
import re

from langchain_core.runnables import RunnableConfig

from backend import config as settings
from backend.agents.state import ProposalState, save, tenders
from backend.tools import llm
from backend.tools.checklist import QUERY as QUALIFICATION_QUERY
from backend.tools.evidence import check_reason
from backend.tools.interview import with_note
from backend.tools.pdf import tender_chunks
from backend.tools.retrieval import retrieve
from backend.tracing import traced

QUALIFICATION_K = 6
_QUAL_SECTION = re.compile(r"자격")  # 섹션 제목 (조항 구조 청킹이 붙임)
# 참가자격 섹션 밖(입찰방법·계약방법 등)에 적히는 핵심 조건
_QUAL_TEXT = re.compile(r"지역제한|주된\s*영업소|본점\s*소재지|대기업|중견기업|중소기업\S{0,10}\s*(한정|확인서)|업종코드")
# 자격 충돌로 인정하는 업체 항목 (선호계약방식 같은 선호, 프로필에 없는 소기업 여부 등은 issue가 아님)
CONFLICT_FIELDS = ("소재지", "중소기업여부", "업종", "참여가능금액")


def qualification_chunks(head: list[dict], by_no: dict[str, dict]) -> dict[str, list[dict]]:
    """공고마다 자격 청크 QUALIFICATION_K개: ① '참가자격' 섹션 청크 ② 지역제한·대기업·중견 등 핵심 조건 문장이 있는 청크
    ③ 남는 자리는 검색(체크리스트와 같은 질의). 검색만 쓰면 자격 섹션을 놓치는 일이 많았다(정답셋 부적격 6건 중 4건).
    상위 N건은 RAG 상위 10건 안이라 청크 벡터가 이미 저장돼 있다 → tender_no로 그 벡터를 다시 쓴다."""
    out = {}
    for c in head:
        no = c["bidNtceNo"]
        chunks = tender_chunks(by_no[no])
        picked = [ch for ch in chunks if _QUAL_SECTION.search(ch.get("section") or "")]
        picked += [ch for ch in chunks if _QUAL_TEXT.search(ch["text"]) and ch not in picked]
        picked = picked[:QUALIFICATION_K]
        if len(picked) < QUALIFICATION_K:
            picked += [ch for ch in retrieve(chunks, QUALIFICATION_QUERY, k=QUALIFICATION_K, tender_no=no)
                       if ch not in picked][:QUALIFICATION_K - len(picked)]
        out[no] = picked
    return out


def _conflicts(issue: dict, company: dict) -> bool:
    """충돌한다고 적은 업체 프로필 값이 실제 프로필에 있어야 한다 ('확인 불가'류 issue를 거른다)."""
    name = (issue.get("company_field") or "").strip().removeprefix("company.")  # LLM이 접두어를 빼먹기도 한다
    value = re.sub(r"\s+", "", str(issue.get("company_value") or "")).lower()
    if name == "중소기업여부" and company.get(name) is not False:  # 중소기업이면 규모 제한과 충돌할 일이 없다 (소기업 여부는 모름)
        return False
    # 공백은 무시: LLM은 업종 값을 공백 없는 JSON({"대분류":[...]})으로 적는데 json.dumps는 ': '·', '를 넣는다
    return (name in CONFLICT_FIELDS and name in company and value != ""
            and value in re.sub(r"\s+", "", json.dumps(company[name], ensure_ascii=False)).lower())


@traced("confirm_issues", big=("by_no", "chunks"),
        outputs=lambda r: {"dropped": r[1], "issues": {i["bidNtceNo"]: i.get("issues", []) for i in r[0]["items"]}})
def confirm_issues(result: dict, by_no: dict[str, dict], company: dict,
                   chunks: dict[str, list[dict]]) -> tuple[dict, int]:
    """issues 중 공고 근거(인용·필드 값)가 원문에 있고, 충돌하는 업체 값이 프로필에 있는 것만 남긴다. (결과, 버린 개수)"""
    dropped, items = 0, []
    for it in result["items"]:
        no = it["bidNtceNo"]
        kept = [i for i in it.get("issues", [])
                if check_reason(i, by_no.get(no, {}), company, chunks.get(no, [])) and _conflicts(i, company)]
        dropped += len(it.get("issues", [])) - len(kept)
        items.append({**it, "issues": kept})
    return {**result, "items": items}, dropped


def apply_verification(recs: list[dict], result: dict | None, reason: str | None, n: int,
                       dropped: int = 0) -> tuple[list[dict], dict]:
    """검증 결과 반영. 실패(result None)면 원순위 그대로 + 사유.

    순서는 LLM이 조정한 order를 따르되, 확인된 issue가 있는 공고는 상위 N건 안에서 뒤로 보낸다.
    """
    head = recs[:n]
    if result is None:
        return recs, {"status": "skipped", "reason": reason, "checked": len(head), "changed": False,
                      "flagged": 0, "issues_dropped": 0}
    by_no, items = {c["bidNtceNo"]: c for c in head}, {i["bidNtceNo"]: i for i in result["items"]}
    order = sorted(result["order"], key=lambda no: bool(items[no].get("issues")))  # 안정 정렬: issue 없는 공고 먼저
    new_head = [{**by_no[no], "verify": {"fit_score": items[no]["fit_score"], "comment": items[no]["comment"],
                                         "issues": items[no].get("issues", []),
                                         "rank_before": by_no[no]["rank"]}} for no in order]
    out = [{**r, "rank": i} for i, r in enumerate(new_head + recs[n:], 1)]
    changed = [c["bidNtceNo"] for c in head] != order
    flagged = sum(bool(items[no].get("issues")) for no in order)
    return out, {"status": "passed", "reason": None, "checked": len(head), "changed": changed,
                 "flagged": flagged, "issues_dropped": dropped}


def verify_recommendations(state: ProposalState, config: RunnableConfig):
    recs, n = state["recommendations"], max(1, min(settings.VERIFY_TOP_N, 5))
    dropped = 0
    if not settings.llm_enabled():
        result, reason = None, "LLM 키 없음"
    else:
        try:
            by_no = tenders(state)
            chunks = qualification_chunks(recs[:n], by_no)
            result, reason = llm.verify_ranking(with_note(state["company_profile"], state.get("prefs")), recs[:n],
                                                by_no, chunks)
            if result is not None:
                result, dropped = confirm_issues(result, by_no, state["company_profile"], chunks)
        except Exception as e:  # 서브에이전트 예외는 그래프를 멈추지 않는다
            result, reason = None, f"예외: {type(e).__name__}"
    recs, verification = apply_verification(recs, result, reason, n, dropped)
    save(config, state, "recommended", recommendations=recs)
    return {"recommendations": recs, "verification": verification}
