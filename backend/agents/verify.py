"""검증 서브에이전트: 상위 N건을 참가 조건(필드·자격 청크) 기준으로 다시 확인. 실패해도 원순위로 계속.

추천 단계는 업체 역량 쪽 청크를 보고 순위를 매긴다. 검증은 자격 쪽 청크를 따로 검색해 넘겨,
앞 단계가 놓친 불충족 조건(issues)을 찾는다. issues는 출처 대조를 통과한 것만 남기고,
남은 issue가 있는 공고는 검증하지 않은 공고까지 포함한 전체 목록의 맨 뒤로 보낸다.
(상위 N건 안에서만 뒤로 보내면 확인된 부적격이 상위 5건 자리를 계속 차지했다 — 진단 C005: 지역제한 부적격이 4·5위)
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
                       dropped: int = 0, keep_order: bool = False) -> tuple[list[dict], dict]:
    """검증 결과 반영. 실패(result None)면 원순위 그대로 + 사유.

    순서는 LLM이 조정한 order를 따르되, 확인된 issue가 있는 공고는 검증 밖 공고(recs[n:])보다도 뒤로 보낸다.
    keep_order(ToT 순위안)면 검증의 order 대신 순위안 순서를 그대로 두고 issue 공고만 뒤로 보낸다 — 검증은 평가자 역할.
    LLM이 재평가를 일부 빠뜨린 공고(result["unchecked"])는 verify 없이 그 자리에 둔다.
    """
    head = recs[:n]
    if result is None:
        return recs, {"status": "skipped", "reason": reason, "checked": len(head), "changed": False,
                      "flagged": 0, "issues_dropped": 0}
    by_no, items = {c["bidNtceNo"]: c for c in head}, {i["bidNtceNo"]: i for i in result["items"]}
    has_issue = lambda no: bool(items.get(no, {}).get("issues"))  # noqa: E731
    order = sorted(list(by_no) if keep_order else result["order"], key=has_issue)  # 안정 정렬: issue 없는 공고 먼저
    checked = {no: {**by_no[no], "verify": {"fit_score": items[no]["fit_score"], "comment": items[no]["comment"],
                                            "issues": items[no].get("issues", []),
                                            "rank_before": by_no[no]["rank"]}} if no in items else by_no[no]
               for no in order}
    clean = [checked[no] for no in order if not has_issue(no)]
    flagged_recs = [checked[no] for no in order if has_issue(no)]
    out = [{**r, "rank": i, "journey": {**r.get("journey", {}), "final": i}}
           for i, r in enumerate(clean + recs[n:] + flagged_recs, 1)]
    changed = [c["bidNtceNo"] for c in head] != order
    flagged = sum(has_issue(no) for no in order)
    return out, {"status": "passed", "reason": None, "checked": len(head), "changed": changed,
                 "flagged": flagged, "issues_dropped": dropped, "unchecked": len(result.get("unchecked", []))}


# ---------- 자체 점검: 4단계 순위를 5단계 검증 결과와 맞춰 보고, 어긋나면 비평 프롬프트로 4단계를 다시 돌린다 ----------
# 정답셋은 평가 전용이다. 런타임 판정에 쓰면 라벨이 없는 실제 공고에서는 루프가 돌지 않고,
# 비평에 정답 사유가 들어가 같은 정답셋으로 잰 KPI가 부풀려진다.
FIT_GAP = 20  # 검증 적합도가 이만큼 높은 공고가 4단계에서 더 아래면 순위 역전


def judge(recs: list[dict]) -> list[dict]:
    """검증까지 반영한 추천 목록에서 4단계(LLM 순위)와 5단계(검증)가 어긋난 곳 → [{bidNtceNo, kind, text}]. 비면 통과.
    - 자격 누락: 검증이 출처를 확인한 자격 문제(issue)가 있는데 4단계가 qualification_flags 없이 순위를 매김
    - 근거 미확인: 4단계가 쓴 근거가 원문 대조에서 전부 탈락
    - 순위 역전: 자격 문제 없는 공고끼리, 검증 적합도가 FIT_GAP 이상 높은 공고가 4단계에서 더 아래
    검증 결과(verify)와 LLM 순위(llm)가 둘 다 있는 공고만 본다."""
    fails, checked = [], [r for r in recs if r.get("verify") and r.get("llm")]
    for r in checked:
        v, l, name = r["verify"], r["llm"], r.get("bidNtceNm") or r["bidNtceNo"]
        if v.get("issues") and not l.get("qualification_flags"):
            fails.append({"bidNtceNo": r["bidNtceNo"], "kind": "자격 누락",
                          "text": f"'{name}'(4단계 {l['rank']}위)는 검증에서 출처가 확인된 자격 문제가 나왔는데 "
                                  f"자격 의심 없이 순위를 매김: {v['issues'][0].get('text', '')}"})
        if not l.get("reasons") and l.get("dropped"):
            fails.append({"bidNtceNo": r["bidNtceNo"], "kind": "근거 미확인",
                          "text": f"'{name}'(4단계 {l['rank']}위)의 근거 {l['dropped']}개가 모두 원문 대조에 실패함"})
    clean = sorted((r for r in checked if not r["verify"].get("issues")), key=lambda r: r["llm"]["rank"])
    fit = lambda r: r["verify"]["fit_score"]  # noqa: E731
    for j, lo in enumerate(clean):
        above = [hi for hi in clean[:j] if fit(lo) - fit(hi) >= FIT_GAP]
        if above:
            fails.append({"bidNtceNo": lo["bidNtceNo"], "kind": "순위 역전",
                          "text": f"'{lo.get('bidNtceNm') or lo['bidNtceNo']}'(4단계 {lo['llm']['rank']}위, "
                                  f"검증 적합도 {fit(lo)})가 "
                                  + "·".join(f"{hi['llm']['rank']}위(적합도 {fit(hi)})" for hi in above) + "보다 아래"})
    return fails


def pick_branch(branches: list[dict], result: dict, reason: str | None, n: int, dropped: int):
    """ToT: 순위안마다 검증 결과를 적용해 자체 점검 → 불통과가 가장 적은 안, 같으면 상위 3건 검증 적합도 합이 큰 안.
    → (추천 목록, verification, 비교표). 검증 LLM은 한 번만 돈다 (공고별 issue·적합도는 순서와 무관)."""
    scored = []
    for b in branches:
        recs, ver = apply_verification(b["recommendations"], result, reason, n, dropped, keep_order=True)
        top_fit = sum(r.get("verify", {}).get("fit_score", 0) for r in recs[:3])
        scored.append({"perspective": b["perspective"], "fails": len(judge(recs)), "top_fit": top_fit,
                       "top3": [r["bidNtceNm"] for r in recs[:3]], "_recs": recs, "_ver": ver})
    chosen = min(range(len(scored)), key=lambda i: (scored[i]["fails"], -scored[i]["top_fit"], i))
    table = [{k: v for k, v in s.items() if not k.startswith("_")} | {"chosen": i == chosen} for i, s in enumerate(scored)]
    return scored[chosen]["_recs"], {**scored[chosen]["_ver"], "changed": chosen != 0}, table


def _top_n() -> int:
    return max(1, min(settings.VERIFY_TOP_N, 10))


def _order_text(recs: list[dict]) -> str:
    return "\n".join(f"- {r['rank']}위 {r['bidNtceNo']} {r['bidNtceNm']}"
                     + (" (자격 문제 표시)" if r.get("verify", {}).get("issues") else "") for r in recs)


def verify_recommendations(state: ProposalState, config: RunnableConfig):
    """5단계 검증 에이전트: 상위 N건 재평가 → 4단계 순위와 맞춰 보는 자체 점검(judge)으로 통과/불통과 판정.
    불통과면 불통과 내용으로 비평 프롬프트를 만들어 verify_loop.pending에 넣는다 → Supervisor가 4단계로 되돌린다
    (최대 VERIFY_MAX_RETRY회). 재시도를 다 써도 불통과면 지금까지 불통과가 가장 적었던 결과를 쓴다."""
    recs, n = state["recommendations"], _top_n()
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
    branches, tot = state.get("rag_branches") or [], None
    if result is not None and len(branches) > 1:  # ToT: 4단계 순위안 중 하나를 고른다
        recs, verification, tot = pick_branch(branches, result, reason, n, dropped)
        verification = {**verification, "tot": tot}
    else:
        recs, verification = apply_verification(recs, result, reason, n, dropped)

    loop = state.get("verify_loop") or {"attempt": 0, "history": [], "best": None}
    attempt = loop["attempt"]
    checked = sum(bool(r.get("verify") and r.get("llm")) for r in recs)  # 4단계 LLM 순위와 검증이 둘 다 있는 공고
    fails = judge(recs) if checked else []
    status = "skipped" if not checked else "passed" if not fails else "failed"  # 룰 순위·검증 생략이면 비교할 게 없음
    best = loop.get("best")
    if best is None or len(fails) <= best["fails"]:
        best = {"attempt": attempt, "fails": len(fails), "recommendations": recs, "verification": verification}
    pending, restored = None, None
    if fails and attempt < settings.VERIFY_MAX_RETRY:
        texts = [f"[{f['kind']}] {f['text']}" for f in fails]
        pending = {"attempt": attempt + 1, "reasons": texts,
                   "prompt": llm.critique_prompt(attempt + 1, texts, _order_text(recs))}
    elif fails and best["attempt"] != attempt:  # 재시도 소진 → 불통과가 가장 적었던 회차로
        recs, verification, restored = best["recommendations"], best["verification"], best["attempt"]
    history = [*loop["history"], {"attempt": attempt, "status": status, "fails": len(fails)}]
    verification = {**verification, "judge": {
        "status": status, "attempt": attempt, "checked": checked, "fails": fails, "restored": restored,
        "next": "rag_recommend" if pending else "select_tender", "history": history}}
    save(config, state, "recommended", recommendations=recs)
    return {"recommendations": recs, "verification": verification,
            "verify_loop": {"attempt": attempt, "history": history, "best": best, "pending": pending}}
