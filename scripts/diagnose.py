"""추천 실패 원인 진단: 라벨 붙은 공고가 어느 단계에서 틀렸는지 가른다 (LangSmith 트레이스 1개/업체).

단계: 하드필터 → 하이브리드 검색(rule·bm25·dense RRF) → 리랭크(상위 15) → 상위 10 → RAG 순위 → 자격 검증(상위 N)
판정
  추천 공고  : 필터 제외 / 검색 누락(RRF > 리랭크 구간) / 리랭크 밀림(10위 안 → 밖) / 검색 순위 낮음 / RAG 순위 밀림 / 정상
  부적격 공고: 자격 검증 누락(상위 N에 issue 없이 남음 — 근거 청크에 조건 문장이 있었는지도 표시) / 정상(필터·검증이 걸러냄)

uv run python scripts/diagnose.py C005                       # 정답셋(verify_gold.json) 라벨 공고 전체
uv run python scripts/diagnose.py C006 --expect R26BK01501429:추천   # 정답셋에 없는 공고도 기대 라벨로 진단
uv run python scripts/diagnose.py C005 --no-llm              # RAG·검증 없이 검색·리랭크까지만
"""
import argparse
import json
import sys
from pathlib import Path

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.graphs.recommend import rerank_stage, rule_stage, search_stage  # noqa: E402
from backend.tools.loaders import load_company_file, load_region_limits, load_tenders  # noqa: E402
from backend.tools.search import RERANK_TOP, rank_table  # noqa: E402

GOLD = ROOT / "data" / "eval" / "verify_gold.json"
OUT_DIR = ROOT / "outputs" / "eval"


def company_path(cid: str) -> Path:
    for p in sorted(config.COMPANIES_DIR.glob("*.json")):
        if load_company_file(p)["업체ID"] == cid:
            return p
    raise SystemExit(f"업체 {cid} 없음")


def gold_labels(cid: str) -> dict[str, str]:
    g = json.loads(GOLD.read_text(encoding="utf-8")).get(cid, {})
    return {t["bidNtceNo"]: t["label"] for t in g.get("tenders", []) if t.get("label")}


def verdict(row: dict, label: str, n: int) -> str:
    top_k = config.TOP_K
    if label == "부적격":
        if row["filter"]:
            return "정상 (하드필터 제외)"
        if row["final"] is None or row["final"] > n:
            return "정상 (상위 N 밖)"
        if row["issues"]:
            return "정상 (검증이 issue로 뒤로 보냄)"
        if row["qual_clause"] is None:
            return "상위 N에 남음 (검증 미실행)"
        return "자격 검증 누락 (" + ("근거 청크에 조건 문장 있음 → LLM 오판" if row["qual_clause"]
                                 else "근거 청크에 조건 문장 없음 → 청크 선택 실패") + ")"
    if row["filter"]:
        return f"필터 제외 ({','.join(row['filter'])})"
    if row["rrf"] is None or row["rrf"] > RERANK_TOP:
        return f"검색 누락 (RRF {row['rrf']}위 > 리랭크 구간 {RERANK_TOP})"
    if row["top10"] is None:
        return ("리랭크 밀림" if row["rrf"] <= top_k else "검색 순위 낮음 (리랭크가 못 끌어올림)") + \
            f" (RRF {row['rrf']}위 → 리랭크 {row['rerank_after']}위)"
    if label == "추천" and row["final"] is not None and row["final"] > n:
        return f"RAG/검증 순위 밀림 (상위10 {row['top10']}위 → 최종 {row['final']}위)"
    return "정상"


@traceable(name="diagnose", run_type="chain", tags=["diagnose"])
def diagnose(cid: str, expect: dict[str, str], use_llm: bool) -> dict:
    from backend.agents.rag import rag_recommend
    from backend.agents.verify import _QUAL_TEXT, qualification_chunks, verify_recommendations

    if rt := get_current_run_tree():
        rt.metadata.update({"업체ID": cid})
    company = load_company_file(company_path(cid))
    tenders, n = load_tenders(), 5  # 화면 상위 5건 기준으로 판정
    by_no = {t["bidNtceNo"]: t for t in tenders}
    stage = rule_stage(tenders, company, load_region_limits())
    search = search_stage(stage["candidates"], by_no, company)
    ranks = rank_table(search)
    out = rerank_stage(search, stage["candidates"], by_no)
    top10 = {c["bidNtceNo"]: c["rank"] for c in out["top"]}
    after = {m["bidNtceNo"]: m["after"] for m in out["search"].get("moves") or []}
    excluded = {e["bidNtceNo"]: e["rules"] for e in stage["filter_log"]["excluded"]}

    final, issues, rag_rank, qual, top5 = {}, {}, {}, {}, []
    if use_llm and config.llm_enabled():
        state = {"company_profile": company, "top10": out["top"], "search": out["search"], "prefs": None}
        rag = rag_recommend(state)
        rag_rank = {r["bidNtceNo"]: r["rank"] for r in rag["recommendations"]}
        v = verify_recommendations({**state, "recommendations": rag["recommendations"]}, {})
        final = {r["bidNtceNo"]: r["rank"] for r in v["recommendations"]}
        issues = {r["bidNtceNo"]: r.get("verify", {}).get("issues", []) for r in v["recommendations"]}
        head = v["recommendations"][:n]
        top5 = [{"rank": r["rank"], "bidNtceNo": r["bidNtceNo"], "bidNtceNm": r["bidNtceNm"][:40],
                 "label": expect.get(r["bidNtceNo"]), "issues": len(issues[r["bidNtceNo"]])} for r in head]
        qual = {no: any(_QUAL_TEXT.search(ch["text"]) for ch in chs)
                for no, chs in qualification_chunks(head, by_no).items()}
    else:
        final = top10

    rows = []
    for no, label in expect.items():
        r = ranks.get(no, {})
        row = {"bidNtceNo": no, "bidNtceNm": by_no.get(no, {}).get("bidNtceNm", "(공고 없음)")[:40], "label": label,
               "filter": excluded.get(no, []), "rule": r.get("rule"), "bm25": r.get("bm25"), "dense": r.get("dense"),
               "rrf": r.get("rrf"), "rerank_after": after.get(no, r.get("rrf")), "top10": top10.get(no),
               "rag": rag_rank.get(no), "final": final.get(no),
               "issues": [i["text"] for i in issues.get(no, [])], "qual_clause": qual.get(no)}
        row["verdict"] = verdict(row, label, n)
        rows.append(row)
    result = {"업체ID": cid, "query": search["query"], "hard_pass": len(stage["candidates"]),
              "verify": "llm" if rag_rank else "skipped", "rows": rows, "top5": top5,
              "issues_in_top5": sum(t["issues"] > 0 for t in top5)}
    if rt:
        try:
            result["trace_url"] = rt.get_url()
        except Exception:  # 추적 꺼짐·키 없음
            pass
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("company", help="업체ID (예: C005)")
    p.add_argument("--expect", nargs="*", default=[], help="공고번호:라벨 (추천/적격-비추천/부적격)")
    p.add_argument("--no-llm", action="store_true", help="RAG·검증 생략")
    args = p.parse_args()
    expect = gold_labels(args.company) | dict(e.split(":", 1) for e in args.expect)
    res = diagnose(args.company, expect, not args.no_llm)

    print(f"[{res['업체ID']}] 하드필터 통과 {res['hard_pass']}건 · 검증 {res['verify']}\n질의: {res['query'][:120]}")
    cols = ("label", "filter", "rule", "bm25", "dense", "rrf", "rerank_after", "top10", "rag", "final")
    print("공고번호        " + " ".join(f"{c:>7}" for c in cols))
    for r in res["rows"]:
        print(f"{r['bidNtceNo']} " + " ".join(f"{str(r[c] if r[c] not in ([], None) else '-'):>7}" for c in cols)
              + f"  {r['bidNtceNm']}\n    → {r['verdict']}" + (f" | issues: {r['issues']}" if r["issues"] else ""))
    if res["top5"]:
        print(f"최종 상위 {len(res['top5'])}건 (자격 issue 있는 공고 {res['issues_in_top5']}건)")
        for t in res["top5"]:
            print(f"  {t['rank']}. {t['bidNtceNo']} [{t['label'] or '라벨 없음'}] issue {t['issues']} {t['bidNtceNm']}")
    if res.get("trace_url"):
        print(f"trace: {res['trace_url']}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"diagnose_{res['업체ID']}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
